from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
import subprocess
import json
from pathlib import Path
from typing import Callable, Optional, Tuple

from .fit_loader import FitData
from .gpx_loader import get_gpx_time_bounds
from .video_meta import get_video_start_utc
from ..utils.logging import get_logger


log = get_logger(__name__)


def _build_mapper_from_start(start_utc: Optional[datetime]) -> Callable[[float], Optional[datetime]]:
    if start_utc is None:
        return lambda t: None

    start_utc = start_utc.astimezone(timezone.utc)

    def map_t(t: float) -> Optional[datetime]:
        try:
            return start_utc + timedelta(seconds=float(t))
        except Exception:
            return None

    return map_t


def _try_build_gpmf_mapper(video: Path) -> Optional[Callable[[float], Optional[datetime]]]:
    """Build t_video→UTC mapper using GoPro GPMF (gpmd) and packet PTS via ffprobe.

    Pure-FFprobe + lightweight parser (no third-party deps):
    - find gpmd data stream index with ffprobe
    - list packets for that stream with PTS times and raw data (-show_packets -show_data)
    - scan packet payloads for GPSU entries and parse their UTC timestamps
    - pair packet PTS (seconds) with UTC; interpolate by nearest neighbor
    """
    # 1) Find gpmd stream index (small JSON – safe to load fully)
    try:
        probe = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_streams",
                str(video),
            ]
        )
        streams = json.loads(probe.decode("utf-8", "ignore")).get("streams", [])
    except Exception as e:
        log.info("GPMF sync: ffprobe failed: %s", e)
        return None

    gpmd_idx = None
    for st in streams:
        if st.get("codec_type") != "data":
            continue
        tag = (st.get("codec_tag_string") or "").lower()
        name = (st.get("codec_name") or "").lower()
        if tag == "gpmd" or "gopro" in name or "gpmf" in name:
            gpmd_idx = st.get("index")
            break
    if gpmd_idx is None:
        log.info("GPMF sync: no gpmd data stream found; skipping")
        return None

    # 2) Stream packets with PTS and data (memory-safe incremental parse)
    # Limit anchors to ~1 per second by default to keep arrays small
    try:
        step_sec = float(os.environ.get("GOPROVERLAY_GPMF_STEP_SEC", "1.0"))
        if not (step_sec > 0):
            step_sec = 1.0
    except Exception:
        step_sec = 1.0

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_packets",
        "-show_data",
        "-print_format",
        "default",  # line-based blocks; avoids huge JSON materialization
        "-show_entries",
        "packet=stream_index,pts_time,data",
        "-select_streams",
        "d",  # only data streams; we'll still filter by index
        str(video),
    ]
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        )
    except Exception as e:
        log.info("GPMF sync: ffprobe packets (stream) failed: %s", e)
        return None

    # 3) Parse GPSU per packet and build mapping incrementally
    t_video: list[float] = []
    t_utc: list[datetime] = []
    last_kept_t: Optional[float] = None

    def parse_hex_or_b64(hex_str: str) -> bytes:
        try:
            return bytes.fromhex(hex_str)
        except Exception:
            try:
                import base64
                return base64.b64decode(hex_str)
            except Exception:
                return b""

    def extract_gpsu_ascii(buf: bytes) -> Optional[str]:
        # Recursive scan for GPSU in nested GPMF KLV structure
        try:
            n = len(buf)
            off = 0
            while off + 8 <= n:
                key = buf[off : off + 4]
                typech = buf[off + 4]
                struct_sz = buf[off + 5]
                repeat = int.from_bytes(buf[off + 6 : off + 8], "big", signed=False)
                off += 8
                size = int(struct_sz) * int(repeat)
                if size < 0 or off + size > n:
                    break
                data = buf[off : off + size]
                # 32-bit align
                off = ((off + size + 3) // 4) * 4

                if key == b"GPSU":
                    # Two common encodings: type 'U' (compact) or 'c' (ASCII)
                    try:
                        if typech == ord('U'):
                            x = data.decode('latin1', 'ignore')
                            year = "20" + x[:2]
                            month = x[2:4]
                            day = x[4:6]
                            hours = x[6:8]
                            mins = x[8:10]
                            seconds = x[10:].strip()
                            return f"{year}-{month}-{day} {hours}:{mins}:{seconds}"
                        else:
                            s = data.split(b"\x00", 1)[0]
                            return s.decode("ascii", "ignore").strip()
                    except Exception:
                        return None

                # Nested structure (type 0) contains inner KLVs
                if typech == 0 and data:
                    inner = extract_gpsu_ascii(data)
                    if inner:
                        return inner
        except Exception:
            return None
        return None

    def parse_utc(s: str) -> Optional[datetime]:
        s = s.strip()
        # Try common formats seen in GoPro: with or without 'T'/'Z', with micros or millis
        fmts = [
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
        ]
        for f in fmts:
            try:
                dt = datetime.strptime(s, f)
                return dt.replace(tzinfo=timezone.utc)
            except Exception:
                pass
        return None

    # Stream parse state
    in_packet = False
    pkt_fields: dict[str, str] = {}
    line_count = 0
    kept = 0
    scanned = 0

    def handle_packet(fields: dict[str, str]):
        nonlocal last_kept_t, kept, scanned
        scanned += 1
        try:
            sidx = fields.get("stream_index")
            if sidx is None or int(sidx) != gpmd_idx:
                return
        except Exception:
            return
        pts_s = fields.get("pts_time")
        data_hex = fields.get("data")
        if not pts_s or not data_hex:
            return
        try:
            tv = float(pts_s)
        except Exception:
            return
        # Downsample by time step
        if last_kept_t is not None and (tv - last_kept_t) < step_sec:
            return
        raw = parse_hex_or_b64(data_hex)
        if not raw:
            return
        ts_ascii = extract_gpsu_ascii(raw)
        if not ts_ascii:
            return
        tu = parse_utc(ts_ascii)
        if tu is None:
            return
        t_video.append(tv)
        t_utc.append(tu)
        last_kept_t = tv
        kept += 1
        # Periodic log every ~1000 kept anchors
        if kept % 1000 == 0:
            try:
                log.info("GPMF sync: kept %d anchors (last t=%.3fs %s)", kept, tv, tu.strftime("%H:%M:%S"))
            except Exception:
                pass

    # Read stdout line-by-line
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line_count += 1
            line = line.rstrip("\n")
            if line == "[PACKET]":
                in_packet = True
                pkt_fields = {}
                continue
            if line == "[/PACKET]":
                if in_packet:
                    handle_packet(pkt_fields)
                in_packet = False
                pkt_fields = {}
                continue
            if not in_packet:
                continue
            # key=value lines
            if "=" in line:
                k, v = line.split("=", 1)
                pkt_fields[k.strip()] = v.strip()
    finally:
        try:
            proc.stdout and proc.stdout.close()
        except Exception:
            pass
        try:
            proc.stderr and proc.stderr.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=1.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    if len(t_video) < 2:
        log.info("GPMF sync: insufficient GPSU samples with PTS; skipping")
        return None

    # Sort and build nearest-neighbor mapper
    pairs = sorted(zip(t_video, t_utc), key=lambda x: x[0])
    t_video = [p[0] for p in pairs]
    t_utc = [p[1] for p in pairs]

    from bisect import bisect_left

    def map_t(t: float) -> Optional[datetime]:
        i = bisect_left(t_video, float(t))
        if i <= 0:
            return t_utc[0]
        if i >= len(t_video):
            return t_utc[-1]
        # pick closer neighbor
        dl = t - t_video[i - 1]
        dr = t_video[i] - t
        return t_utc[i - 1] if dl <= dr else t_utc[i]

    log.info(
        "GPMF sync: using %d packet anchors (pts→UTC): first %.3fs→%s, last %.3fs→%s",
        len(t_video),
        t_video[0],
        t_utc[0].isoformat(),
        t_video[-1],
        t_utc[-1].isoformat(),
    )
    return map_t


def build_video_to_utc_mapper(
    video: Path, gpx: Optional[Path]
) -> Tuple[Callable[[float], Optional[datetime]], str]:
    """Return a function mapping video-time (s) to UTC and a strategy label.

    Strategy order:
    - GPMF (gpmd) if available (requires optional gopro-telemetry)
    - GPX start time (if provided)
    - Video metadata creation_time + start_time
    """
    # Try GPMF
    mapper = _try_build_gpmf_mapper(video)
    if mapper is not None:
        return mapper, "gpmf"

    # Fallback to GPX start
    if gpx is not None:
        start, end = get_gpx_time_bounds(gpx)
        if start is not None:
            log.info(
                "Time sync: using GPX start as reference (start=%s, end=%s)",
                start.isoformat(),
                end.isoformat() if end else "unknown",
            )
            return _build_mapper_from_start(start), "gpx"

    # Fallback to video metadata
    start = get_video_start_utc(video)
    if start is not None:
        log.info("Time sync: using video metadata start as reference")
        return _build_mapper_from_start(start), "video_meta"

    log.warning("Time sync: no reference timestamps available; using base FIT timeline")
    return (lambda t: None), "none"


class TimeSyncedFitData:
    """Adapter around FitData that maps video-time -> FIT-time using UTC mapping.

    - map_video_to_utc: callable returning absolute UTC for a given video t.
    - For each query, we convert UTC to FIT-relative seconds using fit.t0_utc,
      then select the closest FIT record for that series.
    """

    def __init__(self, base: FitData, map_video_to_utc: Callable[[float], Optional[datetime]]):
        self._base = base
        self._map = map_video_to_utc
        self.series = base.series
        self.offset_seconds = base.offset_seconds
        self.t0_utc = base.t0_utc
        self._warn_no_map_logged = False

    def _to_fit_seconds(self, t_video: float) -> Optional[float]:
        if self.t0_utc is None:
            return None
        utc = self._map(float(t_video))
        if utc is None:
            return None
        return (utc - self.t0_utc).total_seconds() - float(self.offset_seconds)

    # Public API used by widgets
    def get_metric_value(self, metric, t: float):
        from .datatypes import Metric as M
        from .interpolation import nearest_index

        tv = self._to_fit_seconds(t)
        if tv is None:
            # Fallback to base timeline (assume already aligned)
            if not self._warn_no_map_logged:
                log = get_logger(__name__)
                log.info("Time sync: no mapper result; using base FIT timeline for lookups")
                self._warn_no_map_logged = True
            return self._base.get_metric_value(metric, t)

        s = self._base.series
        if metric == M.power and s.power:
            from .units import ms_to_kmh  # not used here, but keep symmetry
            idx = nearest_index(tv, s.power.t)
            return s.power.v[idx] if idx is not None else None
        if metric == M.speed and s.speed:
            idx = nearest_index(tv, s.speed.t)
            v = s.speed.v[idx] if idx is not None else None
            from .units import ms_to_kmh
            return ms_to_kmh(v) if v is not None else None
        if metric == M.pace and s.speed:
            idx = nearest_index(tv, s.speed.t)
            v = s.speed.v[idx] if idx is not None else None
            from .units import ms_to_pace_s_per_km
            return ms_to_pace_s_per_km(v) if v is not None else None
        if metric == M.cadence and s.cadence:
            idx = nearest_index(tv, s.cadence.t)
            return s.cadence.v[idx] if idx is not None else None
        if metric == M.elevation and s.elevation:
            idx = nearest_index(tv, s.elevation.t)
            return s.elevation.v[idx] if idx is not None else None
        return None

    def get_position(self, t: float):
        from .interpolation import nearest_index

        tv = self._to_fit_seconds(t)
        if tv is None:
            if not self._warn_no_map_logged:
                log = get_logger(__name__)
                log.info("Time sync: no mapper result; using base FIT timeline for GPS position")
                self._warn_no_map_logged = True
            return self._base.get_position(t)
        gps = self._base.series.gps
        if not gps or gps.is_empty():
            return None
        idx = nearest_index(tv, gps.t)
        if idx is None:
            return None
        return gps.lat[idx], gps.lon[idx]
