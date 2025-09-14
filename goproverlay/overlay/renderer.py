from __future__ import annotations

from dataclasses import dataclass
import os
import platform
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from moviepy import VideoFileClip
from PIL import Image

from ..core.datatypes import Metric
from ..core.fit_loader import FitData
from ..core.time_sync import build_video_to_utc_mapper, TimeSyncedFitData
from ..utils.logging import get_logger
from .widgets.metric import MetricWidget
from .widgets.track import TrackWidget
from .widgets.elevation_plot import ElevationPlotWidget
from .widgets.gauge import PowerGaugeWidget, SpeedGaugeWidget


log = get_logger(__name__)


THEMES = {
    "dark": {
        "bg": (0, 0, 0, 140),
        "fg": (255, 255, 255, 255),
        "accent": (120, 200, 255, 255),
        "muted": (200, 200, 200, 200),
        "track": (120, 200, 255, 255),
    },
    "light": {
        "bg": (255, 255, 255, 180),
        "fg": (10, 10, 10, 255),
        "accent": (0, 90, 200, 255),
        "muted": (60, 60, 60, 200),
        "track": (0, 90, 200, 255),
    },
}


@dataclass
class OverlayRenderer:
    video_path: Path
    fit_data: FitData
    metrics: List[Metric]
    theme: str = "dark"
    font_path: Optional[Path] = None
    power_zones: Optional[List[float]] = None
    ftp: Optional[float] = None
    speed_max_kmh: Optional[float] = None
    gpx_path: Optional[Path] = None
    power_avg_secs: Optional[float] = None
    # no caching fields

    def _build_widgets(
        self, video_size: Tuple[int, int], duration: Optional[float] = None
    ):
        theme = THEMES.get(self.theme, THEMES["dark"])
        W, H = video_size

        # Widget size proportional to video size
        w = int(W * 0.28)
        h = int(H * 0.18)
        margin = int(min(W, H) * 0.03)

        # Corner positions
        boxes = [
            (margin, margin, margin + w, margin + h),  # top-left
            (W - margin - w, margin, W - margin, margin + h),  # top-right
            (margin, H - margin - h, margin + w, H - margin),  # bottom-left
            (W - margin - w, H - margin - h, W - margin, H - margin),  # bottom-right
        ]

        widgets = []
        # Precompute speed max if not provided
        speed_max_kmh = self.speed_max_kmh
        if (
            speed_max_kmh is None
            and self.fit_data.series.speed
            and self.fit_data.series.speed.v
        ):
            kmh_vals = []
            if duration is not None:
                # Consider only the part of the activity shown in this clip
                s = self.fit_data.series.speed
                for ti, vi in zip(s.t, s.v):
                    if 0.0 <= ti <= duration and vi is not None:
                        kmh_vals.append(vi * 3.6)
            if not kmh_vals:
                kmh_vals = [
                    v * 3.6 for v in self.fit_data.series.speed.v if v is not None
                ]
            if kmh_vals:
                mmax = max(kmh_vals)
                # 1.1x the maximum displayed during video
                speed_max_kmh = max(15.0, ((mmax * 1.1 + 4.9) // 5) * 5)
        if speed_max_kmh is None:
            speed_max_kmh = 50.0

        # Compute power zones if not provided
        power_zones = self.power_zones
        if power_zones is None:
            # Try ftp-based defaults if ftp provided
            if self.ftp:
                ftp = float(self.ftp)
                # 5-zone scheme: 55%, 75%, 90%, 105% of FTP
                power_zones = [
                    0.55 * ftp,
                    0.75 * ftp,
                    0.90 * ftp,
                    1.05 * ftp,
                    1.20 * ftp,
                ]
            elif self.fit_data.series.power and self.fit_data.series.power.v:
                pmax = max(
                    [pv for pv in self.fit_data.series.power.v if pv is not None]
                    + [200.0]
                )
                # Scale by percent of max as fallback
                power_zones = [
                    0.55 * pmax,
                    0.75 * pmax,
                    0.90 * pmax,
                    1.05 * pmax,
                    1.20 * pmax,
                ]
            else:
                power_zones = [100, 150, 200, 250, 300]

        for i, m in enumerate(self.metrics):
            box = boxes[i]
            if m == Metric.gps:
                widgets.append(
                    TrackWidget(
                        self.fit_data,
                        box=box,
                        theme=theme,
                        duration=duration,
                    )
                )
            elif m == Metric.speed:
                widgets.append(
                    SpeedGaugeWidget(
                        fit_data=self.fit_data,
                        box=box,
                        theme=theme,
                        font_path=self.font_path,
                        max_kmh=float(speed_max_kmh),
                    )
                )
            elif m == Metric.power:
                widgets.append(
                    PowerGaugeWidget(
                        fit_data=self.fit_data,
                        box=box,
                        theme=theme,
                        font_path=self.font_path,
                        zone_bounds=list(power_zones),
                        avg_window_s=float(self.power_avg_secs)
                        if self.power_avg_secs is not None and self.power_avg_secs > 0
                        else None,
                    )
                )
            else:
                widgets.append(
                    ElevationPlotWidget(
                        fit_data=self.fit_data,
                        box=box,
                        theme=theme,
                        duration=duration,
                    )
                    if m == Metric.elevation
                    else MetricWidget(
                        metric=m,
                        fit_data=self.fit_data,
                        box=box,
                        theme=theme,
                        font_path=self.font_path,
                    )
                )
        return widgets

    def _frame_with_overlay(self, base_frame: np.ndarray, t: float) -> np.ndarray:
        """Draw overlays directly onto the provided RGB frame in-place.

        Avoids numpy↔PIL roundtrips by wrapping the frame's buffer with
        Image.frombuffer and drawing onto it, returning the original array.
        """
        if (
            base_frame.dtype is not np.uint8
            or base_frame.ndim != 3
            or base_frame.shape[2] != 3
            or not base_frame.flags.c_contiguous
            or not base_frame.flags.writeable
        ):
            base_frame = np.ascontiguousarray(base_frame, dtype=np.uint8)

        h, w = int(base_frame.shape[0]), int(base_frame.shape[1])
        base_img = Image.frombuffer("RGB", (w, h), base_frame, "raw", "RGB", 0, 1)
        for wdg in self._widgets:
            wdg.draw_onto(base_img, t)
        # Safety: some environments may not reflect in-place edits; return a materialized array
        # to guarantee overlays are present.
        return np.array(base_img)

    def overlay_clip(self, clip):
        """Return a new MoviePy clip with overlays applied to each frame."""
        log.info(
            "Overlay init: building time mapper for video=%s (duration=%.3fs)",
            str(self.video_path),
            float(getattr(clip, "duration", 0.0) or 0.0),
        )
        # Build a time mapper: prefer in-video GPMF, else GPX, else video metadata
        mapper, strategy = build_video_to_utc_mapper(self.video_path, self.gpx_path)
        log.info("Time sync strategy: %s", strategy)
        if self.fit_data.t0_utc is not None:
            log.info(
                "FIT reference t0_utc=%s, offset_seconds=%.3f",
                self.fit_data.t0_utc.isoformat(),
                float(self.fit_data.offset_seconds),
            )
        # Log sample mappings
        try:
            dur = float(getattr(clip, "duration", 0.0) or 0.0)
            u0 = mapper(0.0)
            u1 = mapper(dur) if dur > 0 else None
            log.info(
                "Time map samples: t=0.000s -> %s; t=%.3fs -> %s",
                u0.isoformat() if u0 else "None",
                dur,
                u1.isoformat() if u1 else "None",
            )
        except Exception:
            pass

        # Wrap FIT data with time-synced adapter
        synced_fit = TimeSyncedFitData(self.fit_data, mapper)

        # Build widgets with the synced fit. We avoid duration-based filtering in widgets
        # to prevent mismatches when timelines differ; pass None for duration.
        log.info("Overlay init: building widgets for metrics=%s", ",".join([m.value for m in self.metrics]))
        self._widgets = self._build_widgets((clip.w, clip.h), None)
        # Patch widgets' fit references where applicable
        patched = 0
        for w in self._widgets:
            if hasattr(w, "fit"):
                try:
                    w.fit = synced_fit  # type: ignore
                    patched += 1
                except Exception:
                    pass
        log.info("Overlay init: time-synced FitData injected into %d widgets", patched)

        original_get_frame = clip.get_frame

        def new_get_frame(t):
            frame = original_get_frame(t)
            return self._frame_with_overlay(frame, t)

        return clip.with_updated_frame_function(new_get_frame)

    def render_to(self, output_path: Path) -> None:
        clip = VideoFileClip(str(self.video_path))
        out = self.overlay_clip(clip)
        # Write output
        # Prefer hardware encoder on macOS (VideoToolbox), fallback to libx264
        encoder = os.environ.get("GOPROVERLAY_ENCODER")
        if not encoder:
            encoder = "h264_videotoolbox" if platform.system() == "Darwin" else "libx264"

        def _write(codec: str):
            # Configure bitrate for hardware encoder; VideoToolbox needs an explicit positive bitrate.
            bitrate = None
            ffmpeg_params = []
            threads = 4
            if codec == "h264_videotoolbox":
                bitrate = os.environ.get("GOPROVERLAY_BITRATE", "8000k")
                threads = None  # not applicable for VideoToolbox
                # Set a safe profile
                ffmpeg_params += ["-profile:v", "main"]

            out.write_videofile(
                str(output_path),
                codec=codec,
                audio_codec="aac",
                temp_audiofile=str(output_path.with_suffix(".temp-audio.m4a")),
                remove_temp=True,
                threads=threads,
                pixel_format="yuv420p",
                bitrate=bitrate,
                ffmpeg_params=ffmpeg_params or None,
            )

        try:
            _write(encoder)
        except Exception as e:
            log.warning(
                "Hardware encoder '%s' failed (%s). Falling back to libx264.",
                encoder,
                e,
            )
            _write("libx264")
