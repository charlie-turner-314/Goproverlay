from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Tuple

import fitdecode  # type: ignore

from .datatypes import GpsSeries, Metric, SeriesBundle, TimeSeries
from .interpolation import hold_last_interpolate, nearest_index
from .units import ms_to_kmh, ms_to_pace_s_per_km
from ..utils.logging import get_logger


SEMICIRCLES_TO_DEGREES = 180.0 / 2**31


@dataclass
class FitData:
    series: SeriesBundle
    offset_seconds: float = 0.0

    @classmethod
    def from_fit_file(
        cls,
        path: Path,
        offset_seconds: float = 0.0,
        align_to_datetime: Optional[datetime] = None,
    ) -> "FitData":
        # Parse using fitdecode
        power: list[Optional[float]] = []
        speed: list[Optional[float]] = []
        cadence: list[Optional[float]] = []
        elevation: list[Optional[float]] = []
        lat: list[float] = []
        lon: list[float] = []
        timestamps: list[datetime] = []
        log = get_logger(__name__)
        with fitdecode.FitReader(str(path), check_crc=False) as reader:  # type: ignore
            for frame in reader:
                if not isinstance(frame, fitdecode.FitDataMessage):  # type: ignore
                    continue
                if frame.name != "record":
                    continue
                fget = frame.get_value  # type: ignore
                ts = fget("timestamp")
                if ts is None:
                    continue
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                timestamps.append(ts)

                p = fget("power") if frame.has_field("power") else None  # type: ignore
                s = fget("enhanced_speed") if frame.has_field("enhanced_speed") else (fget("speed") if frame.has_field("speed") else None)  # type: ignore
                c = fget("cadence") if frame.has_field("cadence") else None  # type: ignore
                a = (
                    fget("enhanced_altitude")
                    if frame.has_field("enhanced_altitude")
                    else (fget("altitude") if frame.has_field("altitude") else None)
                )  # type: ignore
                plat = fget("position_lat") if frame.has_field("position_lat") else None  # type: ignore
                plon = fget("position_long") if frame.has_field("position_long") else None  # type: ignore

                power.append(_safe_float(p))
                speed.append(_safe_float(s))
                cadence.append(_safe_float(c))
                elevation.append(_safe_float(a))
                if plat is not None and plon is not None:
                    lat.append(float(plat) * SEMICIRCLES_TO_DEGREES)
                    lon.append(float(plon) * SEMICIRCLES_TO_DEGREES)
                else:
                    lat.append(float("nan"))
                    lon.append(float("nan"))

        if not timestamps:
            return cls(series=SeriesBundle(), offset_seconds=offset_seconds)

        t0 = min(timestamps)
        # If align_to_datetime is provided, compute offset_seconds to align FIT to that absolute time.
        if align_to_datetime is not None:
            log.info(f"Aligning to: {align_to_datetime}")
            # We want series times to be (timestamp - align_to_datetime)
            offset_seconds = (align_to_datetime - t0).total_seconds()
            log.info(f"Offset (s): {str(offset_seconds)}")
        times = [(ts - t0).total_seconds() for ts in timestamps]
        # Apply offset: t_aligned = (ts - t0) - offset_seconds => equals (ts - align_to_datetime) when aligned
        times = [t - offset_seconds for t in times]

        # Filter out None to numeric by carrying NaNs? Simpler: keep as floats with None masked by previous
        # We'll create series only if any non-None present
        bundle = SeriesBundle()

        def filtered_series(vals):
            ts = []
            vs = []
            for ti, vi in zip(times, vals):
                if vi is None:
                    continue
                ts.append(float(ti))
                vs.append(float(vi))
            return TimeSeries(ts, vs) if ts else None

        bundle.power = filtered_series(power)
        bundle.speed = filtered_series(speed)
        bundle.cadence = filtered_series(cadence)
        bundle.elevation = filtered_series(elevation)

        # GPS: require both lat and lon and not NaN
        gps_t = []
        gps_lat = []
        gps_lon = []
        for ti, la, lo in zip(times, lat, lon):
            if la != la or lo != lo:  # NaN check
                continue
            gps_t.append(float(ti))
            gps_lat.append(float(la))
            gps_lon.append(float(lo))
        bundle.gps = GpsSeries(gps_t, gps_lat, gps_lon) if gps_t else None

        return cls(series=bundle, offset_seconds=offset_seconds)

    def is_empty(self) -> bool:
        s = self.series
        return not any(
            [
                s.power and not s.power.is_empty(),
                s.speed and not s.speed.is_empty(),
                s.cadence and not s.cadence.is_empty(),
                s.elevation and not s.elevation.is_empty(),
                s.gps and not s.gps.is_empty(),
            ]
        )

    def get_metric_value(self, metric: Metric, t: float) -> Optional[float]:
        s = self.series
        if metric == Metric.power and s.power:
            return hold_last_interpolate(t, s.power.t, s.power.v)
        if metric == Metric.speed and s.speed:
            v = hold_last_interpolate(t, s.speed.t, s.speed.v)
            return ms_to_kmh(v) if v is not None else None
        if metric == Metric.pace and s.speed:
            v = hold_last_interpolate(t, s.speed.t, s.speed.v)
            return ms_to_pace_s_per_km(v) if v is not None else None
        if metric == Metric.cadence and s.cadence:
            return hold_last_interpolate(t, s.cadence.t, s.cadence.v)
        if metric == Metric.elevation and s.elevation:
            return hold_last_interpolate(t, s.elevation.t, s.elevation.v)
        return None

    def get_position(self, t: float) -> Optional[Tuple[float, float]]:
        s = self.series
        if not s.gps or s.gps.is_empty():
            return None
        idx = nearest_index(t, s.gps.t)
        if idx is None:
            return None
        return s.gps.lat[idx], s.gps.lon[idx]


def _safe_float(v) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


        # removed legacy fallback: we now parse via fitdecode only
