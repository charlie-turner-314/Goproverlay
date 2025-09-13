from __future__ import annotations

from typing import Optional


def ms_to_kmh(ms: float) -> float:
    return ms * 3.6


def ms_to_pace_s_per_km(ms: float) -> Optional[float]:
    if ms <= 0:
        return None
    return 1000.0 / ms  # seconds per km


def format_speed_kmh(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v:.1f} km/h"


def format_pace_s_per_km(sec: Optional[float]) -> str:
    if sec is None or sec == float("inf"):
        return "—"
    minutes = int(sec // 60)
    seconds = int(round(sec % 60))
    if seconds == 60:
        minutes += 1
        seconds = 0
    return f"{minutes}:{seconds:02d} /km"


def format_power_w(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{int(round(v))} W"


def format_cadence_rpm(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{int(round(v))} rpm"


def format_elevation_m(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{int(round(v))} m"

