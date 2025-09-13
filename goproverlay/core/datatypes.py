from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class Metric(str, Enum):
    power = "power"
    speed = "speed"  # km/h
    pace = "pace"  # min/km
    elevation = "elevation"  # meters
    cadence = "cadence"  # rpm
    gps = "gps"  # track widget


@dataclass
class TimeSeries:
    """Generic timeseries of numeric values keyed by seconds since start."""

    t: List[float]
    v: List[float]

    def is_empty(self) -> bool:
        return len(self.t) == 0


@dataclass
class GpsSeries:
    t: List[float]
    lat: List[float]
    lon: List[float]

    def is_empty(self) -> bool:
        return len(self.t) == 0


@dataclass
class SeriesBundle:
    power: Optional[TimeSeries] = None
    speed: Optional[TimeSeries] = None
    cadence: Optional[TimeSeries] = None
    elevation: Optional[TimeSeries] = None
    gps: Optional[GpsSeries] = None

