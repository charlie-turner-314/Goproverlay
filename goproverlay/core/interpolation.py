from __future__ import annotations

from bisect import bisect_left
from typing import List, Optional, Tuple


def hold_last_interpolate(t_query: float, t: List[float], v: List[float]) -> Optional[float]:
    """Return the last known value at or before t_query. None if no data yet."""
    if not t:
        return None
    i = bisect_left(t, t_query)
    if i == 0:
        return None
    return v[i - 1]


def nearest_index(t_query: float, t: List[float]) -> Optional[int]:
    if not t:
        return None
    i = bisect_left(t, t_query)
    if i <= 0:
        return 0
    if i >= len(t):
        return len(t) - 1
    # choose closer neighbor
    if abs(t[i] - t_query) < abs(t[i - 1] - t_query):
        return i
    return i - 1

