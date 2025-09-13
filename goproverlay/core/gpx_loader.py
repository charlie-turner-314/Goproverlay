from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple


def _parse_iso8601_utc(s: str) -> Optional[datetime]:
    try:
        s = s.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def get_gpx_time_bounds(path: Path) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Return (start_utc, end_utc) from a GPX file's track points.

    Scans all <trkpt><time> entries and returns the min/max as timezone-aware UTC.
    Returns (None, None) if no timestamps are found.
    """
    try:
        tree = ET.parse(str(path))
        root = tree.getroot()
        # GPX namespace handling
        ns = {
            "gpx": root.tag.split("}")[0].strip("{") if "}" in root.tag else "",
        }
        times = []
        # Try common path: trk > trkseg > trkpt > time
        # Support both namespaced and non-namespaced tags
        def iter_time_elements():
            # Namespaced search
            if ns["gpx"]:
                for el in root.findall(".//gpx:trkpt/gpx:time", ns):
                    yield el
            # Fallback non-namespaced
            for el in root.findall(".//trkpt/time"):
                yield el

        for el in iter_time_elements():
            if el is None or not el.text:
                continue
            dt = _parse_iso8601_utc(el.text)
            if dt is not None:
                times.append(dt)
        if not times:
            return None, None
        return min(times), max(times)
    except Exception:
        return None, None

