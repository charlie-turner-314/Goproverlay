from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional


def _parse_iso8601_utc(s: str) -> Optional[datetime]:
    try:
        s = s.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None


def get_video_start_utc(path: Path) -> Optional[datetime]:
    """Return the wall-clock UTC datetime for the first frame of the video.

    Uses ffprobe to read MP4 metadata. Formula:
    video_start = creation_time + start_time_seconds
    Falls back to None if required fields are missing.
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
        out = subprocess.check_output(cmd)
        data = json.loads(out.decode("utf-8", errors="ignore"))
        fmt = data.get("format", {})
        tags = fmt.get("tags", {})
        creation = tags.get("creation_time")
        if not creation:
            # try streams
            streams = data.get("streams", [])
            for st in streams:
                st_tags = st.get("tags", {})
                if "creation_time" in st_tags:
                    creation = st_tags["creation_time"]
                    break
        if not creation:
            return None
        creation_dt = _parse_iso8601_utc(creation)
        if creation_dt is None:
            return None
        start_time = fmt.get("start_time")
        start_sec = float(start_time) if start_time is not None else 0.0
        return creation_dt + timedelta(seconds=start_sec)
    except Exception:
        return None

