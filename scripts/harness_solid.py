from __future__ import annotations

import math
from pathlib import Path
import os
import sys
from typing import List

import numpy as np
from PIL import Image

from moviepy import ColorClip

# Ensure local package is importable when running from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from goproverlay.core.datatypes import GpsSeries, SeriesBundle, TimeSeries, Metric
from goproverlay.core.fit_loader import FitData
from goproverlay.overlay.renderer import OverlayRenderer


def gen_timeseries(duration_s: float) -> List[float]:
    # Typical FIT record rate is 1 Hz
    n = int(math.floor(duration_s)) + 1
    return [float(i) for i in range(n)]


def build_fake_fit(duration_s: float = 15.0) -> FitData:
    t = gen_timeseries(duration_s)
    # Power around 200W +/- 50
    power = [200.0 + 50.0 * math.sin(2 * math.pi * (ti / 10.0)) for ti in t]
    # Speed in m/s, keep > 0
    speed = [3.0 + 2.0 * math.sin(2 * math.pi * (ti / 12.0)) + 0.05 * ti for ti in t]
    speed = [max(0.1, v) for v in speed]
    # Cadence ~ 85 +/- 5
    cadence = [85.0 + 5.0 * math.sin(2 * math.pi * (ti / 3.0)) for ti in t]
    # Elevation ~ 50m +/- 5m
    elevation = [50.0 + 5.0 * math.sin(2 * math.pi * (ti / 20.0)) for ti in t]

    # Simple GPS track: diagonal path
    lat0, lon0 = -27.49, 153.0
    lat = [lat0 + 0.0002 * (ti / duration_s) for ti in t]
    lon = [lon0 + 0.0003 * (ti / duration_s) for ti in t]

    bundle = SeriesBundle(
        power=TimeSeries(t, power),
        speed=TimeSeries(t, speed),
        cadence=TimeSeries(t, cadence),
        elevation=TimeSeries(t, elevation),
        gps=GpsSeries(t, lat, lon),
    )
    return FitData(series=bundle, offset_seconds=0.0)


def run_harness(
    out_path: Path = Path("harness_overlay.mp4"),
    frame_png: Path = Path("harness_frame.png"),
    duration: float = 10.0,
    fps: int = 30,
    size=(1280, 720),
):
    # Build fake fit data
    fit = build_fake_fit(duration)

    # Solid color background clip
    clip = ColorClip(size=size, color=(30, 30, 30)).with_duration(duration).with_fps(fps)

    # Choose up to 4 widgets
    metrics = [Metric.power, Metric.speed, Metric.gps, Metric.elevation]

    renderer = OverlayRenderer(
        video_path=Path("_in_memory.mp4"),  # unused for overlay_clip
        fit_data=fit,
        metrics=metrics,
        theme="dark",
        font_path=None,
    )

    out = renderer.overlay_clip(clip)

    # Write a short video
    out.write_videofile(
        str(out_path),
        codec="libx264",
        fps=fps,
        audio=False,
        threads=2,
    )

    # Save a representative frame for visual verification
    frame = out.get_frame(min(duration * 0.66, duration - 1e-3))
    Image.fromarray(frame).save(frame_png)
    print(f"Wrote {out_path} and preview frame {frame_png}")


if __name__ == "__main__":
    run_harness()
