from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from moviepy import ColorClip

# Ensure local package import for dev
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from goproverlay.core.datatypes import Metric
from goproverlay.core.fit_loader import FitData
from goproverlay.overlay.renderer import OverlayRenderer


def main():
    p = argparse.ArgumentParser(
        description="Render overlays onto a solid clip using a real FIT file"
    )
    p.add_argument("fit", type=Path, help="Path to FIT file")
    p.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration seconds (default: infer from FIT range or 10s)",
    )
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--size", type=str, default="1280x720", help="WxH, e.g. 1920x1080")
    p.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="FIT offset seconds (FIT ahead if positive)",
    )
    p.add_argument("--out", type=Path, default=Path("harness_fit_overlay.mp4"))
    p.add_argument(
        "--ftp",
        type=float,
        default=None,
        help="Functional Threshold Power (W) for power zones",
    )
    p.add_argument(
        "--power-avg",
        dest="power_avg",
        type=int,
        default=None,
        help="Windowed average seconds for power gauge (e.g. 3)",
    )

    args = p.parse_args()

    w, h = map(int, args.size.lower().split("x"))

    fit = FitData.from_fit_file(args.fit, offset_seconds=args.offset)
    if not fit or fit.is_empty():
        print("No usable data in FIT.")
        sys.exit(1)

    # Infer duration if not provided
    dur = args.duration
    if dur is None:
        series = [
            fit.series.power.t if fit.series.power else [],
            fit.series.speed.t if fit.series.speed else [],
            fit.series.cadence.t if fit.series.cadence else [],
            fit.series.elevation.t if fit.series.elevation else [],
            fit.series.gps.t if fit.series.gps else [],
        ]
        tmax = max([max(s) if s else 0.0 for s in series] + [10.0])
        dur = min(max(5.0, tmax), 30.0)  # keep harness short

    clip = (
        ColorClip(size=(w, h), color=(30, 30, 30)).with_duration(dur).with_fps(args.fps)
    )

    metrics = [Metric.power, Metric.speed, Metric.gps, Metric.elevation]
    renderer = OverlayRenderer(
        video_path=Path("_unused.mp4"),
        fit_data=fit,
        metrics=metrics,
        theme="dark",
        font_path=None,
        ftp=args.ftp,
        power_avg_secs=float(args.power_avg) if args.power_avg else None,
    )

    out = renderer.overlay_clip(clip)
    out.write_videofile(
        str(args.out), codec="libx264", fps=args.fps, audio=False, threads=2
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
