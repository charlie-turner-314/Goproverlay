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
        # base_frame is HxWx3 RGB array uint8
        if base_frame.dtype != np.uint8:
            base_frame = base_frame.astype(np.uint8)
        base_img = Image.fromarray(base_frame)
        for w in self._widgets:
            w.draw_onto(base_img, t)
        return np.array(base_img)

    def overlay_clip(self, clip):
        """Return a new MoviePy clip with overlays applied to each frame."""
        self._widgets = self._build_widgets(
            (clip.w, clip.h), getattr(clip, "duration", None)
        )

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
