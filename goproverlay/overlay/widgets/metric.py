from __future__ import annotations

from pathlib import Path
import os
from typing import Dict, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from ...core.datatypes import Metric
from ...core.fit_loader import FitData
from ...core.units import (
    format_cadence_rpm,
    format_elevation_m,
    format_pace_s_per_km,
    format_power_w,
    format_speed_kmh,
)
from .base import Widget, draw_rounded_panel


LABELS = {
    Metric.power: "Power",
    Metric.speed: "Speed",
    Metric.pace: "Pace",
    Metric.elevation: "Elevation",
    Metric.cadence: "Cadence",
}


class MetricWidget(Widget):
    def __init__(
        self,
        metric: Metric,
        fit_data: FitData,
        box: Tuple[int, int, int, int],
        theme: Dict,
        font_path: Optional[Path] = None,
    ):
        super().__init__(box, theme)
        self.metric = metric
        self.fit = fit_data
        self.font_path = str(font_path) if font_path else self._resolve_font_path()
        self._font_cache = {}

    def _resolve_font_path(self) -> Optional[str]:
        # Prefer PIL DejaVu, then common macOS fonts, then generic names
        try:
            import PIL
            candidates = [
                Path(PIL.__file__).parent / "fonts" / "DejaVuSans.ttf",
                Path(PIL.__file__).parent / "DejaVuSans.ttf",
            ]
            for c in candidates:
                if c.exists():
                    return str(c)
        except Exception:
            pass
        mac_candidates = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttf",
            "/Library/Fonts/Arial.ttf",
            "/Library/Fonts/Helvetica.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
        ]
        for p in mac_candidates:
            if os.path.exists(p):
                return p
        # Last resort: let PIL search by name (may fail and fall back to bitmap)
        return "DejaVuSans.ttf"

    def _get_font(self, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        key = (size, bold)
        if key in self._font_cache:
            return self._font_cache[key]
        try:
            if self.font_path:
                font = ImageFont.truetype(self.font_path, size=size)
            else:
                font = ImageFont.truetype("DejaVuSans.ttf", size=size)
        except Exception:
            # Bitmap fallback (fixed size). Drawing code sizes via bbox; size param ignored.
            font = ImageFont.load_default()
        self._font_cache[key] = font
        return font

    def _fit_font_size(self, draw: ImageDraw.ImageDraw, text: str, max_w: int, max_h: int, max_pt: int) -> int:
        # Find the largest font size (<= max_pt) that fits in max_w x max_h
        lo, hi = 6, max_pt
        best = lo
        while lo <= hi:
            mid = (lo + hi) // 2
            font = self._get_font(mid)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            if tw <= max_w and th <= max_h:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return max(6, best)

    def _format_value(self, value: Optional[float]) -> str:
        if self.metric == Metric.power:
            return format_power_w(value)
        if self.metric == Metric.speed:
            return format_speed_kmh(value)
        if self.metric == Metric.pace:
            return format_pace_s_per_km(value)
        if self.metric == Metric.elevation:
            return format_elevation_m(value)
        if self.metric == Metric.cadence:
            return format_cadence_rpm(value)
        return "—"

    def render(self, t: float) -> Image.Image:
        x1, y1, x2, y2 = self.box
        w, h = x2 - x1, y2 - y1

        panel = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)

        pad = int(min(w, h) * 0.08)
        inner_w = max(1, w - 2 * pad)
        inner_h = max(1, h - 2 * pad)

        # Label area (top)
        label = LABELS.get(self.metric, str(self.metric.value).title())
        label_h = int(inner_h * 0.28)
        label_max_pt = int(h * 0.22)
        label_pt = self._fit_font_size(draw, label, inner_w, label_h, label_max_pt)
        label_font = self._get_font(label_pt)
        lbbox = draw.textbbox((0, 0), label, font=label_font)
        ltw, lth = lbbox[2] - lbbox[0], lbbox[3] - lbbox[1]
        draw.text((pad, pad), label, fill=self.theme["muted"], font=label_font)

        # Value area (remaining)
        val = self.fit.get_metric_value(self.metric, t)
        value_str = self._format_value(val)
        value_max_h = int(inner_h * 0.62)
        value_max_pt = int(h * 0.5)
        value_pt = self._fit_font_size(draw, value_str, inner_w, value_max_h, value_max_pt)
        value_font = self._get_font(value_pt)
        vbox = draw.textbbox((0, 0), value_str, font=value_font)
        vtw, vth = vbox[2] - vbox[0], vbox[3] - vbox[1]

        vx = pad + (inner_w - vtw) // 2
        vy = pad + label_h + max(0, (inner_h - label_h - vth) // 2)
        draw.text((vx, vy), value_str, fill=self.theme["fg"], font=value_font)

        return panel
