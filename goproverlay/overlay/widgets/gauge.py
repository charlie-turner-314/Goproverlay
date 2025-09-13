from __future__ import annotations

"""
Gauge widgets for speed and power.

Angle model used by Pillow (ImageDraw.arc):
- 0 degrees is at 3 o'clock (to the right)
- Angles increase clockwise: 90=down, 180=left, 270=up, 360=right

Speed gauge (270° donut):
- Progress arc starts at 90° (bottom) and fills clockwise up to 360°/0° (right).
- The bottom-right quadrant is left empty for the numeric readout and units.
- Ticks are placed every 45°, with longer ticks on each 90°.
- The center shows a heading arrow derived from GPS (0°=North/up).

Power gauge (full donut):
- Full 360° ring split into 5 colored segments for zones.
- Current zone adds a translucent center fill.
- Large power value is centered; its font size is computed once using the
  widest typical 3-digit sample ("888 W").
"""

from math import cos, sin, radians, atan2
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ...core.datatypes import Metric
from ...core.fit_loader import FitData
from ...core.units import format_power_w, format_speed_kmh
from .base import Widget, draw_rounded_panel


def _resolve_font_path() -> Optional[str]:
    """Pick a readable TrueType font path if possible.

    Prefers PIL's bundled DejaVu, then common macOS fonts. Returning None lets
    PIL fall back to its default bitmap font (not recommended for overlays).
    """
    try:
        import PIL

        p = Path(PIL.__file__).parent / "fonts" / "DejaVuSans.ttf"
        if p.exists():
            return str(p)
    except Exception:
        pass
    # macOS fallbacks
    for p in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]:
        if Path(p).exists():
            return p
    return None


def _resolve_bold_font_path() -> Optional[str]:
    """Locate a bold TTF if available (DejaVu Sans Bold or Arial Bold)."""
    try:
        import PIL

        candidates = [
            Path(PIL.__file__).parent / "fonts" / "DejaVuSans-Bold.ttf",
            Path(PIL.__file__).parent / "DejaVuSans.ttf",
        ]
        for c in candidates:
            if c.exists():
                return str(c)
    except Exception:
        pass
    for p in [
        "/System/Library/Fonts/Supplemental/Arial Black.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Black.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ]:
        if Path(p).exists():
            return p
    return None


def _resolve_block_font_path() -> Optional[str]:
    """Locate a very bold/blocky TTF if available (Arial Black, Impact)."""
    for p in [
        "/System/Library/Fonts/Supplemental/Arial Black.ttf",
        "/Library/Fonts/Arial Black.ttf",
        "/System/Library/Fonts/Supplemental/Impact.ttf",
        "/Library/Fonts/Impact.ttf",
    ]:
        if Path(p).exists():
            return p
    return _resolve_bold_font_path()


class _FontCache:
    """Simple cache of truetype fonts keyed by (point_size, bold)."""

    def __init__(self, font_path: Optional[Path]):
        self.font_path = str(font_path) if font_path else _resolve_font_path()
        self._cache: Dict[Tuple[int, bool], ImageFont.FreeTypeFont] = {}

    def get(self, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        """Return a FreeType font at the requested point size (cached)."""
        key = (size, bold)
        if key in self._cache:
            return self._cache[key]
        try:
            if self.font_path:
                font = ImageFont.truetype(self.font_path, size=size)
            else:
                font = ImageFont.truetype("DejaVuSans.ttf", size=size)
        except Exception:
            font = ImageFont.load_default()
        self._cache[key] = font
        return font


def _fit_font(
    draw: ImageDraw.ImageDraw,
    fm: _FontCache,
    text: str,
    max_w: int,
    max_h: int,
    max_pt: int,
) -> ImageFont.FreeTypeFont:
    """Find the largest font that fits within (max_w x max_h).

    Uses a binary search over point sizes. Returns a truetype font from the
    cache; if truetype is unavailable, falls back to PIL's default font.
    """
    lo, hi = 8, max_pt
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        font = fm.get(mid)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        if tw <= max_w and th <= max_h:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return fm.get(best)


class SpeedGaugeWidget(Widget):
    """270-degree speed gauge with center compass and bottom-right readout.

    - max_kmh defines the right-end of the scale; renderer typically provides
      ~1.1× the clip's observed max speed (rounded).
    - Progress arc maps v∈[0,max] to angles 90°→360° (bottom→right).
    - The bottom-right quadrant is reserved for the numeric value + units.
    """

    def __init__(
        self,
        fit_data: FitData,
        box: Tuple[int, int, int, int],
        theme: Dict,
        font_path: Optional[Path],
        max_kmh: float,
    ):
        super().__init__(box, theme)
        self.fit = fit_data
        self.fm = _FontCache(font_path)
        self._prepared = False
        self._static: Optional[Image.Image] = None
        self._geom: Optional[dict] = None
        # Additional caches allow different typefaces for value vs units
        from pathlib import Path as _P

        block_path = _resolve_block_font_path()
        bold_path = _resolve_bold_font_path()
        self.fm_value = _FontCache(_P(block_path) if block_path else font_path)
        self.fm_units = _FontCache(_P(bold_path) if bold_path else font_path)
        self.max_kmh = max(5.0, float(max_kmh))
        # Cache fonts for speed readout to avoid jitter
        self._spd_value_font_cached: Optional[ImageFont.FreeTypeFont] = None
        self._spd_units_font_cached: Optional[ImageFont.FreeTypeFont] = None

    def _prepare(self, size: Tuple[int, int]):
        if self._prepared and self._static is not None and self._geom is not None:
            return
        w, h = size
        pad = int(min(w, h) * 0.08)
        cx = w // 2
        cy = h // 2
        r = int(min(w, h) * 0.40)
        r = min(r, cx - pad, cy - pad)
        thickness = max(8, int(r * 0.16))
        bbox = (cx - r, cy - r, cx + r, cy + r)

        static = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(static)
        # Ticks (45°), longer at 90°
        for k in range(0, 7):
            a = (90 + 45 * k) % 360
            arad = radians(a)
            outer = r + thickness * 0.55
            long = k % 2 == 0
            tick_len = thickness * (0.35 if not long else 0.55)
            x0 = cx + int(outer * cos(arad))
            y0 = cy + int(outer * sin(arad))
            x1 = cx + int((outer + tick_len) * cos(arad))
            y1 = cy + int((outer + tick_len) * sin(arad))
            sdraw.line(
                (x0, y0, x1, y1),
                fill=self.theme["muted"],
                width=max(2, int(thickness * 0.12)),
            )

        # Scale labels: 0 bottom, max right
        # baseline_bottom = cy + r + int(thickness * 0.35)
        # zero_txt = "0"
        # tf0 = _fit_font(
        #     sdraw, self.fm, zero_txt, int(w * 0.18), int(h * 0.14), int(h * 0.12)
        # )
        # tb0 = sdraw.textbbox((0, 0), zero_txt, font=tf0)
        # sdraw.text(
        #     (cx - (tb0[2] - tb0[0]) // 2, baseline_bottom),
        #     zero_txt,
        #     fill=self.theme["muted"],
        #     font=tf0,
        # )

        # max_txt = f"{int(self.max_kmh):d}"
        # tfm = _fit_font(
        #     sdraw, self.fm, max_txt, int(w * 0.18), int(h * 0.14), int(h * 0.12)
        # )
        # tbm = sdraw.textbbox((0, 0), max_txt, font=tfm)
        # sdraw.text(
        #     (cx + r + int(thickness * 0.25), cy - (tbm[3] - tbm[1]) // 2),
        #     max_txt,
        #     fill=self.theme["muted"],
        #     font=tfm,
        # )

        # Units text in bottom-right quadrant
        qx1, qy1 = cx, cy
        qx2, qy2 = cx + r, cy + r
        qw, qh = qx2 - qx1, qy2 - qy1
        units_txt = "KPH"
        if self._spd_units_font_cached is None:
            self._spd_units_font_cached = _fit_font(
                sdraw,
                self.fm_units,
                units_txt,
                int(qw * 0.9),
                int(qh * 0.2),
                int(h * 0.18),
            )
        # sdraw.text(
        #     (qx1 + int(0.1 * qw), qy2),
        #     units_txt,
        #     fill=self.theme["muted"],
        #     font=self._spd_units_font_cached,
        #     anchor="ls",
        # )

        self._geom = {
            "pad": pad,
            "cx": cx,
            "cy": cy,
            "r": r,
            "thickness": thickness,
            "bbox": bbox,
            "qx1": qx1,
            "qy1": qy1,
            "qx2": qx2,
            "qy2": qy2,
            "qw": qw,
            "qh": qh,
        }
        self._static = static
        self._prepared = True

    def _angle_for_value(self, v: float) -> float:
        """Map a speed value to a Pillow arc angle.

        0 km/h -> 90° (bottom). max_kmh -> 360°/0° (right). Linear mapping
        across the 270° sweep.
        """
        x = max(0.0, min(1.0, v / self.max_kmh))
        gauge_deg = x * 270.0
        pillow_deg = 90 + gauge_deg
        return pillow_deg

    def _bearing_deg(self, t: float) -> Optional[float]:
        """Approximate heading (deg from North, clockwise) near time t using GPS."""
        gps = self.fit.series.gps
        if not gps or len(gps.t) < 2:
            return None
        from ...core.interpolation import nearest_index

        i = nearest_index(t, gps.t)
        if i is None:
            return None
        j = min(len(gps.t) - 1, i + 2)
        k = max(0, i - 2)
        # choose pair with greater separation
        i1, i2 = (i, j) if abs(j - i) >= abs(i - k) else (k, i)
        lat1, lon1 = gps.lat[i1], gps.lon[i1]
        lat2, lon2 = gps.lat[i2], gps.lon[i2]
        # if identical, bail
        if abs(lat1 - lat2) < 1e-9 and abs(lon1 - lon2) < 1e-9:
            return None
        import math

        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dlon = math.radians(lon2 - lon1)
        y = math.sin(dlon) * math.cos(phi2)
        x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(
            phi2
        ) * math.cos(dlon)
        brng = math.degrees(math.atan2(y, x))
        brng = (brng + 360.0) % 360.0
        return brng

    def render(self, t: float) -> Image.Image:
        """Compose the widget frame for time t as a transparent RGBA image."""
        x1, y1, x2, y2 = self.box
        w, h = x2 - x1, y2 - y1
        self._prepare((w, h))
        assert self._geom is not None
        g = self._geom
        cx, cy, r, thickness, bbox = g["cx"], g["cy"], g["r"], g["thickness"], g["bbox"]
        panel = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)

        pad = int(min(w, h) * 0.08)
        cx = w // 2
        cy = h // 2
        # Radius with margin for ticks and labels
        r = int(min(w, h) * 0.40)
        r = min(r, cx - pad, cy - pad)
        thickness = max(8, int(r * 0.16))

        # Bounding box for the gauge circle
        bbox = (cx - r, cy - r, cx + r, cy + r)

        # Filled progress arc (no grey background)
        val = self.fit.get_metric_value(Metric.speed, t) or 0.0
        # Draw filled arc only (no grey background), clockwise from 270° down to target angle
        target = self._angle_for_value(val)
        start = 90  # bottom
        draw.arc(
            bbox, start=start, end=target, fill=self.theme["accent"], width=thickness
        )

        # Needle arc (1 deg)
        # ax = cx + int(r * cos(radians(target)))
        # ay = cy + int(r * sin(radians(target)))
        start = target - 0.5
        end = target + 0.5
        rr = max(4, int(thickness * 0.5))
        draw.arc(bbox, start=start, end=end, fill=self.theme["fg"], width=thickness)

        # Speed display in bottom-right quadrant (unused 90°)
        qx1, qy1, qx2, qy2, qw, qh = (
            g["qx1"],
            g["qy1"],
            g["qx2"],
            g["qy2"],
            g["qw"],
            g["qh"],
        )
        spd_txt = f"{(val if val is not None else 0.0):.1f}"
        units_txt = "KPH"
        # Cache with widest likely samples once
        if self._spd_value_font_cached is None:
            self._spd_value_font_cached = _fit_font(
                draw,
                self.fm_value,
                "88.8",
                int(qw * 0.8),
                int(qh * 0.6),
                int(h * 0.36),
            )
        if self._spd_units_font_cached is None:
            self._spd_units_font_cached = _fit_font(
                draw,
                self.fm_units,
                units_txt,
                int(qw * 0.8),
                int(qh * 0.18),
                int(h * 0.18),
            )
        units_font = self._spd_units_font_cached
        ub = draw.textbbox((0, 0), units_txt, font=units_font)
        ux = qx1 + int(0.2 * qw)
        uy = qy2
        draw.text(
            (ux, uy), units_txt, fill=self.theme["muted"], font=units_font, anchor="ls"
        )
        uh = abs(ub[3] - ub[1])

        value_font = self._spd_value_font_cached
        vx = qx1 + int(0.2 * qw)
        vy = qy2 - uh - int(0.1 * qh)
        draw.text(
            (vx, vy), spd_txt, fill=self.theme["fg"], font=value_font, anchor="ls"
        )

        # Composite static elements (ticks, labels, units) on top
        if self._static is not None:
            panel.alpha_composite(self._static)

        # Center compass arrow (heading). 0°=North (up), 90°=East (right), 180°=South (down), 270°=West (left)
        br = self._bearing_deg(t)
        if br is not None:
            theta = radians(br)
            # Basis: forward (u) points toward heading; rightward (w) is perpendicular
            u_x, u_y = sin(theta), -cos(theta)
            w_x, w_y = -u_y, u_x
            inner_r = r - thickness * 0.6
            # Use a normalized local shape with points: (0,3) tip; (-2,-2) left shoulder; (0,-1) indent; (2,-2) right shoulder
            # Scale so the tip fits nicely within inner circle
            s = (inner_r * 0.30) / 3.0
            pts_local = [
                (0.0, 3.0),
                (2.0, -2.0),
                (0.0, -1.0),
                (-2.0, -2.0),
            ]  # clockwise from tip

            def map_pt(xl: float, yl: float):
                # screen = center + w*xl*s + u*yl*s
                return (
                    cx + int(round(w_x * xl * s + u_x * yl * s)),
                    cy + int(round(w_y * xl * s + u_y * yl * s)),
                )

            poly = [map_pt(xl, yl) for (xl, yl) in pts_local]
            draw.polygon(poly, fill=(220, 60, 60, 230))

        return panel


class PowerGaugeWidget(Widget):
    """Full-donut power gauge segmented into five zones with center fill.

    - zone_bounds: 5 ascending upper bounds in watts (defines colored segments).
    - The center circle is filled with the current zone color (translucent).
    - The large numeric value is centered; font size is cached once using
      the widest sample "888 W" so layout is stable across frames.
    """

    ZONE_COLORS = [
        (80, 160, 255, 255),  # Zone 1 - blue
        (0, 200, 140, 255),  # Zone 2 - green/teal
        (255, 210, 0, 255),  # Zone 3 - yellow
        (255, 130, 0, 255),  # Zone 4 - orange
        (213, 0, 0, 255),  # Zone 5 - red
    ]

    def __init__(
        self,
        fit_data: FitData,
        box: Tuple[int, int, int, int],
        theme: Dict,
        font_path: Optional[Path],
        zone_bounds: List[float],
    ):
        super().__init__(box, theme)
        self.fit = fit_data
        self.fm = _FontCache(font_path)
        self._prepared = False
        self._static: Optional[Image.Image] = None
        self._geom: Optional[dict] = None
        # zone_bounds: list of 5 upper bounds (W)
        if len(zone_bounds) != 5:
            raise ValueError("zone_bounds must be length 5")
        self.zone_bounds = [float(z) for z in zone_bounds]
        self.max_watts = max(
            self.zone_bounds[-1],
            (max(fit_data.series.power.v) if fit_data.series.power else 300),
        )
        self._value_font_cached: Optional[ImageFont.FreeTypeFont] = None

    def _angle_for_value(self, v: float) -> float:
        """Map watts to an angle on the full 360° ring.

        Starts at -90° (12 o'clock) and sweeps clockwise 360°.
        """
        start = -90.0
        span = 360.0
        x = max(0.0, min(1.0, v / self.max_watts))
        return start + span * x

    def _prepare(self, size: Tuple[int, int]):
        if self._prepared and self._static is not None and self._geom is not None:
            return
        w, h = size
        pad = int(min(w, h) * 0.08)
        cx = w // 2
        cy = h // 2
        r = min(w, h) // 2 - pad
        thickness = max(10, int(r * 0.22))
        bbox = (cx - r, cy - r, cx + r, cy + r)

        static = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(static)
        last = 0.0
        total = self.max_watts
        gap_deg = 2.0
        for i, ub in enumerate(self.zone_bounds):
            start_val = last
            end_val = min(ub, total)
            if end_val <= start_val:
                continue
            start_ang = self._angle_for_value(start_val)
            end_ang = self._angle_for_value(end_val)
            end_ang_adj = end_ang - gap_deg if end_ang > start_ang else end_ang
            sdraw.arc(
                bbox,
                start=start_ang,
                end=end_ang_adj,
                fill=self.ZONE_COLORS[i],
                width=thickness,
            )
            last = ub
        if last < total:
            start_ang = self._angle_for_value(last)
            end_ang = self._angle_for_value(total)
            sdraw.arc(
                bbox,
                start=start_ang,
                end=end_ang,
                fill=self.theme["muted"],
                width=thickness,
            )

        self._geom = {
            "pad": pad,
            "cx": cx,
            "cy": cy,
            "r": r,
            "thickness": thickness,
            "bbox": bbox,
        }
        self._static = static
        self._prepared = True

    def render(self, t: float) -> Image.Image:
        x1, y1, x2, y2 = self.box
        w, h = x2 - x1, y2 - y1
        self._prepare((w, h))
        assert self._geom is not None
        cx, cy, r, thickness, bbox = (
            self._geom["cx"],
            self._geom["cy"],
            self._geom["r"],
            self._geom["thickness"],
            self._geom["bbox"],
        )
        panel = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)
        if self._static is not None:
            panel.alpha_composite(self._static)

        # Current value and zone center fill
        val = self.fit.get_metric_value(Metric.power, t) or 0.0
        # Determine current zone index
        zone_idx = 0
        for i, ub in enumerate(self.zone_bounds):
            if val <= ub:
                zone_idx = i
                break
        else:
            zone_idx = len(self.zone_bounds) - 1

        inner_r = r - thickness + max(2, int(thickness * 0.1))
        inner_pad = int(thickness * 0.15)
        fill_r = max(0, inner_r - inner_pad)
        cx1, cy1 = cx, cy
        zone_col = self.ZONE_COLORS[zone_idx]
        zone_fill = (zone_col[0], zone_col[1], zone_col[2], 110)
        draw.ellipse(
            (cx1 - fill_r, cy1 - fill_r, cx1 + fill_r, cy1 + fill_r), fill=zone_fill
        )

        # Value centered (no label)
        value_str = format_power_w(val)
        if self._value_font_cached is None:
            sample = "888 W"  # widest typical 3-digit string with units
            self._value_font_cached = _fit_font(
                draw,
                self.fm,
                sample,
                int(fill_r * 1.8),
                int(fill_r * 0.9),
                int(h * 0.34),
            )
        value_font = self._value_font_cached
        vb = draw.textbbox((0, 0), value_str, font=value_font)
        vx = cx - (vb[2] - vb[0]) // 2
        vy = cy - (vb[3] - vb[1]) // 2
        draw.text((vx, vy), value_str, fill=self.theme["fg"], font=value_font)

        # Lightning bolt icon centered above the value
        # Compute available vertical space above the value within inner circle
        top_inner = cy - fill_r
        space_above = max(0, abs(vy - top_inner))
        if space_above > 4:
            margin = max(4, int(h * 0.01))
            # Follow user-specified bolt path: (0,0)→(-2,-4)→(0,-4)→(-5,-9)→(-3,-5)→(-5,-5)→(0,0)
            # Normalize: width=5, height=9
            bolt_units = [
                (0, 0),
                (-2, -4),
                (0, -4),
                (-5, -9),
                (-3, -5),
                (-5, -5),
                (0, 0),
            ]
            bolt_units = [(-x, y) for (x, y) in bolt_units]  # mirror
            unit_w, unit_h = 5.0, 9.0
            avail_h = float(space_above - margin)
            avail_w = float(2 * fill_r * 0.9)
            s = max(1.0, min(avail_h / unit_h, avail_w / unit_w))
            # Center horizontally at cx (shape center is at x = 2.5 in unit coords)
            x_off = cx + s * (-2.5)
            # Place bottom (y=0) just above the value with margin
            y_off = float(vy - (margin // 2))
            poly_int = [
                (int(round(x_off + s * x)), int(round(y_off + s * y)))
                for (x, y) in bolt_units
            ]
            draw.polygon(poly_int, fill=self.theme.get("accent", (255, 255, 0, 230)))

        return panel
