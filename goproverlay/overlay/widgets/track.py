from __future__ import annotations

from typing import Dict, Optional, Tuple

from PIL import Image, ImageDraw

from ...core.fit_loader import FitData
from .base import Widget, draw_rounded_panel


class TrackWidget(Widget):
    def __init__(self, fit_data: FitData, box: Tuple[int, int, int, int], theme: Dict, duration: Optional[float] = None):
        super().__init__(box, theme)
        self.fit = fit_data
        self.duration = duration
        self._prepared = False
        self._base = None  # Pre-rendered track image RGBA
        self._bounds = None  # lat_min, lat_max, lon_min, lon_max
        self._scale = None  # sx, sy, ox, oy for mapping lon/lat -> x,y

    def _prepare(self, size: Tuple[int, int]):
        if self._prepared:
            return
        w, h = size
        pad = int(min(w, h) * 0.15)
        self._pad = pad
        gps = self.fit.series.gps
        panel = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)

        if not gps or gps.is_empty():
            # draw placeholder text
            draw.text((pad, pad), "No GPS", fill=self.theme["muted"])
            self._base = panel
            self._prepared = True
            return

        # Filter GPS to the clip duration if provided
        if self.duration is not None:
            idxs = [i for i, ti in enumerate(gps.t) if 0.0 <= ti <= self.duration]
            if len(idxs) >= 2:
                lats = [gps.lat[i] for i in idxs]
                lons = [gps.lon[i] for i in idxs]
            else:
                lats = list(gps.lat)
                lons = list(gps.lon)
        else:
            lats = list(gps.lat)
            lons = list(gps.lon)

        if len(lats) < 2:
            draw.text((pad, pad), "No GPS in clip", fill=self.theme["muted"])
            self._base = panel
            self._prepared = True
            return

        lat_min = min(lats)
        lat_max = max(lats)
        lon_min = min(lons)
        lon_max = max(lons)
        # Avoid zero-size
        if lat_max - lat_min < 1e-9:
            lat_max += 1e-9
        if lon_max - lon_min < 1e-9:
            lon_max += 1e-9

        # Compute aspect-fit scaling into inner rect
        inner_w = w - 2 * pad
        inner_h = h - 2 * pad
        dlon = lon_max - lon_min
        dlat = lat_max - lat_min
        # Correct for lat/lon aspect: approximate meter distance scaling by cos(lat)
        import math

        mid_lat = 0.5 * (lat_min + lat_max)
        aspect = math.cos(math.radians(mid_lat))
        map_w = dlon * aspect
        map_h = dlat
        if map_w <= 0:
            map_w = 1e-9
        if map_h <= 0:
            map_h = 1e-9

        sx = inner_w / map_w
        sy = inner_h / map_h
        s = min(sx, sy)

        # Compute offsets to center
        plot_w = map_w * s
        plot_h = map_h * s
        ox = pad + (inner_w - plot_w) * 0.5
        oy = pad + (inner_h - plot_h) * 0.5

        # Helper to map lon/lat to xy
        def map_point(la, lo):
            x = (lo - lon_min) * aspect * s + ox
            y = (lat_max - la) * s + oy  # y inverted
            return (x, y)

        # Draw path using filtered coordinates
        pts = [map_point(la, lo) for la, lo in zip(lats, lons)]
        draw.line(pts, fill=self.theme["track"], width=max(2, int(h * 0.02)))

        self._base = panel
        self._map_point = map_point
        self._prepared = True

    def render(self, t: float) -> Image.Image:
        x1, y1, x2, y2 = self.box
        w, h = x2 - x1, y2 - y1
        self._prepare((w, h))
        img = self._base.copy()
        draw = ImageDraw.Draw(img)

        # North-up indicator (slightly larger to encompass arrow and N)
        comp_r = max(8, int(h * 0.09))
        pad = int(min(w, h) * 0.06)
        cx, cy = (w - pad - comp_r, pad + comp_r)
        draw.ellipse((cx - comp_r, cy - comp_r, cx + comp_r, cy + comp_r), outline=self.theme["muted"], width=2)
        # Arrow pointing up (inside circle)
        arrow_h = int(comp_r * 0.75)
        arrow_w = int(arrow_h * 0.45)
        draw.polygon([(cx, cy - arrow_h), (cx - arrow_w, cy), (cx + arrow_w, cy)], fill=self.theme["accent"]) 
        # 'N' label inside circle near bottom
        try:
            from PIL import ImageFont
            font = ImageFont.load_default()
            tb = draw.textbbox((0, 0), "N", font=font)
            tx = cx - (tb[2]-tb[0])//2
            ty = cy + int(comp_r*0.35) - (tb[3]-tb[1])//2
            draw.text((tx, ty), "N", fill=self.theme["fg"], font=font)
        except Exception:
            pass

        pos = self.fit.get_position(t)
        if pos is not None and hasattr(self, "_map_point"):
            la, lo = pos
            x, y = self._map_point(la, lo)
            r = max(3, int(h * 0.04))
            draw.ellipse((x - r, y - r, x + r, y + r), fill=self.theme["accent"], outline=(255, 255, 255, 200), width=2)

        return img
