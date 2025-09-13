from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, List

from PIL import Image, ImageDraw, ImageFont

from ...core.datatypes import Metric
from ...core.fit_loader import FitData
from ...core.interpolation import hold_last_interpolate, nearest_index
from .base import Widget


def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c


class ElevationPlotWidget(Widget):
    def __init__(self, fit_data: FitData, box: Tuple[int, int, int, int], theme: Dict, font_path=None, duration: Optional[float] = None):
        super().__init__(box, theme)
        self.fit = fit_data
        self._prepared = False
        self._duration = duration

    def _prepare(self, size):
        if self._prepared:
            return
        self._prepared = True
        gps = self.fit.series.gps
        elev = self.fit.series.elevation
        if not gps or not elev or gps.is_empty() or elev.is_empty():
            self._dist = []
            self._elev = []
            return
        # Build distance profile and align elevation at gps times, restricted to clip duration if provided
        if self._duration is not None:
            mask_idx = [i for i, ti in enumerate(gps.t) if 0.0 <= ti <= self._duration]
            if len(mask_idx) < 2:
                # Fallback to whole series if too few points in-range
                t = gps.t
                lat = gps.lat
                lon = gps.lon
            else:
                t = [gps.t[i] for i in mask_idx]
                lat = [gps.lat[i] for i in mask_idx]
                lon = [gps.lon[i] for i in mask_idx]
        else:
            t = gps.t
            lat = gps.lat
            lon = gps.lon
        dist = [0.0]
        for i in range(1, len(t)):
            d = _haversine_m(lat[i-1], lon[i-1], lat[i], lon[i])
            dist.append(dist[-1] + d)
        # Forward-fill any initial missing elevation with the first known value
        first_ev = elev.v[0] if elev and elev.v else None
        e = []
        for ti in t:
            v = hold_last_interpolate(ti, elev.t, elev.v)
            if v is None:
                v = first_ev
            e.append(float(v))
        self._dist = dist
        self._elev = e
        self._t = t
        self._total_dist = dist[-1] if dist else 0.0
        if self._total_dist <= 0:
            self._total_dist = 1.0

        # Compute gradient per segment (%), clamp extremes
        grads = [0.0]
        for i in range(1, len(dist)):
            dd = max(1e-3, dist[i] - dist[i-1])
            de = (e[i] - e[i-1])
            g = 100.0 * de / dd
            grads.append(max(-20.0, min(20.0, g)))
        self._grads = grads

    def _grad_color(self, g: float):
        # Simple gradient zones coloring
        if g <= -6:   # steep down
            return (120, 170, 255, 60)
        if g <= -3:
            return (120, 220, 255, 60)
        if g < 0:
            return (120, 255, 200, 60)
        if g < 3:
            return (110, 220, 110, 60)
        if g < 6:
            return (255, 210, 0, 60)
        if g < 9:
            return (255, 150, 0, 60)
        return (213, 0, 0, 60)

    def render(self, t: float) -> Image.Image:
        x1, y1, x2, y2 = self.box
        w, h = x2 - x1, y2 - y1
        self._prepare((w, h))
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        if not self._dist or not self._elev:
            return img

        pad = int(min(w, h) * 0.08)
        inner_w = max(1, w - 2 * pad)
        inner_h = max(1, h - 2 * pad)
        gx0, gy0 = pad, pad

        d = self._dist
        e = self._elev
        total_d = self._total_dist
        emin = min(e)
        emax = max(e)
        rng = max(1e-6, emax - emin)
        # Add 10% margin below and above the observed range in the (clip) data
        margin = 0.10 * rng
        emin_disp = emin - margin
        emax_disp = emax + margin
        if emax_disp - emin_disp < 1e-6:
            emax_disp = emin_disp + 1.0
        self._emin_disp = emin_disp
        self._emax_disp = emax_disp

        # Elevation polyline and filled area under the curve
        pts = []
        for di, ei in zip(d, e):
            x = gx0 + int((di / total_d) * inner_w)
            y = gy0 + int((1 - (ei - self._emin_disp) / (self._emax_disp - self._emin_disp)) * inner_h)
            pts.append((x, y))
        # Fill area under the curve to the bottom of plot area
        if pts:
            base_y = gy0 + inner_h
            poly = [(gx0, base_y)] + pts + [(gx0 + inner_w, base_y)]
            track_col = self.theme.get("track", (120, 200, 255, 255))
            fill_col = (track_col[0], track_col[1], track_col[2], 70)
            draw.polygon(poly, fill=fill_col)
        draw.line(pts, fill=(255, 255, 255, 220), width=max(2, int(h * 0.02)))

        # Current position marker
        if self._t:
            from ...core.interpolation import nearest_index
            idx = nearest_index(t, self._t)
            if idx is not None:
                dcur = d[idx]
                xc = gx0 + int((dcur / total_d) * inner_w)
                draw.line((xc, gy0, xc, gy0 + inner_h), fill=(255, 255, 255, 180), width=2)

        # Elevation value text
        val = self.fit.get_metric_value(Metric.elevation, t)
        if val is not None:
            txt = f"{int(round(val))} m"
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", size=max(12, int(h * 0.16)))
            except Exception:
                font = ImageFont.load_default()
            tb = draw.textbbox((0, 0), txt, font=font)
            draw.text((pad, pad), txt, fill=self.theme["fg"], font=font)

        return img
