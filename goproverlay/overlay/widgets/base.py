from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple

from PIL import Image, ImageDraw


def draw_rounded_panel(size: Tuple[int, int], radius: int, color: Tuple[int, int, int, int]) -> Image.Image:
    w, h = size
    panel = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(panel)
    rect = (0, 0, w, h)
    draw.rounded_rectangle(rect, radius=radius, fill=color)
    return panel


class Widget(ABC):
    def __init__(self, box: Tuple[int, int, int, int], theme: Dict[str, Tuple[int, int, int, int]]):
        self.box = box  # (x1, y1, x2, y2)
        self.theme = theme

    @abstractmethod
    def render(self, t: float) -> Image.Image:
        """Return an RGBA image representing the widget at time t."""

    def draw_onto(self, base: Image.Image, t: float) -> None:
        x1, y1, x2, y2 = self.box
        w, h = x2 - x1, y2 - y1
        img = self.render(t)
        if img.size != (w, h):
            img = img.resize((w, h), Image.LANCZOS)
        base.paste(img, (x1, y1), img)

