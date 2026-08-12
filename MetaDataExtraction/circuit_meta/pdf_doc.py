"""Layout-aware access layer over a PDF (built on PyMuPDF).

Academic circuit papers are almost always two-column. PyMuPDF returns text
blocks in storage order, which interleaves the two columns and scrambles reading
order. This module exposes lines with geometry, reconstructs column-aware reading
order, and renders/crops page regions to PNG for figure extraction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import fitz

from . import config
from .text_utils import is_noise_line


@dataclass
class Line:
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    size: float                       # max span size on the line
    fonts: List[str] = field(default_factory=list)
    column: int = 0                   # 0 left, 1 right, -1 full-width

    @property
    def x_center(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def width(self) -> float:
        return self.x1 - self.x0


class PdfDocument:
    """Wraps a fitz.Document and offers layout-aware helpers."""

    def __init__(self, path: str):
        self.path = path
        self.doc = fitz.open(path)
        self.page_count = self.doc.page_count
        self._lines_by_page: dict[int, List[Line]] = {}

    # ------------------------------------------------------------------ #
    # Raw line extraction
    # ------------------------------------------------------------------ #
    def _page_width(self, pno: int) -> float:
        return self.doc[pno].rect.width

    def lines(self, pno: int, drop_noise: bool = True) -> List[Line]:
        """All text lines on a page, tagged with column, in reading order."""
        if pno in self._lines_by_page:
            lines = self._lines_by_page[pno]
        else:
            lines = self._extract_lines(pno)
            self._lines_by_page[pno] = lines
        if drop_noise:
            return [l for l in lines if not is_noise_line(l.text)]
        return lines

    def _extract_lines(self, pno: int) -> List[Line]:
        page = self.doc[pno]
        width = page.rect.width
        mid = width / 2.0
        raw: List[Line] = []
        for b in page.get_text("dict").get("blocks", []):
            if b.get("type") != 0:
                continue
            for ln in b.get("lines", []):
                spans = ln.get("spans", [])
                if not spans:
                    continue
                text = "".join(s["text"] for s in spans).strip()
                if not text:
                    continue
                size = max(s["size"] for s in spans)
                fonts = sorted({s["font"] for s in spans})
                x0, y0, x1, y1 = ln["bbox"]
                line = Line(pno, x0, y0, x1, y1, text, size, fonts)
                # Column classification.
                if line.width > 0.55 * width:
                    line.column = -1
                elif line.x_center < mid:
                    line.column = 0
                else:
                    line.column = 1
                raw.append(line)
        return self._reading_order(raw, mid)

    @staticmethod
    def _reading_order(lines: List[Line], mid: float) -> List[Line]:
        """Reconstruct two-column reading order.

        Full-width lines (column == -1) act as horizontal separators: content is
        read top-to-bottom, but within each horizontal band the left column is
        read fully before the right column.
        """
        # Sort full-width markers by y to create bands.
        seps = sorted([l for l in lines if l.column == -1], key=lambda l: l.y0)
        sep_ys = [l.y0 for l in seps] + [float("inf")]

        ordered: List[Line] = []
        prev_y = float("-inf")
        for i, top in enumerate(seps + [None]):
            band_top = prev_y
            band_bot = sep_ys[i]
            band = [l for l in lines
                    if l.column != -1 and band_top <= l.y0 < band_bot]
            left = sorted([l for l in band if l.column == 0], key=lambda l: l.y0)
            right = sorted([l for l in band if l.column == 1], key=lambda l: l.y0)
            ordered.extend(left)
            ordered.extend(right)
            if top is not None:
                ordered.append(top)
                prev_y = top.y0
        return ordered

    # ------------------------------------------------------------------ #
    # Aggregate text
    # ------------------------------------------------------------------ #
    def all_lines(self, drop_noise: bool = True) -> List[Line]:
        out: List[Line] = []
        for pno in range(self.page_count):
            out.extend(self.lines(pno, drop_noise=drop_noise))
        return out

    def full_text(self) -> str:
        """Whole-document text in reading order (newline-joined lines)."""
        return "\n".join(l.text for l in self.all_lines())

    # ------------------------------------------------------------------ #
    # Figure rendering
    # ------------------------------------------------------------------ #
    def column_bounds(self, pno: int) -> Tuple[float, float, float, float, float]:
        """Return (page_x0, left_right_edge, right_left_edge, page_x1, mid)."""
        page = self.doc[pno]
        r = page.rect
        mid = r.width / 2.0
        # Estimate text margins from non-noise lines.
        lines = self.lines(pno)
        if lines:
            left_x0 = min(l.x0 for l in lines)
            right_x1 = max(l.x1 for l in lines)
        else:
            left_x0, right_x1 = r.x0 + 40, r.x1 - 40
        # gutter is a small band around mid
        return left_x0, mid - 8, mid + 8, right_x1, mid

    def render_clip(self, pno: int, bbox: Tuple[float, float, float, float],
                    out_path: str, dpi: int = config.RENDER_DPI) -> str:
        """Render a rectangular region of a page to a PNG file."""
        page = self.doc[pno]
        clip = fitz.Rect(*bbox).intersect(page.rect)
        pix = page.get_pixmap(clip=clip, dpi=dpi, alpha=False)
        pix.save(out_path)
        return out_path

    def close(self):
        self.doc.close()
