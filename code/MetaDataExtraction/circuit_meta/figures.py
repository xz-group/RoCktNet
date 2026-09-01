from __future__ import annotations

import os
import re
from typing import Dict, List

from . import config
from .models import FigureCandidate, FigureReference, Section
from .pdf_doc import PdfDocument, Line
from .sections import section_for_position
from .text_utils import flatten, sentence_containing

# Caption font is smaller than body text in these papers.
_CAPTION_MAX_SIZE = 9.3


def _is_caption_start(line: Line) -> int | None:
    if line.size > _CAPTION_MAX_SIZE:
        return None  # body-text reference like "Fig. 1(a) shows", not a caption
    m = config.CAPTION_RE.match(line.text)
    if not m:
        return None
    return int(m.group(1))


def _collect_caption(lines: List[Line], start_idx: int) -> tuple[str, List[float]]:
    start = lines[start_idx]
    parts = [start.text]
    x0, y0, x1, y1 = start.x0, start.y0, start.x1, start.y1
    j = start_idx + 1
    while j < len(lines):
        nxt = lines[j]
        # Stop on a new caption, a font-size jump to body text, a column change,
        # or a large vertical gap.
        if _is_caption_start(nxt) is not None:
            break
        if nxt.size > _CAPTION_MAX_SIZE:
            break
        if nxt.column != start.column:
            break
        if nxt.y0 - y1 > 2.2 * max(start.size, 1.0):
            break
        if config.TABLE_RE.match(nxt.text) or config.SECTION_RE.match(nxt.text):
            break
        parts.append(nxt.text)
        x0, y0 = min(x0, nxt.x0), min(y0, nxt.y0)
        x1, y1 = max(x1, nxt.x1), max(y1, nxt.y1)
        j += 1
    return flatten(" ".join(parts)), [x0, y0, x1, y1]


def detect_captions(doc: PdfDocument) -> List[FigureCandidate]:
    found: Dict[int, FigureCandidate] = {}
    for pno in range(doc.page_count):
        lines = doc.lines(pno)
        i = 0
        while i < len(lines):
            fid = _is_caption_start(lines[i])
            if fid is None or fid in found:
                i += 1
                continue
            caption, bbox = _collect_caption(lines, i)
            found[fid] = FigureCandidate(
                id=fid,
                label=f"Fig. {fid}",
                page=pno,
                caption=caption,
                caption_bbox=[round(v, 1) for v in bbox],
            )
            i += 1
    return [found[k] for k in sorted(found)]


def _crop_bbox(doc: PdfDocument, fig: FigureCandidate) -> List[float]:
    pno = fig.page
    cap_x0, cap_y0, cap_x1, cap_y1 = fig.caption_bbox
    left_x0, left_edge, right_edge, right_x1, mid = doc.column_bounds(pno)
    cap_w = cap_x1 - cap_x0

    # Horizontal span: follow the caption's column unless it is wide (full-width fig).
    if cap_w > 0.55 * doc._page_width(pno):
        x0, x1 = left_x0, right_x1
        column = -1
    elif (cap_x0 + cap_x1) / 2 < mid:
        x0, x1 = left_x0, mid - 4
        column = 0
    else:
        x0, x1 = mid + 4, right_x1
        column = 1

    # Vertical span: from the bottom of the nearest content above (same column)
    # down through the caption.
    same_col = []
    for l in doc.lines(pno):
        lc = -1 if l.width > 0.55 * doc._page_width(pno) else (0 if l.x_center < mid else 1)
        if column == -1 or lc == column or lc == -1:
            same_col.append(l)
    above = [l for l in same_col if l.y1 <= cap_y0 - 2]
    fig_top = max((l.y1 for l in above), default=40.0) + config.CROP_PAD_TOP
    if fig_top >= cap_y0 - 4:  # nothing but the caption — fall back to column top
        fig_top = 40.0
    y1 = cap_y1 + config.CROP_PAD_BELOW_CAPTION
    return [round(x0, 1), round(fig_top, 1), round(x1, 1), round(y1, 1)]


def crop_figure(doc: PdfDocument, fig: FigureCandidate, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fig.crop_bbox = _crop_bbox(doc, fig)
    out_path = os.path.join(out_dir, f"fig_{fig.id:02d}.png")
    try:
        doc.render_clip(fig.page, tuple(fig.crop_bbox), out_path)
        return out_path
    except Exception as exc:  # pragma: no cover - defensive
        fig.warnings.append(f"figure crop failed: {exc}")
        return ""


def attach_references(fig: FigureCandidate, sections: List[Section],
                      full_text_fallback: str) -> None:
    refs: List[FigureReference] = []
    seen = set()
    for sec in sections:
        for sent in sentence_containing(sec.text, config.FIG_REF_RE, fig.id):
            key = sent[:80]
            if key in seen:
                continue
            seen.add(key)
            refs.append(FigureReference(section_title=sec.title,
                                        section_kind=sec.kind,
                                        page=sec.page, sentence=sent))
    if not refs:
        for sent in sentence_containing(full_text_fallback, config.FIG_REF_RE, fig.id):
            refs.append(FigureReference(sentence=sent))
    fig.references = refs


def assign_section(fig: FigureCandidate, sections: List[Section]) -> None:
    sec = section_for_position(sections, fig.page, fig.caption_bbox[1])
    if sec:
        fig.section_title = f"{sec.number}. {sec.title}"
        fig.section_kind = sec.kind
