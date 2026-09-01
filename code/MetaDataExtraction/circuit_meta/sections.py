from __future__ import annotations

from typing import List

from . import config
from .models import Section
from .pdf_doc import PdfDocument, Line
from .text_utils import flatten


import re

# A line that is *only* a section number, e.g. "I." or "III." — some venues
# (notably conference templates) render the numeral as its own span/line next to
# the title text rather than inline.
_ROMAN_ONLY_RE = re.compile(r'^\s*([IVX]{1,5})\s*[.)]?\s*$')


def _classify(title: str) -> str:
    low = title.lower()
    for kind, keys in config.SECTION_KIND_KEYWORDS.items():
        if any(k in low for k in keys):
            return kind
    return "other"


def _is_header(line: Line) -> tuple[bool, str, str]:
    m = config.SECTION_RE.match(line.text)
    if not m:
        return False, "", ""
    title = m.group(2).strip()
    letters = [c for c in title if c.isalpha()]
    if not letters:
        return False, "", ""
    upper_frac = sum(c.isupper() for c in letters) / len(letters)
    if upper_frac < 0.7 or len(title) > 60:
        return False, "", ""
    return True, m.group(1), title


def _looks_like_title(text: str) -> bool:
    t = text.strip()
    if not (3 <= len(t) <= 60):
        return False
    if config.CAPTION_RE.match(t) or config.TABLE_RE.match(t):
        return False
    if not (t[0].isalpha() and t[0].isupper()):
        return False
    return len(t.split()) <= 8 and not t.endswith((".", ",", ";", ":"))


def extract_sections(doc: PdfDocument) -> List[Section]:
    lines = doc.all_lines()
    sections: List[Section] = []
    current: Section | None = None
    buf: List[str] = []

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        is_hdr, number, title = _is_header(line)
        consumed_title = False
        if not is_hdr:
            # Split numeral+title header: a "I."/"III." line immediately followed
            # (same baseline) by a short capitalised title line.
            m = _ROMAN_ONLY_RE.match(line.text)
            if m and idx + 1 < len(lines):
                nxt = lines[idx + 1]
                if abs(nxt.y0 - line.y0) <= 6 and _looks_like_title(nxt.text):
                    is_hdr, number, title = True, m.group(1), nxt.text.strip()
                    consumed_title = True
        if is_hdr:
            if current is not None:
                current.text = flatten(" ".join(buf))
                sections.append(current)
            current = Section(number=number, title=title,
                              kind=_classify(title), page=line.page,
                              header_y=line.y0, order=idx)
            buf = []
            if consumed_title:
                idx += 1  # skip the title line we folded into the header
        else:
            if current is not None:
                buf.append(line.text)
        idx += 1
    if current is not None:
        current.text = flatten(" ".join(buf))
        sections.append(current)

    return sections


def section_for_position(sections: List[Section], page: int,
                         y: float) -> Section | None:
    chosen = None
    for sec in sorted(sections, key=lambda s: (s.page, s.header_y)):
        if sec.page < page:
            chosen = sec
        elif sec.page == page and sec.header_y <= y + 60:
            chosen = sec
        elif sec.page > page:
            break
    return chosen
