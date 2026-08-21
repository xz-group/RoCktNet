"""Extract paper-level bibliographic metadata from the first page."""
from __future__ import annotations

import re
from typing import List, Optional

from . import config
from .models import PaperMetadata
from .pdf_doc import PdfDocument, Line
from .text_utils import clean_text, flatten


_ABSTRACT_RE = re.compile(r'^\s*Abstract\s*[—–\-:]+\s*', re.IGNORECASE)
# Index-terms / keywords marker. Different venues use different words:
# IEEE Transactions use "Index Terms—", many conference papers use "Keywords—".
_INDEX_RE = re.compile(
    r'^\s*(?:Index\s*Terms|Key\s*[Ww]ords?)\s*[—–\-:]+\s*', re.IGNORECASE)
_YEAR_RE = re.compile(r'\b(19|20)\d{2}\b')
# Copyright / ISBN-ISSN line carries the publication year, e.g.
# "978-1-5090-5025-3/17/$31.00 ©2017 IEEE" or "0018-9480/$25.00 © 2007 IEEE".
_COPYRIGHT_YEAR_RE = re.compile(r'(?:©|\(c\)|/(\d{2})/\$).{0,12}?((?:19|20)\d{2})',
                                re.IGNORECASE)
# Lines that sit where a title would but are page furniture. In ISSCC digests the
# session header is printed *larger* than the real title, so it wins the
# largest-font test unless it is filtered out; the bare paper number ("12.2") that
# precedes the title shares the title's font and would be glued onto it.
_BANNER_RE = re.compile(
    r'^\s*(?:'
    r'(?:Brief\s+Papers|Papers|Letters|Correspondence)[\s_]*'
    r'|ISSCC\s+\d{4}\s*/.*'          # "ISSCC 2006 / SESSION 12 / NYQUIST ADCs / 12.2"
    r'|\d{1,2}\.\d{1,2}'             # bare ISSCC paper number
    r')\s*$',
    re.IGNORECASE)
_IEEE_MEMBER_RE = re.compile(
    r',?\s*(?:Student\s+|Senior\s+|Life\s+)?(?:Member|Fellow|Graduate Student Member),?\s*IEEE',
    re.IGNORECASE)
# Page furniture that can share the author band's font size (conference footers).
_FURNITURE_RE = re.compile(
    r'(?:IEEE\s+International\s+.*Conference|Solid-State\s+Circuits\s+Conference'
    r'|Symposium\s+on|©|\(c\)\s*\d{4}|\b\d{3}-\d-\d{4}-\d{4}-\d\b|ISBN)',
    re.IGNORECASE)
# Lines that mark the start of affiliation/address text (not author names).
_AFFIL_RE = re.compile(
    r'\b(?:University|Universit[àé]|Institute|Department|Laborator|Labs?|Inc\.?|'
    r'Corp|Corporation|GmbH|Co\.,|Ltd|National|Academy|School\s+of|Division|'
    r'Netherlands|Germany|France|Japan|China|Korea|Taiwan|Italy|Spain|Sweden|'
    r'Switzerland|Belgium|Canada|U\.?S\.?A|United\s+States|Email|e-mail)\b',
    re.IGNORECASE)


def _page0_lines_by_y(doc: PdfDocument) -> List[Line]:
    lines = doc.lines(0, drop_noise=False)
    return sorted(lines, key=lambda l: (round(l.y0, 1), l.x0))


def _extract_title(lines: List[Line]) -> tuple[str, List[Line]]:
    """Title = the largest-font multi-char text in the top region of page 1.

    Returns the title text and the lines it was built from, so the author band
    can be anchored to the lines actually used as the title.
    """
    top = [l for l in lines
           if l.y0 < 160 and len(l.text.strip()) > 3
           and not _BANNER_RE.match(l.text)]
    if not top:
        return "", []
    max_size = max(l.size for l in top)
    # Drop-caps and headers can be large but are single chars / tiny banners;
    # require lines close to the dominant title size.
    title_lines = [l for l in top if l.size >= max_size - 1.0]
    title_lines.sort(key=lambda l: l.y0)
    title = flatten(" ".join(l.text for l in title_lines))
    return title, title_lines


def _extract_venue_year(lines: List[Line]) -> tuple[str, Optional[int]]:
    venue, year = "", None
    for l in lines[:4]:
        t = l.text.strip()
        if "TRANSACTIONS" in t.upper() or "JOURNAL" in t.upper() or "PROCEEDINGS" in t.upper() \
                or re.search(r'\bVOL\.', t):
            # venue name = part before ", VOL"
            venue = re.split(r',?\s*VOL\.', t)[0].strip()
            m = _YEAR_RE.search(t)
            if m:
                year = int(m.group(0))
            break
    return clean_text(venue), year


def _extract_authors(lines: List[Line], title_lines: List[Line],
                     abstract_y: float) -> List[str]:
    """Author line sits between the title and the abstract.

    The band is anchored to the *top* of the lowest title line, not its bottom: a
    subscript or descender ("LiNbO3") inflates the bottom past the author line's
    ``y0`` and would drop the authors entirely. The title lines themselves are
    excluded by identity so anchoring that high cannot pull them in.

    Deriving the anchor by substring-matching the title text (the previous
    approach) also matched stray single glyphs elsewhere on the page — drop-caps
    and dropped math symbols — pushing the band below the authors.
    """
    if not title_lines:
        return []
    title_y = max(l.y0 for l in title_lines)
    skip = {id(l) for l in title_lines}
    cand = [l for l in lines
            if title_y < l.y0 < abstract_y and 9.5 <= l.size <= 14.0
            and id(l) not in skip
            and not _FURNITURE_RE.search(l.text)]
    if not cand:
        return []
    cand.sort(key=lambda l: l.y0)
    # Author names sit at the largest font in this band; affiliations/emails are
    # printed smaller just below. Keep only the dominant (name) font size.
    name_size = max(l.size for l in cand)
    cand = [l for l in cand if l.size >= name_size - 0.5]
    # Stop at the first line that reads like an affiliation/address.
    kept: List[Line] = []
    for l in cand:
        if _AFFIL_RE.search(l.text):
            break
        kept.append(l)
    cand = kept or cand
    raw = flatten(" ".join(l.text for l in cand))
    # Strip IEEE membership annotations.
    raw = _IEEE_MEMBER_RE.sub("", raw)
    # Drop any trailing affiliation clause that shares the author line.
    raw = _AFFIL_RE.split(raw)[0]
    # Split on commas and "and".
    parts = re.split(r',|\band\b', raw)
    authors = []
    for p in parts:
        name = p.strip(" .,")
        # A name must contain letters: bare page numbers from a footer line that
        # shares the author band ("470 • 2011 IEEE ...") are not authors.
        if (name and len(name) > 1 and not name.isupper()
                and re.search(r'[A-Za-z]', name)):
            authors.append(name)
    return authors


def _extract_block(lines: List[Line], start_re: re.Pattern,
                   stop_res: List[re.Pattern]) -> str:
    """Collect text from the line matching ``start_re`` until a stop pattern."""
    collecting = False
    buf: List[str] = []
    for l in lines:
        t = l.text
        if not collecting:
            if start_re.match(t):
                collecting = True
                buf.append(start_re.sub("", t))
            continue
        if any(r.match(t) for r in stop_res):
            break
        # Stop at a section header (e.g. "I. INTRODUCTION").
        if config.SECTION_RE.match(t):
            break
        buf.append(t)
    return flatten(" ".join(buf))


def extract_doi(doc: PdfDocument) -> str:
    # The DOI line is filtered as noise; search raw lines on the first pages.
    for pno in range(min(2, doc.page_count)):
        for l in doc.lines(pno, drop_noise=False):
            m = config.DOI_RE.search(l.text)
            if m:
                return m.group(0).rstrip(".")
    return ""


def _copyright_year(doc: PdfDocument) -> Optional[int]:
    """Year from the ISBN/copyright furniture line (conference papers)."""
    for pno in range(min(2, doc.page_count)):
        for l in doc.lines(pno, drop_noise=False):
            m = _COPYRIGHT_YEAR_RE.search(l.text)
            if m:
                return int(m.group(2))
    return None


def extract_keywords(lines: List[Line]) -> List[str]:
    """Keywords/Index-Terms list. The list is delimited by font size: it shares
    the marker line's size and ends where body/header text (a different size)
    begins, so we don't depend on a section header being recognised."""
    start = next((i for i, l in enumerate(lines) if _INDEX_RE.match(l.text)), None)
    if start is None:
        return []
    marker_size = lines[start].size
    buf = [_INDEX_RE.sub("", lines[start].text)]
    for l in lines[start + 1:]:
        if abs(l.size - marker_size) > 0.6 or config.SECTION_RE.match(l.text):
            break
        buf.append(l.text)
    block = flatten(" ".join(buf))
    # Keywords end at the first sentence period (a section sometimes follows).
    block = block.split(".")[0]
    kws = [clean_text(k) for k in re.split(r'[;,]', block)]
    return [k for k in kws if k and len(k) > 1]


def extract_paper_metadata(doc: PdfDocument) -> PaperMetadata:
    lines = _page0_lines_by_y(doc)

    title, title_lines = _extract_title(lines)
    venue, year = _extract_venue_year(lines)

    abstract_y = next((l.y0 for l in lines if _ABSTRACT_RE.match(l.text)), 1e9)
    authors = _extract_authors(lines, title_lines, abstract_y)

    abstract = clean_text(_extract_block(lines, _ABSTRACT_RE, [_INDEX_RE]))
    keywords = extract_keywords(lines)
    doi = extract_doi(doc)

    # Fall back to the copyright/ISBN line for the year (conference papers have
    # no "VOL. .., 20.." running header).
    if year is None:
        year = _copyright_year(doc)

    # Fall back to embedded PDF metadata if heuristics miss the title.
    if not title and doc.doc.metadata.get("title"):
        title = clean_text(doc.doc.metadata["title"])

    return PaperMetadata(
        title=clean_text(title),
        authors=authors,
        abstract=abstract,
        keywords=keywords,
        doi=doi,
        venue=venue,
        year=year,
    )
