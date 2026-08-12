"""Text cleaning and sentence utilities for noisy PDF-extracted text."""
from __future__ import annotations

import re
from typing import List

from . import config


_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def is_noise_line(text: str) -> bool:
    """True if a line is page furniture (headers/footers/licensing banners)."""
    t = text.strip()
    if not t:
        return True
    for pat in config.NOISE_PATTERNS:
        if pat.search(t):
            return True
    return False


def clean_text(text: str) -> str:
    """Normalise whitespace and strip unicode artefacts from PDF extraction."""
    if not text:
        return ""
    # Drop the unicode replacement char (missing glyphs / math symbols).
    text = text.replace("�", "")
    # Symbol-font glyphs (Σ, ∆, µ, ...) often extract as raw control codes;
    # they carry no readable text, so drop them (keep \t and \n).
    text = _CONTROL_RE.sub("", text)
    # Common ligature / spacing fixes.
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = text.replace("­", "")  # soft hyphen
    text = text.replace("‐", "-").replace("‑", "-")
    text = text.replace(" ", " ").replace(" ", " ")
    # Collapse whitespace.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def dehyphenate(text: str) -> str:
    """Join words split across line breaks ("admit-\ntance" -> "admittance")."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)  # also "admit- tance"
    return text


def flatten(text: str) -> str:
    """Collapse a multi-line block into a single dehyphenated line."""
    text = _CONTROL_RE.sub("", dehyphenate(text))
    return re.sub(r"\s+", " ", text).strip()


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter tolerant of abbreviations like 'Fig.'."""
    text = flatten(text)
    if not text:
        return []
    # Protect common abbreviations so we don't split on their periods.
    guarded = text
    for abbr in ["Fig.", "Figs.", "Eq.", "Ref.", "vs.", "etc.", "e.g.", "i.e.",
                 "Sec.", "approx.", "No.", "Vol.", "pp.", "Dr.", "et al."]:
        guarded = guarded.replace(abbr, abbr.replace(".", ""))
    parts = _SENT_SPLIT.split(guarded)
    return [p.replace("", ".").strip() for p in parts if p.strip()]


def sentence_containing(text: str, needle_regex: re.Pattern, fig_id: int) -> List[str]:
    """Return sentences mentioning a figure number ``fig_id``."""
    out = []
    for sent in split_sentences(text):
        for m in needle_regex.finditer(sent):
            try:
                if int(m.group(1)) == fig_id:
                    out.append(sent)
                    break
            except (ValueError, IndexError):
                continue
    return out


def truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + " ..."
