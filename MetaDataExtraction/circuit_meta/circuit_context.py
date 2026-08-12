"""Rule-based extraction of circuit-level contextual metadata.

This is the primary deliverable of the module: figuring out *what circuit the
paper is about* — its type, technology, building blocks, design purpose and a
function summary — without any schematic/netlist analysis and without an LLM.
Everything is derived from the paper's prose (title, abstract, index terms and
the proposed/design sections) and from figure captions/references.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from . import config
from .models import CircuitContext, FigureCandidate, PaperMetadata, Section
from .text_utils import flatten, split_sentences, truncate

# Cues that a sentence states a design goal / purpose.
_PURPOSE_CUES = [
    "in order to", "to enhance", "to reduce", "to improve", "to alleviate",
    "to achieve", "to minimize", "to maximize", "to increase", "to obtain",
    "is proposed", "we propose", "is presented", "suitable for", "enables",
    "can operate", "allows", "aims to", "while maintaining", "for the purpose",
]

# Spec value pattern: a number followed by a unit common in analog/RF papers.
# Energy units (fJ/pJ/...) are included so figure-of-merit metrics such as
# "30fJ/comparison" are captured, not just RF units.
_SPEC_RE = re.compile(
    r'[-−]?\d+(?:\.\d+)?\s?'
    r'(?:GHz|MHz|kHz|Hz|dBm|dBc/Hz|dBc|dB|fJ|pJ|aJ|nJ|µJ|uJ|mJ|J|'
    r'mW|µW|uW|nW|W|mV|µV|kV|V|fF|pF|nF|µF|nH|pH|µm|um|nm|ps|fs|ns|Ω|ohm|%)',
    re.IGNORECASE)

_ARTICLE_RE = re.compile(r'^\s*(a|an|the)\s+', re.IGNORECASE)
# A leading figure-of-merit token in a title, e.g. "30fJ/comparison " or
# "0.5mW ", that prefixes (not names) the circuit type.
_LEADING_FOM_RE = re.compile(
    r'^\s*[-−]?\d[\w.]*(?:/\w+)?\s+', re.IGNORECASE)
# Trailing qualifier clauses we trim off a title to leave the core circuit name.
_TITLE_TAIL_RE = re.compile(
    r'\s+(?:with|using|based on|employing|featuring|via|through|for)\s+.*$',
    re.IGNORECASE)


def _find_terms(text: str, vocab) -> List[str]:
    """Return canonical vocab entries whose surface patterns occur in text.

    ``vocab`` is either a dict canonical->patterns or a flat list of patterns.
    Matching is word-boundary aware, case-insensitive and plural-tolerant, so a
    singular pattern such as "inductor" also matches "Inductors".
    """
    low = " " + re.sub(r"\s+", " ", text.lower()) + " "
    out: List[str] = []
    if isinstance(vocab, dict):
        for canonical, patterns in vocab.items():
            if any(_matches(p, low) for p in patterns):
                out.append(canonical)
    else:
        out = [p for p in vocab if _matches(p, low)]
    return out


def _matches(pattern: str, low: str) -> bool:
    """Word-boundary, plural-tolerant search for ``pattern`` in lowered text."""
    return re.search(r'\b' + re.escape(pattern.lower()) + r'(?:e?s)?\b',
                     low) is not None


def detect_circuit_categories(text: str) -> List[str]:
    """Canonical circuit-type categories present in the text (most specific first).

    Broad-family categories are dropped whenever a specific one is also present,
    so the first entry is the most informative type name available.
    """
    cats = _find_terms(text, config.CIRCUIT_TYPES)
    specific = [c for c in cats if c not in config.GENERIC_CIRCUIT_TYPES]
    if specific:
        return specific
    return cats


def _circuit_type_phrase(title: str, abstract: str) -> str:
    """Derive a human-readable circuit-type phrase, preferring the title."""
    if title:
        core = _ARTICLE_RE.sub("", title).strip()
        core = _TITLE_TAIL_RE.sub("", core).strip(" .,")
        # Drop a leading figure-of-merit token ("30fJ/comparison Dynamic Bias
        # comparator" -> "Dynamic Bias comparator") if a type name remains.
        stripped = _LEADING_FOM_RE.sub("", core).strip(" .,")
        if stripped and detect_circuit_categories(stripped):
            core = stripped
        # Only trust the title core if it actually names a circuit type.
        if detect_circuit_categories(core):
            return core
    # Fall back to "proposed <something>" in the abstract.
    m = re.search(r'proposed\s+([A-Za-z0-9\- ]{3,60}?)\b(?:is|can|with|for|\.)',
                  abstract, re.IGNORECASE)
    if m and detect_circuit_categories(m.group(1)):
        return flatten(m.group(1))
    cats = detect_circuit_categories(title + " " + abstract)
    return cats[0] if cats else ""


def extract_purpose(text: str) -> str:
    """Pick the sentence(s) that best state the design purpose."""
    best = ""
    for sent in split_sentences(text):
        low = sent.lower()
        hits = sum(1 for cue in _PURPOSE_CUES if cue in low)
        if hits and len(sent) > len(best):
            best = sent
    return truncate(best, 320)


def extract_key_specs(text: str, limit: int = 8) -> List[str]:
    seen, out = set(), []
    for m in _SPEC_RE.finditer(text):
        val = re.sub(r"\s+", " ", m.group(0)).strip()
        key = val.lower()
        if key not in seen:
            seen.add(key)
            out.append(val)
        if len(out) >= limit:
            break
    return out


def _context_text(metadata: PaperMetadata, sections: List[Section]) -> str:
    """Concatenate the prose most likely to describe the proposed circuit."""
    parts = [metadata.title, metadata.abstract, " ".join(metadata.keywords)]
    for sec in sections:
        if sec.kind in config.CONTEXT_SECTION_KINDS:
            parts.append(sec.text)
    return "\n".join(p for p in parts if p)


def _first_sentence(text: str) -> str:
    sents = split_sentences(text)
    return sents[0] if sents else ""


def build_paper_context(metadata: PaperMetadata,
                        sections: List[Section],
                        full_text: str = "") -> CircuitContext:
    text = _context_text(metadata, sections)
    # Body text used as a fallback/supplement for vocab- and spec-mining so the
    # extractor degrades gracefully when section headers are not recognised.
    body = full_text or text
    type_phrase = _circuit_type_phrase(metadata.title, metadata.abstract)
    cats = detect_circuit_categories(metadata.title + " " + metadata.abstract)
    technology = _find_terms(text, config.TECHNOLOGY_TERMS) or _find_terms(body, config.TECHNOLOGY_TERMS)
    node = config.PROCESS_NODE_RE.findall(metadata.abstract + " " + body[:6000])
    tech_str = ", ".join(dict.fromkeys([t.upper() if len(t) <= 6 else t
                                        for t in technology]))
    if node:
        tech_str = (tech_str + (", " if tech_str else "") +
                    re.sub(r"\s+", "", node[0]) + " process")
    application = (_find_terms(text, config.APPLICATION_TERMS)
                   or _find_terms(body, config.APPLICATION_TERMS))

    # Design purpose comes from the abstract only (body-mined sentences are
    # noisy); the function summary may additionally fall back to the abstract's
    # opening sentence or, last, a body sentence.
    purpose = extract_purpose(metadata.abstract)
    # Specs: prefer the title + abstract, then top up from the body so headline
    # figures of merit (e.g. "30fJ/comparison") and measured numbers are kept.
    specs: List[str] = []
    for src in (metadata.title, metadata.abstract, body):
        for s in extract_key_specs(src, limit=12):
            if s.lower() not in {x.lower() for x in specs}:
                specs.append(s)
            if len(specs) >= 10:
                break
        if len(specs) >= 10:
            break

    summary = type_phrase or (cats[0] if cats else "circuit")
    # If no design-goal sentence was found, fall back to the abstract's opening
    # sentence (then a body sentence) so the function summary is more than just
    # the type name.
    summary_tail = purpose or _first_sentence(metadata.abstract) or extract_purpose(text)
    if summary_tail and summary_tail.lower() not in summary.lower():
        summary = f"{summary}: {summary_tail}"
    summary = truncate(summary, 360)

    # Sub-blocks: union from context prose + full body (concrete, boundary-matched
    # nouns); figure-caption sub-blocks are merged later in the pipeline.
    sub_blocks = _find_terms(text, config.SUB_BLOCKS)
    for b in _find_terms(body, config.SUB_BLOCKS):
        if b not in sub_blocks:
            sub_blocks.append(b)

    return CircuitContext(
        circuit_type=type_phrase or (cats[0] if cats else ""),
        technology=tech_str,
        application_domain=", ".join(dict.fromkeys(application)),
        sub_blocks=list(dict.fromkeys(sub_blocks)),
        design_purpose=purpose,
        function_summary=summary,
        key_specs=specs,
    )


def build_figure_context(fig: FigureCandidate,
                         paper_ctx: CircuitContext) -> CircuitContext:
    """Per-figure circuit context, derived from caption + reference sentences."""
    ref_text = " ".join(r.sentence for r in fig.references)
    blob = fig.caption + " " + ref_text

    # Circuit type/subject of THIS figure: prefer what the caption names.
    cats = detect_circuit_categories(fig.caption) or detect_circuit_categories(blob)
    # The caption itself, minus the "Fig. N." prefix, is the best subject phrase.
    subject = config.CAPTION_RE.sub("", fig.caption).strip(" .")
    subject = flatten(subject)

    sub_blocks = _find_terms(blob, config.SUB_BLOCKS)
    purpose = extract_purpose(ref_text)
    specs = extract_key_specs(ref_text)

    func = subject
    if not func and cats:
        func = cats[0]
    if purpose:
        func = truncate(f"{func}. {purpose}", 320)

    return CircuitContext(
        circuit_type=(cats[0] if cats else paper_ctx.circuit_type),
        technology=paper_ctx.technology,
        application_domain=paper_ctx.application_domain,
        sub_blocks=list(dict.fromkeys(sub_blocks)),
        design_purpose=purpose,
        function_summary=truncate(func, 320),
        key_specs=specs,
    )
