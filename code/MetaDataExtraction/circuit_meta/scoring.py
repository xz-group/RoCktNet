from __future__ import annotations

import re
from typing import List, Optional

from . import config
from .models import AnalysisResult, CircuitContext, FigureCandidate

_SUB_BLOCK_NAMES = ["buffer", "output buffer", "bias", "driver", "mixer",
                    "varactor", "current mirror", "current source",
                    "balun", "transformer", "divider", "prescaler", "switch"]
_SCHEMATIC_WORDS = ["schematic", "topology", "circuit diagram", "block diagram",
                    "architecture", "circuit"]


def _caption_body(caption: str) -> str:
    return config.CAPTION_RE.sub("", caption).strip().lower()


def classify_role(caption: str) -> str:
    low = _caption_body(caption)

    def has(*words):
        return any(w in low for w in words)

    if has("conventional", "prior art", "previously", "reference design"):
        return "conventional"
    if has("microphotograph", "photograph", "micrograph", "die photo",
           "chip photo", "chip micrograph"):
        return "layout_photo"
    if has("simulated", "measured", "extracted", "calculated", "spectrum",
           "phase noise", "tuning range", "output power", "q factor",
           "capacitance", "conductance", "waveform", "vs.", "versus", "setup"):
        return "result_plot"
    if has("structure of", "layout", "cross section", "cross-section"):
        return "layout_photo"
    if has("equivalent", "half circuit", "model", "simplified"):
        return "analysis_model"
    if "proposed" in low and has(*_SCHEMATIC_WORDS):
        return "proposed_schematic"
    if has("topology", "architecture"):
        return "proposed_schematic"
    if has("schematic", "circuit diagram", "block diagram") or "circuit" in low:
        if has(*_SUB_BLOCK_NAMES):
            return "sub_block_schematic"
        return "proposed_schematic"
    return "other"


def score_figure(fig: FigureCandidate, paper_ctx: CircuitContext) -> None:
    low = _caption_body(fig.caption)
    breakdown: dict[str, float] = {}

    pos = 0.0
    for kw, w in config.CAPTION_POSITIVE.items():
        if kw in low:
            pos += w
            breakdown[f"+{kw}"] = w
    neg = 0.0
    for kw, w in config.CAPTION_NEGATIVE.items():
        if kw in low:
            neg += w
            breakdown[f"-{kw}"] = w

    sec_bonus = config.SECTION_KIND_BONUS.get(fig.section_kind, 0.0)
    if sec_bonus:
        breakdown[f"section:{fig.section_kind}"] = sec_bonus

    # Matching the paper's headline circuit type in the caption is a strong cue.
    type_bonus = 0.0
    for cat in _category_tokens(paper_ctx.circuit_type):
        if re.search(r'\b' + re.escape(cat) + r'\b', low):
            type_bonus = 2.0
            breakdown[f"type_match:{cat}"] = 2.0
            break

    ref_bonus = min(config.REF_COUNT_CAP,
                    config.REF_COUNT_WEIGHT * len(fig.references))
    if ref_bonus:
        breakdown["references"] = round(ref_bonus, 2)

    total = pos + neg + sec_bonus + type_bonus + ref_bonus
    fig.relevance_score = round(total, 2)
    fig.relevance_breakdown = {k: round(v, 2) for k, v in breakdown.items()}
    fig.figure_role = classify_role(fig.caption)


def _category_tokens(circuit_type: str) -> List[str]:
    toks = set()
    low = circuit_type.lower()
    for m in re.findall(r'\(([a-z0-9]{2,6})\)', low):  # acronyms in parentheses
        toks.add(m)
    for word in re.findall(r'[a-z]{3,}', low):
        if word in {"voltage", "controlled", "oscillator", "amplifier", "mixer",
                    "rectifier", "comparator", "filter", "divider"}:
            toks.add(word)
    return list(toks)


def _recommendation_confidence(figs: List[FigureCandidate],
                               best: FigureCandidate) -> float:
    others = sorted((f.relevance_score for f in figs if f is not best),
                    reverse=True)
    second = others[0] if others else 0.0
    gap = best.relevance_score - second
    conf = 0.2
    if best.relevance_score >= config.CONF_MIN_TOP_SCORE:
        conf = 0.45
    conf += min(0.35, max(0.0, gap) / config.CONF_STRONG_GAP * 0.35)
    if best.figure_role == "proposed_schematic":
        conf += 0.2
    elif best.figure_role == "sub_block_schematic":
        conf += 0.05
    return round(min(1.0, conf), 2)


def rank_and_recommend(result: AnalysisResult) -> None:
    figs = result.figures
    for fig in figs:
        score_figure(fig, result.paper_circuit_context)
        # Per-figure confidence: monotonic in its own score.
        fig.confidence = round(1.0 / (1.0 + pow(2.71828, -(fig.relevance_score - 4.0))), 2)

    if not figs:
        result.warnings.append("No figures with captions were detected.")
        return

    # Prefer schematic-like figures with positive score; else fall back to global max.
    schematic_like = [f for f in figs
                      if f.figure_role in ("proposed_schematic", "sub_block_schematic")
                      and f.relevance_score > 0]
    pool = schematic_like or figs
    best = max(pool, key=lambda f: f.relevance_score)

    if best.relevance_score <= 0:
        result.warnings.append(
            "No figure scored positively as a proposed-circuit schematic; "
            "recommendation is low-confidence.")

    best.is_recommended = True
    result.recommended_figure_id = best.id
    best.confidence = _recommendation_confidence(figs, best)

    _emit_warnings(result, best, figs)


def _emit_warnings(result: AnalysisResult, best: FigureCandidate,
                   figs: List[FigureCandidate]) -> None:
    low = _caption_body(best.caption)
    if not any(w in low for w in _SCHEMATIC_WORDS):
        best.warnings.append(
            "Recommended figure caption does not explicitly mention a "
            "schematic/topology/circuit; please confirm manually.")
    if best.figure_role not in ("proposed_schematic", "sub_block_schematic"):
        best.warnings.append(
            f"Recommended figure was classified as '{best.figure_role}', "
            "not a schematic; manual confirmation strongly recommended.")
    others = sorted((f.relevance_score for f in figs if f is not best), reverse=True)
    second = others[0] if others else 0.0
    if best.relevance_score - second < 1.0 and len(figs) > 1:
        result.warnings.append(
            "Small score margin between the top figure candidates; "
            "manual confirmation recommended.")
