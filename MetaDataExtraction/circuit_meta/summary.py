"""Build a concise, human-readable summary in the style of a hand-curated
``info.txt`` — focused on the recommended (proposed-circuit) figure.

The full ``analysis.json`` stays as the machine/downstream contract; this is the
trimmed view meant for reading and for pasting into a paper. Only sections we can
genuinely extract are emitted (no mechanism-level "function detail" and no
"highlight metrics", since those require understanding rather than extraction).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

# unit -> metric-name mapping, longest/most-specific units first.
_UNIT_TO_METRIC = [
    ("dBc/Hz", "Phase noise"),
    ("dBm", "Output power"),
    ("dBc", "Spurious / phase noise"),
    ("dB", "Gain"),
    ("fJ", "Energy per operation"),
    ("pJ", "Energy per operation"),
    ("aJ", "Energy per operation"),
    ("nJ", "Energy per operation"),
    ("µJ", "Energy per operation"),
    ("uJ", "Energy per operation"),
    ("mJ", "Energy per operation"),
    ("J", "Energy per operation"),
    ("ps", "Timing / delay"),
    ("fs", "Timing / delay"),
    ("ns", "Timing / delay"),
    ("GHz", "Operating / oscillation frequency"),
    ("MHz", "Frequency"),
    ("kHz", "Frequency"),
    ("mW", "Power consumption"),
    ("µW", "Power consumption"),
    ("uW", "Power consumption"),
    ("nW", "Power consumption"),
    ("W", "Power consumption"),
    ("mV", "Supply / voltage"),
    ("kV", "Supply / voltage"),
    ("V", "Supply voltage"),
    ("nH", "Inductance"),
    ("pH", "Inductance"),
    ("fF", "Capacitance"),
    ("pF", "Capacitance"),
    ("nF", "Capacitance"),
    ("Ω", "Impedance"),
    ("ohm", "Impedance"),
    ("nm", "Process node"),
    ("µm", "Process node"),
    ("um", "Process node"),
    ("%", "Efficiency / ratio"),
]


def _dedup_terms(csv: str) -> str:
    """Drop near-duplicate comma-separated terms (hyphen/space/case variants)."""
    seen, out = set(), []
    for term in (t.strip() for t in csv.split(",")):
        if not term:
            continue
        norm = term.lower().replace("-", "").replace(" ", "")
        if norm not in seen:
            seen.add(norm)
            out.append(term)
    return ", ".join(out)


def _metrics_from_specs(specs: List[str]) -> List[str]:
    """Derive the named metric categories implied by the numeric spec values."""
    names: List[str] = []
    for spec in specs:
        # Units are ordered longest/most-specific first, so the first endswith
        # match wins (e.g. "mV" before "V", "dBc/Hz" before "dB").
        for unit, name in _UNIT_TO_METRIC:
            if spec.endswith(unit):
                if name not in names:
                    names.append(name)
                break
    return names


def build_summary(analysis: Dict[str, Any],
                  fig_id: Optional[int] = None) -> Dict[str, Any]:
    """Assemble the concise summary dict from a full analysis dict.

    Focuses on the recommended figure (or ``fig_id`` if given); falls back to
    paper-level context for fields a single figure does not carry.
    """
    if fig_id is None:
        fig_id = analysis.get("recommended_figure_id")
    fig = next((f for f in analysis.get("figures", []) if f.get("id") == fig_id), None)

    meta = analysis.get("paper_metadata", {})
    paper_ctx = analysis.get("paper_circuit_context", {})
    fig_ctx = (fig or {}).get("circuit_context") or {}

    figure_number = ""
    if fig:
        figure_number = fig.get("caption") or fig.get("label", "")

    circuit_type = paper_ctx.get("circuit_type", "") or fig_ctx.get("circuit_type", "")
    sub_blocks = paper_ctx.get("sub_blocks", []) or fig_ctx.get("sub_blocks", [])
    # Prefer the richer paper-level summary (type + design intent) over the
    # figure caption, which would just echo FIGURE NUMBER.
    function_summary = (paper_ctx.get("function_summary")
                        or fig_ctx.get("function_summary", ""))
    design_purpose = (paper_ctx.get("design_purpose")
                      or fig_ctx.get("design_purpose", ""))
    # Avoid duplication: the paper function summary already embeds the purpose.
    if design_purpose and design_purpose in function_summary:
        design_purpose = ""
    specs = paper_ctx.get("key_specs", []) or fig_ctx.get("key_specs", [])

    warnings = list(analysis.get("warnings", [])) + list((fig or {}).get("warnings", []))

    return {
        "title": meta.get("title", ""),
        "bibliography": {
            "authors": meta.get("authors", []),
            "venue": meta.get("venue", ""),
            "year": meta.get("year"),
            "doi": meta.get("doi", ""),
            "keywords": meta.get("keywords", []),
        },
        "figure_number": figure_number,
        "type": circuit_type,
        "technology": paper_ctx.get("technology", ""),
        "application": _dedup_terms(paper_ctx.get("application_domain", "")),
        "sub_blocks": sub_blocks,
        "function_summary": function_summary,
        "design_purpose": design_purpose,
        "metrics": _metrics_from_specs(specs),
        "numerical_performance": specs,
        "confidence": (fig or {}).get("confidence", 0.0),
        "warnings": warnings,
    }


def render_summary_text(s: Dict[str, Any]) -> str:
    """Render the summary in the plain ``info.txt`` section style."""
    def block(title: str, value: str) -> str:
        return f"{title}\n{value}\n"

    def quoted(v: str) -> str:
        return json.dumps(v, ensure_ascii=False)

    def arr(items: List[str]) -> str:
        if not items:
            return "[]"
        body = ",\n".join(f"  {json.dumps(i, ensure_ascii=False)}" for i in items)
        return "[\n" + body + "\n]"

    out: List[str] = []
    if s.get("title"):
        out.append(f"# {s['title']}\n")
    bib = s.get("bibliography", {})
    if any(bib.get(k) for k in ("authors", "venue", "year", "doi", "keywords")):
        if bib.get("authors"):
            out.append(block("AUTHORS", arr(bib["authors"])))
        venue_year = ", ".join(str(v) for v in (bib.get("venue"), bib.get("year")) if v)
        if venue_year:
            out.append(block("VENUE", quoted(venue_year)))
        if bib.get("doi"):
            out.append(block("DOI", quoted(bib["doi"])))
        if bib.get("keywords"):
            out.append(block("KEYWORDS", arr(bib["keywords"])))
    out.append(block("FIGURE NUMBER", quoted(s.get("figure_number", ""))))
    out.append(block("TYPE", quoted(s.get("type", ""))))
    if s.get("technology"):
        out.append(block("TECHNOLOGY", quoted(s["technology"])))
    if s.get("application"):
        out.append(block("APPLICATION", quoted(s["application"])))
    out.append(block("SUB-BLOCKS", arr(s.get("sub_blocks", []))))
    if s.get("function_summary"):
        out.append(block("FUNCTION SUMMARY", quoted(s["function_summary"])))
    if s.get("design_purpose"):
        out.append(block("DESIGN PURPOSE", quoted(s["design_purpose"])))
    out.append(block("METRICS", arr(s.get("metrics", []))))
    out.append(block("NUMERICAL PERFORMANCE", arr(s.get("numerical_performance", []))))
    out.append(block("CONFIDENCE", str(s.get("confidence", 0.0))))
    if s.get("warnings"):
        out.append(block("WARNINGS", arr(s["warnings"])))
    return "\n".join(out).rstrip() + "\n"


def write_summary(analysis: Dict[str, Any], out_dir: str,
                  fig_id: Optional[int] = None) -> Dict[str, str]:
    """Write summary.txt (info.txt style) and summary.json; return their paths."""
    os.makedirs(out_dir, exist_ok=True)
    s = build_summary(analysis, fig_id=fig_id)
    txt_path = os.path.join(out_dir, "summary.txt")
    json_path = os.path.join(out_dir, "summary.json")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(render_summary_text(s))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2, ensure_ascii=False)
    return {"summary_txt": txt_path, "summary_json": json_path}
