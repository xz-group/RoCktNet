"""Build the 'selected package' handed to the downstream schematic-to-netlist
pipeline once a user confirms which figure is the target circuit.

The package bundles the chosen figure image together with the circuit-level
contextual metadata. It can be produced directly from an in-memory
``AnalysisResult`` or, for a decoupled two-step UI flow, from a previously
written ``analysis.json`` file.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
from typing import Any, Dict, Optional

from .models import AnalysisResult
from .pdf_doc import PdfDocument


def _merge_unique(*lists) -> list:
    out: list = []
    for lst in lists:
        for item in (lst or []):
            if item not in out:
                out.append(item)
    return out


def _ensure_image(pdf_path: str, page: int, crop_bbox, src_image: str,
                  dst_image: str) -> str:
    """Copy an existing crop, or render one from the PDF using crop_bbox."""
    if src_image and os.path.isfile(src_image):
        shutil.copyfile(src_image, dst_image)
        return dst_image
    if pdf_path and os.path.isfile(pdf_path) and crop_bbox:
        doc = PdfDocument(pdf_path)
        try:
            doc.render_clip(page, tuple(crop_bbox), dst_image)
        finally:
            doc.close()
        return dst_image
    return ""


def build_package_dict(analysis: Dict[str, Any], fig_id: int) -> Dict[str, Any]:
    """Assemble the selected-package dict from an analysis dict + figure id."""
    fig = next((f for f in analysis.get("figures", []) if f.get("id") == fig_id), None)
    if fig is None:
        raise ValueError(f"Figure id {fig_id} not found in analysis "
                         f"(available: {[f.get('id') for f in analysis.get('figures', [])]})")

    meta = analysis.get("paper_metadata", {})
    paper_ctx = analysis.get("paper_circuit_context", {})
    fig_ctx = fig.get("circuit_context") or {}

    # Per-figure context wins for the figure-specific fields, paper context fills gaps.
    circuit_type = fig_ctx.get("circuit_type") or paper_ctx.get("circuit_type", "")
    target_circuit_type = paper_ctx.get("circuit_type", "") or circuit_type
    sub_blocks = _merge_unique(fig_ctx.get("sub_blocks"), paper_ctx.get("sub_blocks"))
    design_purpose = fig_ctx.get("design_purpose") or paper_ctx.get("design_purpose", "")
    function_summary = fig_ctx.get("function_summary") or paper_ctx.get("function_summary", "")

    warnings = list(analysis.get("warnings", [])) + list(fig.get("warnings", []))

    return {
        "schema": "circuit-context-selected-package/v1",
        "for_downstream": "schematic-to-netlist",
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "source_pdf": analysis.get("pdf_path", ""),
        "selected_figure": {
            "id": fig.get("id"),
            "label": fig.get("label"),
            "page": fig.get("page"),
            "image": "",  # filled in by select_*()
            "caption": fig.get("caption", ""),
            "section_title": fig.get("section_title", ""),
            "figure_role": fig.get("figure_role", ""),
            "crop_bbox": fig.get("crop_bbox", []),
            "relevance_score": fig.get("relevance_score"),
            "was_recommended": fig.get("is_recommended", False),
        },
        "paper": {
            "title": meta.get("title", ""),
            "authors": meta.get("authors", []),
            "abstract": meta.get("abstract", ""),
            "keywords": meta.get("keywords", []),
            "venue": meta.get("venue", ""),
            "year": meta.get("year"),
            "doi": meta.get("doi", ""),
        },
        "circuit_context": {
            "target_circuit_type": target_circuit_type,
            "figure_circuit_type": circuit_type,
            "technology": paper_ctx.get("technology", ""),
            "application_domain": paper_ctx.get("application_domain", ""),
            "sub_blocks": sub_blocks,
            "design_purpose": design_purpose,
            "function_summary": function_summary,
            "key_specs": _merge_unique(fig_ctx.get("key_specs"), paper_ctx.get("key_specs")),
            "related_caption": fig.get("caption", ""),
        },
        "confidence": fig.get("confidence", 0.0),
        "warnings": warnings,
    }


def select_from_dict(analysis: Dict[str, Any], fig_id: int,
                     out_dir: str) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    pkg = build_package_dict(analysis, fig_id)

    fig = next(f for f in analysis["figures"] if f.get("id") == fig_id)
    dst_image = os.path.join(out_dir, f"selected_fig_{fig_id:02d}.png")
    image = _ensure_image(analysis.get("pdf_path", ""), fig.get("page", 0),
                          fig.get("crop_bbox", []), fig.get("image_path", ""),
                          dst_image)
    pkg["selected_figure"]["image"] = os.path.abspath(image) if image else ""
    if not image:
        pkg["warnings"].append("Selected figure image could not be produced.")

    pkg_path = os.path.join(out_dir, "selected_package.json")
    with open(pkg_path, "w", encoding="utf-8") as f:
        json.dump(pkg, f, indent=2, ensure_ascii=False)
    pkg["_package_path"] = os.path.abspath(pkg_path)
    return pkg


def select_from_analysis(result: AnalysisResult, fig_id: int,
                         out_dir: Optional[str] = None) -> Dict[str, Any]:
    if out_dir is None:
        out_dir = os.path.join(result.output_dir, "selected")
    return select_from_dict(result.to_dict(), fig_id, out_dir)


def select_from_json(analysis_json_path: str, fig_id: int,
                     out_dir: Optional[str] = None) -> Dict[str, Any]:
    with open(analysis_json_path, "r", encoding="utf-8") as f:
        analysis = json.load(f)
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(analysis_json_path), "selected")
    return select_from_dict(analysis, fig_id, out_dir)
