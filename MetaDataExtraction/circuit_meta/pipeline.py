"""End-to-end orchestration: PDF -> analysis result (JSON + figure images)."""
from __future__ import annotations

import datetime as _dt
import json
import os
from typing import Optional

from . import circuit_context as cc
from . import figures as fg
from .models import AnalysisResult
from .paper_metadata import extract_paper_metadata
from .pdf_doc import PdfDocument
from .scoring import rank_and_recommend
from .sections import extract_sections
from .summary import write_summary


def analyze_pdf(pdf_path: str, output_dir: Optional[str] = None,
                save_figures: bool = True) -> AnalysisResult:
    """Run the full paper-to-circuit-context extraction on a PDF."""
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(pdf_path)
    if output_dir is None:
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        output_dir = os.path.join("output", base)
    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, "figures")

    doc = PdfDocument(pdf_path)
    try:
        result = AnalysisResult(
            pdf_path=os.path.abspath(pdf_path),
            output_dir=os.path.abspath(output_dir),
            generated_at=_dt.datetime.now().isoformat(timespec="seconds"),
        )

        # 1. Paper-level metadata.
        result.paper_metadata = extract_paper_metadata(doc)

        # 2. Sections (reading-order aware).
        result.sections = extract_sections(doc)

        # 3. Paper-level circuit context.
        full_text = doc.full_text()
        result.paper_circuit_context = cc.build_paper_context(
            result.paper_metadata, result.sections, full_text)

        # 4. Figures: captions, section assignment, references, crops, context.
        figs = fg.detect_captions(doc)
        for fig in figs:
            fg.assign_section(fig, result.sections)
            fg.attach_references(fig, result.sections, full_text)
            fig.circuit_context = cc.build_figure_context(
                fig, result.paper_circuit_context)
            if save_figures:
                fig.image_path = fg.crop_figure(doc, fig, fig_dir)
        result.figures = figs

        # 5. Merge figure-caption sub-blocks into the paper-level context.
        _merge_sub_blocks(result)

        # 6. Score + recommend.
        rank_and_recommend(result)

        # 7. Persist the full analysis (machine/downstream contract) and a
        #    concise human-readable summary (info.txt style) for the recommendation.
        analysis_dict = result.to_dict()
        analysis_path = os.path.join(output_dir, "analysis.json")
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(analysis_dict, f, indent=2, ensure_ascii=False)
        write_summary(analysis_dict, output_dir)

        return result
    finally:
        doc.close()


def _merge_sub_blocks(result: AnalysisResult) -> None:
    blocks = list(result.paper_circuit_context.sub_blocks)
    for fig in result.figures:
        if fig.circuit_context:
            for b in fig.circuit_context.sub_blocks:
                if b not in blocks:
                    blocks.append(b)
    result.paper_circuit_context.sub_blocks = blocks
