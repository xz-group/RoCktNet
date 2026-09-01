from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class PaperMetadata:
    title: str = ""
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    keywords: List[str] = field(default_factory=list)
    doi: str = ""
    venue: str = ""
    year: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Section:
    number: str = ""          # e.g. "III"
    title: str = ""           # e.g. "PROPOSED TOPOLOGY OF VCO"
    kind: str = "other"       # intro | proposed | design | results | conclusion | other
    page: int = 0             # 0-based page index where the header sits
    header_y: float = 0.0     # y-coordinate of the header line on its page
    order: int = 0            # reading-order index of the header among all lines
    text: str = ""            # body text of the section (cleaned)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # body text can be large; expose a trimmed preview in JSON
        d["text_preview"] = (self.text[:400] + " ...") if len(self.text) > 400 else self.text
        del d["text"]
        del d["order"]
        del d["header_y"]
        return d


@dataclass
class FigureReference:
    section_title: str = ""
    section_kind: str = "other"
    page: int = 0
    sentence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CircuitContext:
    circuit_type: str = ""
    technology: str = ""
    application_domain: str = ""
    sub_blocks: List[str] = field(default_factory=list)
    design_purpose: str = ""
    function_summary: str = ""
    key_specs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FigureCandidate:
    id: int = 0                       # figure number, e.g. 4
    label: str = ""                   # "Fig. 4"
    page: int = 0                     # 0-based page index of the caption
    caption: str = ""
    caption_bbox: List[float] = field(default_factory=list)  # x0,y0,x1,y1
    section_title: str = ""
    section_kind: str = "other"
    references: List[FigureReference] = field(default_factory=list)
    image_path: str = ""
    crop_bbox: List[float] = field(default_factory=list)
    relevance_score: float = 0.0
    relevance_breakdown: Dict[str, float] = field(default_factory=dict)
    figure_role: str = "other"        # proposed_schematic | sub_block_schematic | analysis_model | conventional | result_plot | layout_photo | other
    circuit_context: Optional[CircuitContext] = None
    is_recommended: bool = False
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["references"] = [r.to_dict() for r in self.references]
        d["circuit_context"] = self.circuit_context.to_dict() if self.circuit_context else None
        return d


@dataclass
class AnalysisResult:
    pdf_path: str = ""
    output_dir: str = ""
    generated_at: str = ""
    paper_metadata: PaperMetadata = field(default_factory=PaperMetadata)
    paper_circuit_context: CircuitContext = field(default_factory=CircuitContext)
    sections: List[Section] = field(default_factory=list)
    figures: List[FigureCandidate] = field(default_factory=list)
    recommended_figure_id: Optional[int] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pdf_path": self.pdf_path,
            "output_dir": self.output_dir,
            "generated_at": self.generated_at,
            "paper_metadata": self.paper_metadata.to_dict(),
            "paper_circuit_context": self.paper_circuit_context.to_dict(),
            "sections": [s.to_dict() for s in self.sections],
            "figures": [f.to_dict() for f in self.figures],
            "recommended_figure_id": self.recommended_figure_id,
            "warnings": self.warnings,
        }

    def figure_by_id(self, fig_id: int) -> Optional[FigureCandidate]:
        for f in self.figures:
            if f.id == fig_id:
                return f
        return None
