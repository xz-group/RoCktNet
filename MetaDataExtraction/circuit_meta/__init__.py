"""circuit_meta — paper-to-circuit-context + figure package preparation.

A pre-parsing module for circuit papers. It extracts paper-level metadata and
circuit-level contextual metadata (circuit type, sub-blocks, design purpose,
function summary), locates and saves figures, recommends the figure most likely
to be the proposed circuit schematic, and packages a user-selected figure with
its context for a downstream schematic-to-netlist pipeline.

It does NOT perform schematic-to-netlist / device-level analysis.
"""
from .pipeline import analyze_pdf
from .selection import (select_from_analysis, select_from_json,
                        select_from_dict, build_package_dict)
from .summary import build_summary, render_summary_text, write_summary
from .models import (AnalysisResult, PaperMetadata, CircuitContext,
                     FigureCandidate, FigureReference, Section)

__all__ = [
    "analyze_pdf",
    "select_from_analysis", "select_from_json", "select_from_dict",
    "build_package_dict",
    "build_summary", "render_summary_text", "write_summary",
    "AnalysisResult", "PaperMetadata", "CircuitContext",
    "FigureCandidate", "FigureReference", "Section",
]

__version__ = "0.1.0"
