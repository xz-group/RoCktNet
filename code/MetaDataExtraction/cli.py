#!/usr/bin/env python
"""Command-line interface for the paper-to-circuit-context module.

Two subcommands mirror the intended workflow:

    analyze   PDF -> analysis.json + figures/*.png + recommendation
    select    analysis.json + figure-id -> selected_package.json + image

Examples
--------
    python cli.py analyze Data/10.1109_tmtt.2007.903347.pdf
    python cli.py select output/10.1109_tmtt.2007.903347/analysis.json --figure 4
"""
from __future__ import annotations

import argparse
import sys

from circuit_meta import analyze_pdf, select_from_json


def _print_summary(result) -> None:
    m = result.paper_metadata
    ctx = result.paper_circuit_context
    print("=" * 70)
    print(f"TITLE   : {m.title}")
    print(f"AUTHORS : {', '.join(m.authors)}")
    print(f"VENUE   : {m.venue}  ({m.year})")
    print(f"DOI     : {m.doi}")
    print(f"KEYWORDS: {', '.join(m.keywords)}")
    print("-" * 70)
    print(f"CIRCUIT TYPE  : {ctx.circuit_type}")
    print(f"TECHNOLOGY    : {ctx.technology}")
    print(f"APPLICATION   : {ctx.application_domain}")
    print(f"SUB-BLOCKS    : {', '.join(ctx.sub_blocks)}")
    print(f"DESIGN PURPOSE: {ctx.design_purpose}")
    print(f"KEY SPECS     : {', '.join(ctx.key_specs)}")
    print("-" * 70)
    print(f"FIGURES ({len(result.figures)}):")
    for f in sorted(result.figures, key=lambda x: -x.relevance_score):
        star = "  <== RECOMMENDED" if f.is_recommended else ""
        print(f"  {f.label:<8} score={f.relevance_score:6.2f} "
              f"role={f.figure_role:<20} | {f.caption[:60]}{star}")
    print("-" * 70)
    if result.recommended_figure_id is not None:
        rec = result.figure_by_id(result.recommended_figure_id)
        print(f"RECOMMENDED   : {rec.label} (confidence={rec.confidence}) "
              f"-> {rec.caption}")
    if result.warnings:
        print("WARNINGS:")
        for w in result.warnings:
            print(f"  - {w}")
    print(f"\nWrote: {result.output_dir}\\analysis.json  (full / downstream)")
    print(f"       {result.output_dir}\\summary.txt   (concise / human-readable)")
    print("=" * 70)


def cmd_analyze(args) -> int:
    result = analyze_pdf(args.pdf, output_dir=args.out,
                         save_figures=not args.no_figures)
    _print_summary(result)
    return 0


def cmd_select(args) -> int:
    pkg = select_from_json(args.analysis, args.figure, out_dir=args.out)
    print("=" * 70)
    print(f"SELECTED: {pkg['selected_figure']['label']} "
          f"(role={pkg['selected_figure']['figure_role']}, "
          f"confidence={pkg['confidence']})")
    print(f"CAPTION : {pkg['selected_figure']['caption']}")
    print(f"TYPE    : {pkg['circuit_context']['target_circuit_type']}")
    print(f"SUB-BLK : {', '.join(pkg['circuit_context']['sub_blocks'])}")
    print(f"IMAGE   : {pkg['selected_figure']['image']}")
    print(f"PACKAGE : {pkg['_package_path']}")
    if pkg["warnings"]:
        print("WARNINGS:")
        for w in pkg["warnings"]:
            print(f"  - {w}")
    print("=" * 70)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="circuit_meta",
        description="Paper-to-circuit-context + figure package preparation.")
    sub = p.add_subparsers(dest="command", required=True)

    pa = sub.add_parser("analyze", help="Analyze a PDF paper.")
    pa.add_argument("pdf", help="Path to the circuit paper PDF.")
    pa.add_argument("--out", default=None,
                    help="Output directory (default: output/<pdf-stem>).")
    pa.add_argument("--no-figures", action="store_true",
                    help="Skip cropping/saving figure images.")
    pa.set_defaults(func=cmd_analyze)

    ps = sub.add_parser("select", help="Build the downstream package for a figure.")
    ps.add_argument("analysis", help="Path to a previously written analysis.json.")
    ps.add_argument("--figure", "-f", type=int, required=True,
                    help="Figure number to select (e.g. 4).")
    ps.add_argument("--out", default=None,
                    help="Output directory (default: <analysis-dir>/selected).")
    ps.set_defaults(func=cmd_select)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
