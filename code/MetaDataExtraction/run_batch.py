"""Batch driver: run circuit_meta.analyze_pdf over every PDF in a directory.

Per-file failures are caught and logged so one bad PDF doesn't abort the run.
Figure cropping is skipped by default (text/caption-based metadata only);
pass --save-figures to also crop and save the figure images.

Two kinds of problems are recorded to <out>/_report/:
  * hard failures  - analyze_pdf raised (no output produced at all)
  * incomplete     - output produced, but critical fields came out empty
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import traceback

from circuit_meta import analyze_pdf

# Critical fields: if these are empty the extraction is not usable downstream.
CRITICAL_CHECKS = ("no_title", "no_sections", "no_figures", "no_circuit_type",
                   "no_recommendation")


def check_quality(result) -> list:
    """Return a list of issue tags for an otherwise-successful extraction."""
    issues = []
    meta = result.paper_metadata
    ctx = result.paper_circuit_context
    if not (meta and meta.title.strip()):
        issues.append("no_title")
    if not result.sections:
        issues.append("no_sections")
    if not result.figures:
        issues.append("no_figures")
    if not (ctx and ctx.circuit_type.strip()):
        issues.append("no_circuit_type")
    if result.recommended_figure_id is None:
        issues.append("no_recommendation")
    # Non-critical, informational only.
    if not (meta and meta.authors):
        issues.append("no_authors")
    if not (meta and meta.abstract.strip()):
        issues.append("no_abstract")
    if not (ctx and ctx.key_specs):
        issues.append("no_key_specs")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-run circuit_meta.analyze_pdf over all PDFs in a directory.")
    parser.add_argument("data_dir", nargs="?", default="Data",
                        help="Directory containing PDFs (default: Data)")
    parser.add_argument("--out", default="Output",
                        help="Output root directory; each PDF gets its own "
                             "<out>/<paper_id>/ subfolder (default: Output)")
    parser.add_argument("--save-figures", action="store_true",
                        help="Also crop and save figure images (default: skip)")
    parser.add_argument("--only", default=None,
                        help="Comma-separated list of paper ids to process "
                             "(default: every PDF in data_dir)")
    parser.add_argument("--only-file", default=None,
                        help="File of paper ids to process (comma- or "
                             "newline-separated); useful for re-running failures")
    parser.add_argument("--rerun-failed-from", default=None,
                        help="Path to a previous extraction_report.csv; "
                             "re-runs only the papers that had any issue")
    parser.add_argument("--report-dir", default=None,
                        help="Where to write the run report "
                             "(default: <out>/_report)")
    args = parser.parse_args()

    pdf_paths = sorted(glob.glob(os.path.join(args.data_dir, "*.pdf")))

    wanted = None
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
    elif args.only_file:
        with open(args.only_file, encoding="utf-8") as f:
            wanted = {s.strip() for s in f.read().replace("\n", ",").split(",")
                      if s.strip()}
    elif args.rerun_failed_from:
        with open(args.rerun_failed_from, newline="", encoding="utf-8") as f:
            wanted = {r["paper_id"] for r in csv.DictReader(f)
                      if r["issues"].strip() or r["status"] != "ok"}
    if wanted is not None:
        pdf_paths = [p for p in pdf_paths
                     if os.path.splitext(os.path.basename(p))[0] in wanted]
        print(f"Re-running {len(pdf_paths)} of {len(wanted)} requested papers.")
    if not pdf_paths:
        print(f"No PDFs found in {args.data_dir}")
        return

    report_dir = args.report_dir or os.path.join(args.out, "_report")
    os.makedirs(report_dir, exist_ok=True)

    rows = []          # per-paper record for the report
    failed = []        # hard failures
    incomplete = []    # produced output but missing critical fields

    for i, pdf_path in enumerate(pdf_paths, start=1):
        stem = os.path.splitext(os.path.basename(pdf_path))[0]
        out_dir = os.path.join(args.out, stem)
        print(f"[{i}/{len(pdf_paths)}] {stem} ...", end=" ")
        try:
            result = analyze_pdf(pdf_path, output_dir=out_dir,
                                 save_figures=args.save_figures)
        except Exception as exc:
            failed.append({"paper_id": stem, "pdf": pdf_path,
                           "error": f"{type(exc).__name__}: {exc}",
                           "traceback": traceback.format_exc()})
            rows.append({"paper_id": stem, "status": "failed",
                         "issues": "", "figures": "",
                         "sections": "", "title": "",
                         "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAILED: {type(exc).__name__}: {exc}")
            continue

        issues = check_quality(result)
        critical = [t for t in issues if t in CRITICAL_CHECKS]
        status = "incomplete" if critical else "ok"
        if critical:
            incomplete.append({"paper_id": stem, "pdf": pdf_path,
                               "issues": ";".join(issues)})
        rows.append({
            "paper_id": stem,
            "status": status,
            "issues": ";".join(issues),
            "figures": len(result.figures),
            "sections": len(result.sections),
            "title": (result.paper_metadata.title if result.paper_metadata else ""),
            "error": "",
        })
        print(status if not issues else f"{status} ({';'.join(issues)})")

    # --- reports -----------------------------------------------------------
    # Merge into any existing report so a partial re-run updates just the papers
    # it processed and keeps the previously-recorded rows for the rest.
    fields = ["paper_id", "status", "issues", "figures", "sections", "title", "error"]
    report_csv = os.path.join(report_dir, "extraction_report.csv")
    merged = {}
    if os.path.isfile(report_csv):
        with open(report_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                merged[r["paper_id"]] = r
    for r in rows:
        merged[r["paper_id"]] = r
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(merged[k] for k in sorted(merged))

    failures_json = os.path.join(report_dir, "failures.json")
    with open(failures_json, "w", encoding="utf-8") as f:
        json.dump({"failed": failed, "incomplete": incomplete}, f,
                  indent=2, ensure_ascii=False)

    ok_count = sum(1 for r in rows if r["status"] == "ok")
    print(f"\nDone: {ok_count}/{len(pdf_paths)} fully ok, "
          f"{len(incomplete)} incomplete, {len(failed)} failed.")
    print(f"Report: {report_csv}")
    print(f"        {failures_json}")
    if failed:
        print(f"\n{len(failed)} hard failures:")
        for rec in failed:
            print(f"  {rec['paper_id']}: {rec['error']}")
    if incomplete:
        print(f"\n{len(incomplete)} incomplete extractions:")
        for rec in incomplete:
            print(f"  {rec['paper_id']}: {rec['issues']}")


if __name__ == "__main__":
    main()
