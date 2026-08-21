#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any

from compare_netlist_rule_levels import RULES, component_counts, graphs_match
from netlist_parser import extract_candidate_reason, normalize_candidate_text, parse_netlist_file, parse_netlist_text
from run_general_validation import V, _infer_ports_from_cir_path, _ISOURCE_SWEEP_BASES, _make_element_lines_fn
from verify_schematic_netlist import DEFAULT_OPENROUTER_MODEL, openrouter_chat, resolve_case_image


def parse_ids(values: list[str]) -> list[int]:
    ids: list[int] = []
    for value in values:
        if "-" in value:
            start_raw, end_raw = value.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if end < start:
                raise ValueError(f"invalid id range: {value}")
            ids.extend(range(start, end + 1))
        else:
            ids.append(int(value))
    return sorted(dict.fromkeys(ids))


def id_name(case_id: int) -> str:
    return f"{case_id:06d}"


def resolve_generated_netlist(generated_dir: Path, case_id: int) -> Path:
    for name in (id_name(case_id), str(case_id)):
        path = generated_dir / f"{name}.cir"
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"no generated netlist found for {case_id}")


def resolve_ground_truth(ground_truth_dir: Path | None, case_id: int) -> Path | None:
    if ground_truth_dir is None:
        return None
    for name in (id_name(case_id), str(case_id)):
        path = ground_truth_dir / f"{name}.cir"
        if path.exists():
            return path.resolve()
    return None


def image_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(image_path.read_bytes()).decode('ascii')}"


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def write_netlist(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def validation_passes_level_1_3(validation: dict[str, Any]) -> bool:
    simulation = validation["simulation"]
    op = simulation["op"]
    return bool(validation["syntax"]["ok"] and validation["connectivity"]["ok"] and simulation["ok"] and op["parse_ok"] and op["valid"])


def _filter_connectivity_issues(issues: list[str]) -> list[str]:
    """Remove false-positive floating-node warnings for V.../I... nets.

    Nets starting with V or I are external ports (supplies, inputs, outputs)
    that legitimately connect to only one component in the netlist.
    """
    filtered = []
    for issue in issues:
        if issue.startswith("Floating internal node '"):
            first_char = issue[len("Floating internal node '")]
            if first_char.upper() in ("V", "I"):
                continue
        filtered.append(issue)
    return filtered


def tc_result_to_dict(result: V.TCResult, *, cir_path: Path) -> dict[str, Any]:
    conn_issues = _filter_connectivity_issues(result.connectivity_issues)
    return {
        "inferred_ports": _infer_ports_from_cir_path(cir_path) if cir_path.exists() else [],
        "syntax": {
            "ok": result.syntax_ok,
            "errors": result.syntax_errors,
        },
        "connectivity": {
            "ok": not conn_issues,
            "issues": conn_issues,
        },
        "simulation": {
            "ok": result.sim_ok,
            "error": result.sim_error if not result.sim_ok else "",
            "op": {
                "parse_ok": result.sim_parse_ok,
                "valid": result.op_valid,
                "notes": result.op_notes,
                "voltages": {key: round(value, 6) for key, value in result.op_voltages.items()},
            },
            "dc": {
                "ok": result.dc_ok,
                "gain": round(result.dc_gain, 4),
                "inverting": result.dc_inverting,
                "swing_low": round(result.dc_swing_low, 4),
                "swing_high": round(result.dc_swing_high, 4),
                "error": result.dc_error,
            },
            "ac": {
                "ok": result.ac_ok,
                "midband_db": round(result.ac_midband_db, 3),
                "bw_hz": round(result.ac_bw_hz, 2),
                "phase_deg": round(result.ac_phase_deg, 2),
                "error": result.ac_error,
            },
            "tran": {
                "ok": result.tran_ok,
                "pp_norm": round(result.tran_pp_norm, 4),
                "error": result.tran_error,
            },
        },
        "passes_level_1_3": validation_passes_level_1_3(
            {
                "syntax": {"ok": result.syntax_ok},
                "connectivity": {"ok": not conn_issues},
                "simulation": {
                    "ok": result.sim_ok,
                    "op": {"parse_ok": result.sim_parse_ok, "valid": result.op_valid},
                },
            }
        ),
    }


def validate_netlist_file(case_id: int, cir_path: Path, work_dir: Path) -> dict[str, Any]:
    tc_id = id_name(case_id)
    tc_base = work_dir / "validation_cases"
    case_dir = tc_base / tc_id
    case_dir.mkdir(parents=True, exist_ok=True)
    staged_cir = case_dir / f"{tc_id}.cir"
    staged_cir.write_text(cir_path.read_text())
    best: V.TCResult | None = None
    for base_a in _ISOURCE_SWEEP_BASES:
        V._element_lines_ideal = _make_element_lines_fn(base_a)
        result = V.validate(tc_id, mode="ideal", tc_base=tc_base, sim_base=work_dir / "simulation")
        if best is None or (not best.op_valid and result.op_valid):
            best = result
        if result.op_valid:
            break
    V._element_lines_ideal = _make_element_lines_fn(_ISOURCE_SWEEP_BASES[0])
    return tc_result_to_dict(best, cir_path=staged_cir)


def evaluate_against_ground_truth(cir_path: Path, ground_truth_path: Path | None) -> dict[str, Any]:
    base = {
        "available": ground_truth_path is not None,
        "included_in_llm_context": False,
        "ground_truth_path": str(ground_truth_path) if ground_truth_path else None,
        "rules": None,
    }
    if ground_truth_path is None:
        return base
    generated = parse_netlist_file(cir_path)
    truth = parse_netlist_file(ground_truth_path)
    base.update(
        {
            "generated_component_count": len(generated),
            "ground_truth_component_count": len(truth),
            "generated_component_type_counts": component_counts(generated),
            "ground_truth_component_type_counts": component_counts(truth),
            "rules": {rule["name"]: graphs_match(generated, truth, rule) for rule in RULES},
        }
    )
    return base


def validation_summary_for_prompt(validation: dict[str, Any]) -> str:
    sim = validation["simulation"]
    op = sim["op"]
    lines = [
        f"Inferred ports: {', '.join(validation.get('inferred_ports') or []) or 'none'}",
        f"Level 1 syntax ok: {validation['syntax']['ok']}",
        f"Level 2 connectivity ok: {validation['connectivity']['ok']}",
        f"Level 3 simulation converged: {sim['ok']}",
        f"OP parsed: {op['parse_ok']}",
        f"OP physically valid: {op['valid']}",
    ]
    if validation["syntax"]["errors"]:
        lines.append("Syntax errors: " + "; ".join(validation["syntax"]["errors"][:6]))
    if validation["connectivity"]["issues"]:
        lines.append("Connectivity issues: " + "; ".join(validation["connectivity"]["issues"][:6]))
    if op["notes"]:
        lines.append("OP notes: " + "; ".join(op["notes"][:6]))
    dc = sim.get("dc") or {}
    if dc.get("ok"):
        direction = "inverting" if dc.get("inverting") else "non-inverting"
        swing = round((dc.get("swing_high") or 0) - (dc.get("swing_low") or 0), 4)
        lines.append(f"DC sweep: gain={dc.get('gain', 0):.3f} ({direction}), output swing={swing:.3f} V")
    elif dc.get("error"):
        lines.append(f"DC sweep: not available — {dc['error']}")
    ac = sim.get("ac") or {}
    if ac.get("ok"):
        lines.append(
            f"AC sweep: midband={ac.get('midband_db', 0):.1f} dB, "
            f"BW={ac.get('bw_hz', 0):.2e} Hz, phase={ac.get('phase_deg', 0):.1f} deg"
        )
    elif ac.get("error"):
        lines.append(f"AC sweep: not available — {ac['error']}")
    tran = sim.get("tran") or {}
    if tran.get("ok"):
        lines.append(f"Transient: pp_norm={tran.get('pp_norm', 0):.4f} (normalised to VDD)")
    elif tran.get("error"):
        lines.append(f"Transient: not available — {tran['error']}")
    return "\n".join(lines)


def prior_candidates_prompt(prior_candidates: list[dict[str, Any]]) -> str:
    if not prior_candidates:
        return ""
    blocks = ["\nPrior candidate attempts and validation context:"]
    for item in prior_candidates[-8:]:
        blocks.append(
            "\n".join(
                [
                    f"Candidate from iteration {item['iteration']} #{item['candidate_index']}:",
                    item["normalized_text"].strip(),
                    "Validation:",
                    validation_summary_for_prompt(item["validation"]),
                ]
            )
        )
    return "\n\n".join(blocks)


def repair_prompt(
    *,
    case_id: int,
    current_netlist_text: str,
    current_validation: dict[str, Any],
    prior_candidates: list[dict[str, Any]],
) -> str:
    return (
        "You are repairing a generated circuit netlist using the schematic image.\n"
        f"Dataset ID: {case_id}\n"
        "The schematic image is the source of truth. The ground-truth netlist is not provided.\n"
        "Use the current generated netlist as the reference netlist and starting point. Check it against the schematic image, then correct it.\n"
        "Return a short reason followed by the full revised netlist.\n"
        "Allowed formats:\n"
        "M*: <InstanceName> <drain> <gate> <source> <bulk> <pmos|nmos> [W] [L]\n"
        "Q*: <InstanceName> <collector> <base> <emitter> <pnp|npn> [value]\n"
        "R*/C*/L*/I*/V*: <InstanceName> <node1> <node2> [value]\n"
        "Prioritize device type, component count, connectivity, and pin order.\n"
        "For rule consistency, keep topology correct first; then preserve supply/ground names such as VDD, VSS, and GND; then preserve special/external port names such as VIN, VOUT, IN, OUT, VREF, VBIAS, IBIAS, and VCLK; then preserve visible component indexes.\n"
        "Inferred port names from validation may be inaccurate. Rename, add, or remove port-like names when the image and connectivity require it.\n"
        "Image/OCR text labels are optional evidence only. Do not force every visible text token into the netlist.\n"
        "Avoid floating internal nodes. If a text label would create a one-connection internal node, prefer the connected net name or omit that label unless it is a real external port.\n"
        "If a visible sizing or element value is present, append it at the end of that line after the required nets/type. Examples: R1 VDD VOUT 10k; C1 VOUT GND 1p; V1 VIN GND DC 0.9; I1 N1 GND 1u; M1 D G S B nmos 1u 180n.\n"
        "Do not include comments or unsupported extra columns in the netlist.\n"
        "Keep correct portions unchanged; change only what the image and validation evidence support.\n\n"
        "Output format:\n"
        "Reason: one or two short sentences describing the image/reference-netlist checks and the main corrections.\n"
        "Netlist:\n"
        "<revised netlist lines only>\n\n"
        "Current generated netlist:\n"
        f"{current_netlist_text.strip()}\n\n"
        "Current Level 1-3 validation context:\n"
        f"{validation_summary_for_prompt(current_validation)}"
        f"{prior_candidates_prompt(prior_candidates)}"
    )


def call_llm_candidates(
    *,
    case_id: int,
    image_path: Path,
    current_netlist_text: str,
    current_validation: dict[str, Any],
    prior_candidates: list[dict[str, Any]],
    model: str,
    candidate_count: int,
    timeout_s: int,
) -> list[dict[str, Any]]:
    prompt = repair_prompt(
        case_id=case_id,
        current_netlist_text=current_netlist_text,
        current_validation=current_validation,
        prior_candidates=prior_candidates,
    )
    candidates: list[dict[str, Any]] = []
    temperatures = [0.0, 0.5, 0.95, 0.1, 0.2, 0.35, 0.65, 0.8][:candidate_count]
    for index, temperature in enumerate(temperatures, 1):
        try:
            raw = openrouter_chat(
                api_key=os.environ.get("OPENROUTER_API_KEY", ""),
                model=model,
                temperature=temperature,
                timeout_s=timeout_s,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
                        ],
                    }
                ],
            )
            normalized = normalize_candidate_text(raw)
            parsed_error = None
            if normalized:
                try:
                    parse_netlist_text(normalized)
                except Exception as exc:
                    parsed_error = str(exc)
            candidates.append(
                {
                    "candidate_index": index,
                    "temperature": temperature,
                    "raw_text": raw,
                    "reason": extract_candidate_reason(raw),
                    "normalized_text": normalized,
                    "normalized_line_count": len([line for line in normalized.splitlines() if line.strip()]),
                    "parse_error": parsed_error,
                }
            )
        except Exception as exc:
            candidates.append(
                {
                    "candidate_index": index,
                    "temperature": temperature,
                    "error": str(exc),
                    "normalized_text": "",
                    "normalized_line_count": 0,
                }
            )
    return candidates


def run_case(
    *,
    case_id: int,
    generated_dir: Path,
    ground_truth_dir: Path | None,
    dataset_root: Path,
    out_dir: Path,
    model: str,
    candidate_count: int,
    max_iterations: int,
    timeout_s: int,
) -> dict[str, Any]:
    case_dir = out_dir / id_name(case_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    image_path, image_source = resolve_case_image(dataset_root, case_id)
    original_path = resolve_generated_netlist(generated_dir, case_id)
    ground_truth_path = resolve_ground_truth(ground_truth_dir, case_id)

    current_text = original_path.read_text()
    current_source = "original"
    prior_candidate_contexts: list[dict[str, Any]] = []
    iterations: list[dict[str, Any]] = []
    final_result: dict[str, Any] | None = None

    for iteration in range(1, max_iterations + 1):
        iter_dir = case_dir / f"iter_{iteration:02d}"
        current_path = iter_dir / f"current_{current_source}.cir"
        write_netlist(current_path, current_text)
        current_validation = validate_netlist_file(case_id, current_path, iter_dir / "current_validation")
        current_gt_eval = evaluate_against_ground_truth(current_path, ground_truth_path)

        llm_candidates = call_llm_candidates(
            case_id=case_id,
            image_path=image_path,
            current_netlist_text=current_text,
            current_validation=current_validation,
            prior_candidates=prior_candidate_contexts,
            model=model,
            candidate_count=candidate_count,
            timeout_s=timeout_s,
        )

        candidate_records: list[dict[str, Any]] = []
        valid_solution_records: list[dict[str, Any]] = []
        for candidate in llm_candidates:
            record = dict(candidate)
            normalized = candidate.get("normalized_text") or ""
            if normalized:
                candidate_path = iter_dir / f"candidate_{candidate['candidate_index']:02d}.cir"
                write_netlist(candidate_path, normalized)
                validation = validate_netlist_file(case_id, candidate_path, iter_dir / f"candidate_{candidate['candidate_index']:02d}_validation")
                gt_eval = evaluate_against_ground_truth(candidate_path, ground_truth_path)
                record.update(
                    {
                        "path": str(candidate_path),
                        "validation": validation,
                        "ground_truth_evaluation": gt_eval,
                    }
                )
                prior_candidate_contexts.append(
                    {
                        "iteration": iteration,
                        "candidate_index": candidate["candidate_index"],
                        "normalized_text": normalized,
                        "validation": validation,
                    }
                )
                if validation["passes_level_1_3"]:
                    valid_solution_records.append(record)
            candidate_records.append(record)

        iteration_record = {
            "iteration": iteration,
            "current_source": current_source,
            "current_path": str(current_path),
            "current_validation": current_validation,
            "current_ground_truth_evaluation": current_gt_eval,
            "llm_context": {
                "validation_included": True,
                "current_netlist_included": True,
                "prior_candidates_included": bool(prior_candidate_contexts),
                "ground_truth_included": False,
            },
            "candidates": candidate_records,
        }
        iterations.append(iteration_record)

        if valid_solution_records:
            best = valid_solution_records[0]
            final_result = {
                "status": "final_solution_level_1_3_passed",
                "iteration": iteration,
                "candidate_index": best["candidate_index"],
                "netlist_path": best["path"],
                "validation": best["validation"],
                "ground_truth_evaluation": best["ground_truth_evaluation"],
            }
            break

        parsed_candidates = [record for record in candidate_records if record.get("normalized_text")]
        if parsed_candidates:
            selected = parsed_candidates[0]
            current_text = selected["normalized_text"]
            current_source = f"candidate_{iteration}_{selected['candidate_index']}"
        else:
            current_source = f"no_solution_iter_{iteration}"

    if final_result is None:
        final_path = case_dir / "final_unresolved.cir"
        write_netlist(final_path, current_text)
        final_validation = validate_netlist_file(case_id, final_path, case_dir / "final_validation")
        final_result = {
            "status": "unresolved_after_max_iterations",
            "iteration": max_iterations,
            "netlist_path": str(final_path),
            "validation": final_validation,
            "ground_truth_evaluation": evaluate_against_ground_truth(final_path, ground_truth_path),
        }

    result = {
        "id": case_id,
        "original_path": str(original_path),
        "image_path": str(image_path),
        "image_source": image_source,
        "ground_truth": {
            "available": ground_truth_path is not None,
            "path": str(ground_truth_path) if ground_truth_path else None,
            "included_in_llm_context": False,
        },
        "iterations": iterations,
        "final": final_result,
    }
    write_json(case_dir / "result.json", result)
    return result


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "case_count": len(results),
        "final_status_counts": {},
        "final_level_1_3_pass_count": 0,
        "ground_truth_available_count": 0,
        "final_rule_pass_counts": {rule["name"]: 0 for rule in RULES},
    }
    for result in results:
        status = result["final"]["status"]
        summary["final_status_counts"][status] = summary["final_status_counts"].get(status, 0) + 1
        if result["final"]["validation"]["passes_level_1_3"]:
            summary["final_level_1_3_pass_count"] += 1
        gt_eval = result["final"]["ground_truth_evaluation"]
        if gt_eval.get("available"):
            summary["ground_truth_available_count"] += 1
            for rule_name, passed in (gt_eval.get("rules") or {}).items():
                if passed:
                    summary["final_rule_pass_counts"][rule_name] += 1
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Iteratively repair generated netlists with LLM + Level 1-3 validation context.")
    parser.add_argument("ids", nargs="+", help="IDs or ranges, e.g. 1 18 299")
    parser.add_argument("--generated-dir", type=Path, default=Path("Data/netlist_generated"))
    parser.add_argument("--ground-truth-dir", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=Path("Data"))
    parser.add_argument("--out-dir", type=Path, default=Path("simulation_general/iterative_llm_repair"))
    parser.add_argument("--model", default=DEFAULT_OPENROUTER_MODEL)
    parser.add_argument("--candidate-count", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--timeout-s", type=int, default=45)
    args = parser.parse_args()

    ids = parse_ids(args.ids)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_dir = args.ground_truth_dir.resolve() if args.ground_truth_dir else None

    results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for case_id in ids:
        try:
            result = run_case(
                case_id=case_id,
                generated_dir=args.generated_dir.resolve(),
                ground_truth_dir=ground_truth_dir,
                dataset_root=args.dataset_root.resolve(),
                out_dir=out_dir,
                model=args.model,
                candidate_count=args.candidate_count,
                max_iterations=args.max_iterations,
                timeout_s=args.timeout_s,
            )
        except FileNotFoundError as exc:
            print(f"[skip] case {case_id}: {exc}", file=sys.stderr)
            skipped.append({"id": case_id, "reason": str(exc)})
            continue
        results.append(result)

    payload = {
        "config": {
            "ids": ids,
            "generated_dir": str(args.generated_dir.resolve()),
            "ground_truth_dir": str(ground_truth_dir) if ground_truth_dir else None,
            "ground_truth_included_in_llm_context": False,
            "dataset_root": str(args.dataset_root.resolve()),
            "out_dir": str(out_dir),
            "model": args.model,
            "candidate_count": args.candidate_count,
            "max_iterations": args.max_iterations,
            "timeout_s": args.timeout_s,
        },
        "summary": summarize_results(results),
        "skipped": skipped,
        "cases": results,
    }
    write_json(out_dir / "summary.json", payload)

    print(json.dumps(payload["summary"], indent=2))
    if skipped:
        print(f"Skipped {len(skipped)} case(s) with missing netlist/image: {[s['id'] for s in skipped]}")
    print(f"JSON -> {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
