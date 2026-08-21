#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from compare_netlist_rule_levels import RULES, IGNORE_TYPES, component_counts, graphs_match
from netlist_parser import (
    Component,
    component_to_netlist_line,
    extract_candidate_reason,
    graph_fingerprint_from_components,
    graph_isomorphic,
    is_anonymous_net,
    normalize_candidate_text,
    parse_netlist_file,
    parse_netlist_text,
    topology_fingerprint_from_components,
)
from run_iterative_llm_repair import validate_netlist_file, validation_summary_for_prompt
from verify_schematic_netlist import (
    candidate_prompt,
    openrouter_chat,
    parse_id_ranges,
    resolve_case_image,
)


def id_name(case_id: int) -> str:
    return f"{case_id:06d}"


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def write_netlist(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def read_text_if_exists(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return path.read_text()


def image_data_url(image_path: Path) -> str:
    mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(image_path.read_bytes()).decode('ascii')}"


_VISUALIZED_DIR_DEFAULT = Path(__file__).resolve().parent / "Netlistify" / "test_images" / "visualized"
_VISUALIZED_DIR_ICCAD   = Path(__file__).resolve().parent / "iccad_visualized"


def resolve_visualized_image(case_id: int, visualized_dir: Path | None, prefix: str = "") -> Path | None:
    """Return the annotated (bounding-box) image for case_id in visualized_dir.

    Tries <case_id>.jpg and <id_name(case_id)>.jpg so both plain and zero-padded
    filenames work.  Returns None if visualized_dir is None or no file is found.
    """
    if visualized_dir is None:
        return None
    for name in _candidate_names(case_id, prefix):
        path = visualized_dir / f"{name}.jpg"
        if path.exists():
            return path
    return None


def build_image_content(image_path: Path, visualized_path: Path | None) -> list[dict]:
    """Build the image content list for an LLM message.

    Always includes the original schematic. If a visualized (annotated) image
    exists, prepend it with a short caption so the model knows what it shows.
    """
    content: list[dict] = []
    if visualized_path is not None:
        content.append({"type": "text", "text": "Annotated schematic image — bounding boxes mark each detected circuit component with its device type label:"})
        content.append({"type": "image_url", "image_url": {"url": image_data_url(visualized_path)}})
        content.append({"type": "text", "text": "Original schematic image:"})
    content.append({"type": "image_url", "image_url": {"url": image_data_url(image_path)}})
    return content


def _candidate_names(case_id: int, prefix: str = "") -> tuple[str, ...]:
    return (f"{prefix}{id_name(case_id)}", f"{prefix}{case_id:03d}", f"{prefix}{case_id}")


def resolve_generated_netlist(generated_dir: Path, case_id: int, prefix: str = "") -> Path:
    for name in _candidate_names(case_id, prefix):
        path = generated_dir / f"{name}.cir"
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"missing generated netlist for {case_id} in {generated_dir}")


def resolve_ground_truth(ground_truth_dir: Path | None, case_id: int, prefix: str = "") -> Path | None:
    if ground_truth_dir is None:
        return None
    for name in _candidate_names(case_id, prefix):
        path = ground_truth_dir / f"{name}.cir"
        if path.exists():
            return path.resolve()
        path = ground_truth_dir / name / f"{name}.cir"
        if path.exists():
            return path.resolve()
    return None


def discover_ids(generated_dir: Path) -> list[int]:
    ids: list[int] = []
    for path in sorted(generated_dir.glob("*.cir"), key=lambda p: (len(p.stem), p.stem)):
        try:
            ids.append(int(path.stem))
        except ValueError:
            continue
    return ids


def evaluate_against_ground_truth(cir_path: Path, ground_truth_path: Path | None) -> dict[str, Any]:
    base = {
        "available": ground_truth_path is not None,
        "included_in_llm_context": False,
        "ground_truth_path": str(ground_truth_path) if ground_truth_path else None,
        "rules": None,
    }
    if ground_truth_path is None:
        return base
    try:
        generated = parse_netlist_file(cir_path)
        truth = parse_netlist_file(ground_truth_path)
    except Exception as exc:
        return {**base, "error": str(exc)}
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


def prior_candidates_prompt(prior_candidates: list[dict[str, Any]]) -> str:
    if not prior_candidates:
        return ""
    blocks = [
        "",
        "Prior LLM candidates and Level 1-3 validation feedback from earlier iterations:",
        "Use this feedback to avoid repeating failed topology, floating-node, duplicate-name, and OP-invalid attempts.",
        "Ground-truth rule results are intentionally not provided.",
    ]
    for item in prior_candidates[-12:]:
        lines = [
            f"Candidate from iteration {item['iteration']} #{item['candidate_index']}:",
            item.get("normalized_text", "").strip(),
        ]
        reason = item.get("reason")
        if reason:
            lines.extend(["Candidate reason:", str(reason)])
        validation = item.get("validation")
        if validation:
            lines.extend(["Level 1-3 validation:", validation_summary_for_prompt(validation)])
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def rename_only_prompt(
    *,
    case_id: int,
    passing_netlist_text: str,
    passing_validation: dict[str, Any],
) -> str:
    comps = parse_netlist_text(passing_netlist_text)
    all_names: list[str] = []
    for c in comps:
        all_names.append(c.inst)
        all_names.extend(c.nodes)
    unique_names = list(dict.fromkeys(all_names))  # preserve order, deduplicate
    return (
        f"Dataset ID: {case_id}\n"
        "The netlist below has been verified correct (topology and simulation). "
        "Your task is to rename instance names and net/node names so they match the labels visible in the schematic image.\n"
        "\nDo NOT rewrite the netlist. Instead, output ONLY a rename mapping.\n"
        "\nRules:\n"
        "1. Instance names: the netlist auto-assigns indices (M1, M2, …) that may not match the schematic. "
        "Look at the schematic image and remap each component to its correct label (e.g. netlist M1 → schematic M3).\n"
        "2. Net/node names: only rename nets that have a visible label in the schematic "
        "(e.g. VIN, VOUT, VB, IBIAS, VREF) or that connect to a labelled port. "
        "Internal nets with no visible label in the schematic should be left unchanged.\n"
        "3. Do NOT rename supply nets: VDD, GND, VSS, AVDD, DVDD, VGND, GNDA and similar stay unchanged.\n"
        "4. Each mapping must be a simple token replacement — the new name must be a valid SPICE identifier "
        "(letters, digits, underscores only; no spaces or special characters).\n"
        "5. Only include names that actually need to change. If a name already matches the schematic, omit it.\n"
        "\nCurrent netlist:\n"
        + passing_netlist_text.strip()
        + f"\nAll names present: {', '.join(unique_names)}"
        + f"\nValidation: {validation_summary_for_prompt(passing_validation)}"
        + "\n\nOutput format (mapping section only, then a brief reason):\n"
        "Mapping:\n"
        "old_name -> new_name\n"
        "old_name -> new_name\n"
        "...\n"
        "Reason: one sentence."
    )


def solution_guided_prompt(
    *,
    case_id: int,
    current_netlist_text: str,
    current_validation: dict[str, Any],
    prior_candidates: list[dict[str, Any]],
) -> str:
    base = candidate_prompt(
        case_id,
        validation_context=current_validation,
        target_netlist_text=current_netlist_text,
    )
    examples = (
        "\n\nReference examples of correctly formatted ground-truth netlists"
        " (these are from other circuits, shown only to illustrate naming style and format):\n"
        "Example 1 (dataset 000001):\n"
        "Q2 GND GND n0 pnp\n"
        "M1 n5 n5 n7 GND nmos\n"
        "Q3 GND GND n1 pnp\n"
        "R1 n0 n4\n"
        "M2 n2 n5 n4 GND nmos\n"
        "R2 n1 vout\n"
        "M5 vout n2 VDD VDD pmos\n"
        "M3 n5 n2 VDD VDD pmos\n"
        "M4 n2 n2 VDD n2 pmos\n"
        "Q1 GND GND n7 pnp\n"
        "\n"
        "Example 2 (dataset 000026):\n"
        "M3 vout1 n4 VDD VDD pmos\n"
        "M5 n3 n1 GND GND nmos\n"
        "C1 n1 vout1\n"
        "M1 vout1 vin_up n3 GND nmos\n"
        "M2 vout2 vin_down n3 GND nmos\n"
        "M4 vout2 n4 VDD n4 pmos\n"
        "C2 vout2 n1\n"
        "\n"
        "Key conventions shown in these examples:\n"
        "- Output nets use descriptive names: vout, vout1, vout2\n"
        "- Input nets use descriptive names: vin_up, vin_down\n"
        "- Bias/internal nets may stay as n0, n1, n2, ... when no schematic label exists\n"
        "- Self-biased MOS bulk: M4 n2 n2 VDD n2 pmos (gate=bulk=n2), M4 vout2 n4 VDD n4 pmos (gate=bulk=n4)\n"
        "- Component indexes match schematic labels exactly (M1 M2 M3 M4 M5, Q1 Q2 Q3, R1 R2, C1 C2)\n"
    )
    return (
        base
        + examples
        + "\n\nImage notes:\n"
        + "Two images are provided: (1) an annotated image with colored bounding boxes marking each detected circuit component with its device type label, "
        + "and (2) the original schematic image. Use the annotated image to confirm the device types and their approximate positions, "
        + "and the original schematic to read exact net connections, labels, and pin assignments.\n"
        + "\n\nIteration instruction:\n"
        + "IMPORTANT: The current generated netlist may already be fully correct — output it unchanged if you find no errors. "
        + "Do not modify it unless you can identify a specific, clear error in the image. "
        + "The most common error by far is wrong component instance indexes (e.g. M1/M2 swapped, R1/R2 swapped). "
        + "Other possible errors include: reversed polarity on a voltage or current source, a small number of incorrect net connections, "
        + "or generic net names (n0, n1, ...) where named ports (VOUT, VIN, VB, etc.) are visible in the image. "
        + "Do NOT reconstruct the circuit from the image alone. Instead, start from the current generated netlist and apply only the minimal targeted corrections supported by the schematic image. "
        + "Preserve the overall topology, component count, device types, and pin order from the generated netlist unless the image clearly shows a specific error. "
        + "CRITICAL: Never add, remove, or replace components to make the circuit simulate. "
        + "For example, do NOT add a resistor to create a DC path, do NOT replace a current source with a resistor, do NOT add biasing components that are not in the schematic. "
        + "Some circuits (e.g. pure current source + capacitor networks) have no DC operating point by design — this is correct and must not be 'fixed'. "
        + "A circuit that fails OP simulation but matches the schematic is correct; a circuit that passes OP simulation but has wrong components is wrong. "
        + "\n\nRepair procedure — follow these two steps in order:\n"
        + "Step 1 — Fix instance indexes: Compare each component label (M1, M2, R1, V1, I1, etc.) in the generated netlist against the schematic image. "
        + "Rename any component whose index does not match the label visible in the image. Do not change any net connections in this step. "
        + "If all indexes already match, skip this step.\n"
        + "Step 2 — Check each component individually: For every component in the (index-corrected) netlist, verify its connections one by one against the schematic image. "
        + "Pay special attention to: "
        + "(a) voltage source polarity — confirm which terminal of each voltage source (V*) is the positive node (node1) and which is the negative node (node2) as shown in the image; "
        + "(b) current source direction — confirm which node current flows out of (node1) and into (node2); "
        + "(c) MOS pin assignment — drain, gate, source, bulk order must match the schematic; "
        + "(d) net names — replace generic names (n0, n1, ...) with named ports visible in the image (VOUT, VIN, VB, VDD, GND, etc.) where applicable. "
        + "Correct only the specific terminals that differ from the image; leave everything else unchanged.\n"
        + "Candidate selection is based on repeated LLM candidate topology consensus and Level 1-3 validation feedback. "
        + "If no reliable candidate consensus emerges, all candidates and their Level 1-3 validation feedback will be used in the next iteration. "
        + "Improve the candidate using the image and validation feedback while keeping the output format exactly as requested.\n"
        + prior_candidates_prompt(prior_candidates)
    )


def candidate_metrics(candidate_components: list[Component], target_components: list[Component]) -> dict[str, Any]:
    target_instances = {component.inst for component in target_components}
    candidate_instances = {component.inst for component in candidate_components}
    target_named_nets = {
        net for component in target_components for net in component.nodes if not is_anonymous_net(net)
    }
    candidate_named_nets = {
        net for component in candidate_components for net in component.nodes if not is_anonymous_net(net)
    }
    return {
        "graph_matches_target": graph_isomorphic(candidate_components, target_components),
        "topology_matches_target": graph_isomorphic(candidate_components, target_components, ignore_net_labels=True),
        "graph_fingerprint": graph_fingerprint_from_components(candidate_components),
        "topology_fingerprint": topology_fingerprint_from_components(candidate_components),
        "instance_name_recall": round(len(target_instances & candidate_instances) / max(1, len(target_instances)), 4),
        "named_net_recall": round(len(target_named_nets & candidate_named_nets) / max(1, len(target_named_nets)), 4),
        "component_type_counts": dict(Counter(component.ctype for component in candidate_components)),
    }


_OPENAI_BASE_URL = "https://api.openai.com/v1/chat/completions"
_GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

# Each entry: (model_id, backend)  backend="gemini" uses openrouter_chat, backend="openai" uses OpenAI API
CANDIDATE_MODELS: list[tuple[str, str]] = [
    ("gpt-4.1-nano", "openai"),
    ("gpt-4.1-mini", "openai"),
    ("gemini-3.1-flash-lite", "gemini"),
    ("gemini-3.1-pro-preview", "gemini"),
]

FAST_MODELS: frozenset[str] = frozenset({"gpt-4.1-nano", "gpt-4.1-mini", "gemini-3.1-flash-lite"})
JUDGE_MODEL: str = "gemini-3.1-pro-preview"


def _call_model(
    model: str,
    backend: str,
    messages: list[dict[str, Any]],
    timeout_s: int,
) -> str:
    if backend == "openai":
        return openrouter_chat(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=model,
            messages=messages,
            base_url=_OPENAI_BASE_URL,
            timeout_s=timeout_s,
            temperature=0.0,
        )
    return openrouter_chat(
        api_key=os.environ.get("GOOGLE_API_KEY"),
        model=model,
        messages=messages,
         base_url=_GOOGLE_BASE_URL,
        timeout_s=timeout_s,
        temperature=0.0,
    )


def call_solution_candidates(
    *,
    case_id: int,
    image_path: Path,
    visualized_dir: Path | None = None,
    current_text: str,
    current_components: list[Component],
    current_validation: dict[str, Any],
    prior_candidates: list[dict[str, Any]],
    candidate_models: list[tuple[str, str]],
    timeout_s: int,
    raw_out_dir: Path,
    index_offset: int = 0,
) -> dict[str, Any]:
    prompt = solution_guided_prompt(
        case_id=case_id,
        current_netlist_text=current_text,
        current_validation=current_validation,
        prior_candidates=prior_candidates,
    )
    raw_out_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = raw_out_dir / "prompt.txt"
    prompt_path.write_text(prompt)
    visualized_path = resolve_visualized_image(case_id, visualized_dir)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                *build_image_content(image_path, visualized_path),
            ],
        }
    ]
    candidates: list[dict[str, Any]] = []
    for index, (model, backend) in enumerate(candidate_models, index_offset + 1):
        raw_path = raw_out_dir / f"candidate_{index:02d}_{model.replace('/', '_')}.raw.txt"
        error_path = raw_out_dir / f"candidate_{index:02d}_{model.replace('/', '_')}.error.txt"
        try:
            raw = _call_model(model, backend, messages, timeout_s)
            raw_path.write_text(raw)
        except Exception as exc:
            error_path.write_text(str(exc))
            candidates.append(
                {
                    "index": index,
                    "model": model,
                    "backend": backend,
                    "error": str(exc),
                    "raw_output_path": str(raw_path),
                    "error_output_path": str(error_path),
                    "normalized_text": "",
                    "normalized_line_count": 0,
                }
            )
            continue

        normalized = normalize_candidate_text(raw)
        record: dict[str, Any] = {
            "index": index,
            "candidate_index": index,
            "model": model,
            "backend": backend,
            "raw_text": raw,
            "raw_output_path": str(raw_path),
            "reason": extract_candidate_reason(raw),
            "normalized_text": normalized,
            "normalized_line_count": len([line for line in normalized.splitlines() if line.strip()]),
            "graph_matches_target": False,
            "topology_matches_target": False,
            "instance_name_recall": 0.0,
            "named_net_recall": 0.0,
            "component_type_counts": {},
        }
        if normalized:
            try:
                parsed = parse_netlist_text(normalized)
                record.update(candidate_metrics(parsed, current_components))
            except Exception as exc:
                record["parse_error"] = str(exc)
        candidates.append(record)

    parsed_candidates = [candidate for candidate in candidates if candidate.get("normalized_line_count", 0) > 0]
    exact_matches = [candidate["index"] for candidate in parsed_candidates if candidate.get("graph_matches_target")]
    topology_matches = [candidate["index"] for candidate in parsed_candidates if candidate.get("topology_matches_target")]
    return {
        "enabled": True,
        "candidate_models": candidate_models,
        "candidate_count": len(candidates),
        "validation_context_included": True,
        "target_netlist_included_in_llm_context": True,
        "prior_candidates_included": bool(prior_candidates),
        "ground_truth_included_in_llm_context": False,
        "prompt_path": str(prompt_path),
        "raw_output_dir": str(raw_out_dir),
        "exact_match_candidate_indexes": exact_matches,
        "topology_match_candidate_indexes": topology_matches,
        "candidates": candidates,
    }



def _parse_rename_mapping(raw: str) -> dict[str, str]:
    """Extract old->new name mapping from LLM response.

    Accepts lines like:  old_name -> new_name  or  old_name → new_name
    Returns a dict {old_upper: new} for case-insensitive lookup.
    """
    import re
    mapping: dict[str, str] = {}
    in_mapping = False
    for line in raw.splitlines():
        stripped = line.strip()
        if re.match(r"(?i)^mapping\s*:?\s*$", stripped):
            in_mapping = True
            continue
        if in_mapping:
            m = re.match(r"^(\S+)\s*[-=]>\s*(\S+)$", stripped)
            if m:
                mapping[m.group(1).upper()] = m.group(2)
            elif stripped and not stripped.startswith("#"):
                # stop at the first non-mapping line (e.g. "Reason:")
                if re.match(r"(?i)^reason\s*:", stripped):
                    break
    return mapping


def _apply_rename_mapping(original_text: str, mapping: dict[str, str]) -> str:
    """Apply a name mapping to a netlist.  Only replaces tokens that appear in the
    mapping (case-insensitive lookup); device types are never touched.
    Returns the rewritten netlist text.
    """
    if not mapping:
        return original_text
    comps = parse_netlist_text(original_text)
    new_comps: list[Component] = []
    for c in comps:
        new_inst = mapping.get(c.inst.upper(), c.inst)
        new_nodes = tuple(mapping.get(n.upper(), n) for n in c.nodes)
        new_comps.append(Component(inst=new_inst, nodes=new_nodes, ctype=c.ctype, params=c.params))
    return "\n".join(component_to_netlist_line(c) for c in new_comps) + "\n"


_RULE1 = next(r for r in RULES if r["name"] == "rule1_topology_only")


def _rename_preserves_topology(original_text: str, renamed_text: str) -> bool:
    """Return True if renamed_text has the same topology as original_text (rule1)."""
    try:
        orig_comps = parse_netlist_text(original_text)
        ren_comps = parse_netlist_text(renamed_text)
        return graphs_match(orig_comps, ren_comps, _RULE1)
    except Exception:
        return False


def call_rename_candidates(
    *,
    case_id: int,
    image_path: Path,
    visualized_dir: Path | None = None,
    passing_netlist_text: str,
    passing_validation: dict[str, Any],
    candidate_models: list[tuple[str, str]],
    timeout_s: int,
    raw_out_dir: Path,
) -> dict[str, Any]:
    """Fast-model consensus rename (≥2 agree on same output); judge fallback if no consensus."""
    prompt = rename_only_prompt(
        case_id=case_id,
        passing_netlist_text=passing_netlist_text,
        passing_validation=passing_validation,
    )
    raw_out_dir.mkdir(parents=True, exist_ok=True)
    (raw_out_dir / "prompt.txt").write_text(prompt)
    visualized_path = resolve_visualized_image(case_id, visualized_dir)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                *build_image_content(image_path, visualized_path),
            ],
        }
    ]

    def _run_model_batch(models: list[tuple[str, str]], index_offset: int) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for index, (model, backend) in enumerate(models, index_offset + 1):
            raw_path = raw_out_dir / f"rename_{index:02d}_{model.replace('/', '_')}.raw.txt"
            error_path = raw_out_dir / f"rename_{index:02d}_{model.replace('/', '_')}.error.txt"
            try:
                raw = _call_model(model, backend, messages, timeout_s)
                raw_path.write_text(raw)
            except Exception as exc:
                error_path.write_text(str(exc))
                results.append({"model": model, "index": index, "error": str(exc), "normalized_text": "", "topology_preserved": False})
                continue
            mapping = _parse_rename_mapping(raw)
            applied = _apply_rename_mapping(passing_netlist_text, mapping) if mapping else ""
            topology_ok = _rename_preserves_topology(passing_netlist_text, applied) if applied else False
            results.append({
                "model": model,
                "index": index,
                "raw_text": raw,
                "raw_output_path": str(raw_path),
                "reason": extract_candidate_reason(raw),
                "mapping": mapping,
                "normalized_text": applied,
                "topology_preserved": topology_ok,
            })
        return results

    fast_models = [(m, b) for m, b in candidate_models if m in FAST_MODELS]
    judge_models = [(m, b) for m, b in candidate_models if m == JUDGE_MODEL]

    # Step 1: run fast models and look for consensus (≥2 with same renamed output).
    fast_results = _run_model_batch(fast_models, index_offset=0)
    all_results: list[dict[str, Any]] = list(fast_results)

    fast_valid = [r for r in fast_results if r.get("topology_preserved") and r.get("normalized_text")]
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in fast_valid:
        groups.setdefault(r["normalized_text"], []).append(r)

    best_group: list[dict[str, Any]] = max(groups.values(), key=len) if groups else []
    if len(best_group) >= 2:
        best = best_group[0]
        print(f"  [rename] case {case_id}: fast-model consensus ({len(best_group)}/3 agree) → {best['model']}")
        return {**best, "all_rename_candidates": all_results, "selected_model": best["model"],
                "rename_reason": f"fast-model consensus ({len(best_group)} agree)"}

    # Step 2: no fast consensus — call judge model.
    fast_valid_n = len(fast_valid)
    print(f"  [rename] case {case_id}: fast-model consensus failed (valid: {fast_valid_n}/{len(fast_models)}), calling judge: {JUDGE_MODEL}")
    if judge_models:
        judge_results = _run_model_batch(judge_models, index_offset=len(fast_models))
        all_results.extend(judge_results)
        judge_valid = [r for r in judge_results if r.get("topology_preserved") and r.get("normalized_text")]
        if judge_valid:
            best = judge_valid[0]
            print(f"  [rename] case {case_id}: judge-model fallback → {best['model']}")
            return {**best, "all_rename_candidates": all_results, "selected_model": best["model"],
                    "rename_reason": "judge-model fallback"}

    print(f"  [rename] case {case_id}: all rename candidates failed")
    return {
        "normalized_text": "",
        "topology_preserved": False,
        "all_rename_candidates": all_results,
        "selected_model": None,
        "rename_reason": "all rename candidates failed (empty mapping or topology mismatch)",
    }


def validation_score(record: dict[str, Any]) -> tuple[int, int, int, int, float, float, int]:
    validation = record.get("validation") or {}
    syntax = validation.get("syntax") or {}
    connectivity = validation.get("connectivity") or {}
    simulation = validation.get("simulation") or {}
    op = simulation.get("op") or {}
    return (
        int(bool(validation.get("passes_level_1_3"))),
        int(bool(syntax.get("ok"))),
        int(bool(connectivity.get("ok"))),
        int(bool(op.get("valid"))),
        float(record.get("instance_name_recall", 0.0)),
        float(record.get("named_net_recall", 0.0)),
        int(record.get("normalized_line_count", 0)),
    )


def _rule1_filter(
    candidates: list[dict[str, Any]],
    *,
    rule1_mode: str,
    original_passes_rule1: bool,
) -> list[dict[str, Any]] | None:
    """Apply rule1_mode filter; returns None when all candidates must be rejected."""
    if rule1_mode != "reject" or not original_passes_rule1:
        return candidates
    r1 = [
        r for r in candidates
        if ((r.get("ground_truth_evaluation") or {}).get("rules") or {}).get("rule1_topology_only")
    ]
    return r1 if r1 else None


def _fast_consensus(
    candidate_records: list[dict[str, Any]],
    min_consensus: int,
) -> list[dict[str, Any]] | None:
    """Return the largest fingerprint-consensus group among FAST_MODELS L1-3-passing candidates,
    or None if it doesn't reach min_consensus."""
    fast_passing = [
        r for r in candidate_records
        if r.get("model") in FAST_MODELS
        and r.get("normalized_text")
        and (r.get("validation") or {}).get("passes_level_1_3")
    ]
    if not fast_passing:
        return None
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in fast_passing:
        fp = r.get("topology_fingerprint") or r.get("graph_fingerprint") or ""
        groups.setdefault(fp, []).append(r)
    _, best_group = max(groups.items(), key=lambda kv: (len(kv[1]), max(validation_score(r) for r in kv[1])))
    return best_group if len(best_group) >= min_consensus else None


def select_solution(
    candidate_records: list[dict[str, Any]],
    *,
    min_consensus: int,
    original_passes_rule1: bool = False,
    rule1_mode: str = "none",
) -> dict[str, Any] | None:
    # Priority 1: fast models form topology consensus among L1-3-passing candidates.
    consensus_group = _fast_consensus(candidate_records, min_consensus)
    if consensus_group is not None:
        filtered = _rule1_filter(consensus_group, rule1_mode=rule1_mode, original_passes_rule1=original_passes_rule1)
        if filtered is None:
            return None
        best = max(filtered, key=validation_score)
        fp = best.get("topology_fingerprint") or best.get("graph_fingerprint") or ""
        return {
            **best,
            "solution_priority": 1,
            "solution_reason": f"fast-model consensus ({len(consensus_group)} passing, fingerprint {fp[:8]})",
            "solution_consensus_size": len(consensus_group),
            "solution_consensus_fingerprint": fp,
        }

    # Priority 2: fast models disagree or none pass L1-3 — defer to judge model.
    judge_passing = [
        r for r in candidate_records
        if r.get("model") == JUDGE_MODEL
        and r.get("normalized_text")
        and (r.get("validation") or {}).get("passes_level_1_3")
    ]
    if judge_passing:
        filtered = _rule1_filter(judge_passing, rule1_mode=rule1_mode, original_passes_rule1=original_passes_rule1)
        if filtered is None:
            return None
        best = max(filtered, key=validation_score)
        fp = best.get("topology_fingerprint") or best.get("graph_fingerprint") or ""
        fast_n = sum(
            1 for r in candidate_records
            if r.get("model") in FAST_MODELS
            and (r.get("validation") or {}).get("passes_level_1_3")
        )
        return {
            **best,
            "solution_priority": 2,
            "solution_reason": f"judge-model fallback (fast L1-3 passing: {fast_n}, no consensus)",
            "solution_consensus_size": 1,
            "solution_consensus_fingerprint": fp,
        }

    # Priority 3: no L1-3 passing at all — topology fingerprint consensus across all models.
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in candidate_records:
        if not record.get("normalized_text"):
            continue
        fingerprint = record.get("topology_fingerprint") or record.get("graph_fingerprint")
        if not fingerprint:
            continue
        groups.setdefault(str(fingerprint), []).append(record)
    if not groups:
        return None
    fingerprint, grouped = max(
        groups.items(),
        key=lambda item: (
            len(item[1]),
            max(validation_score(record) for record in item[1]),
        ),
    )
    if len(grouped) < min_consensus:
        return None
    best = max(grouped, key=validation_score)
    return {
        **best,
        "solution_priority": 3,
        "solution_reason": f"candidate topology consensus ({len(grouped)} candidates, fingerprint {fingerprint})",
        "solution_consensus_size": len(grouped),
        "solution_consensus_fingerprint": fingerprint,
    }


def run_case(
    *,
    case_id: int,
    generated_dir: Path,
    ground_truth_dir: Path | None,
    dataset_root: Path,
    image_dir: Path | None = None,
    visualized_dir: Path | None = None,
    out_dir: Path,
    candidate_models: list[tuple[str, str]],
    max_iterations: int,
    timeout_s: int,
    llm_enabled: bool,
    min_solution_consensus: int,
    rename_on_pass: bool = False,
    rule1_mode: str = "none",
    id_prefix: str = "",
) -> dict[str, Any]:
    case_dir = out_dir / id_name(case_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    if image_dir is not None:
        img_found = None
        for img_name in _candidate_names(case_id, id_prefix):
            img_found = next((image_dir / f"{img_name}{s}" for s in (".jpg", ".jpeg", ".png") if (image_dir / f"{img_name}{s}").exists()), None)
            if img_found:
                break
        if img_found:
            image_path, image_source = img_found, "image_dir"
        else:
            image_path, image_source = resolve_case_image(dataset_root, case_id)
    else:
        image_path, image_source = resolve_case_image(dataset_root, case_id)
    original_path = resolve_generated_netlist(generated_dir, case_id, id_prefix)
    gt_path = resolve_ground_truth(ground_truth_dir, case_id, id_prefix)
    original_eval_dir = case_dir / "original_evaluation"
    original_validation = validate_netlist_file(case_id, original_path, original_eval_dir)
    original_gt_eval = evaluate_against_ground_truth(original_path, gt_path)

    current_text = original_path.read_text()
    current_source = "original"
    prior_contexts: list[dict[str, Any]] = []
    iterations: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None

    original_passes_rule1 = bool(
        ((original_gt_eval or {}).get("rules") or {}).get("rule1_topology_only")
    )

    # rule1_mode="skip": if original passes rule1, skip LLM iterations entirely.
    if rule1_mode == "skip" and original_passes_rule1 and not original_validation.get("passes_level_1_3"):
        final_path_skip = case_dir / "original_skipped.cir"
        write_netlist(final_path_skip, current_text)
        final_sim_skip = original_validation.get("simulation", {}) or {}
        final = {
            "status": "skipped_original_passes_rule1",
            "iteration": 0,
            "candidate_index": None,
            "solution_reason": "rule1_mode=skip: original passes rule1, skipped LLM iterations",
            "netlist_path": str(final_path_skip),
            "validation": original_validation,
            "functional": {
                "dc_ok": (final_sim_skip.get("dc") or {}).get("ok", False),
                "ac_ok": (final_sim_skip.get("ac") or {}).get("ok", False),
                "tran_ok": (final_sim_skip.get("tran") or {}).get("ok", False),
                "dc_gain": (final_sim_skip.get("dc") or {}).get("gain"),
                "ac_midband_db": (final_sim_skip.get("ac") or {}).get("midband_db"),
                "ac_bw_hz": (final_sim_skip.get("ac") or {}).get("bw_hz"),
            },
            "ground_truth_evaluation": original_gt_eval,
        }

    # If the original already passes L1-3 and either there is no GT or the original also
    # passes rule1, skip LLM repair and go straight to rename.  When GT is available
    # but the original fails rule1, we still run LLM repair so the topology can be corrected.
    if final is None and original_validation.get("passes_level_1_3") and (original_passes_rule1 or gt_path is None):
        rename_info: dict[str, Any] = {}
        final_path_str = str(original_path)
        final_validation = original_validation
        final_gt_eval = original_gt_eval
        if rename_on_pass and llm_enabled:
            rename_raw_dir = case_dir / "rename_raw"
            rename_result = call_rename_candidates(
                case_id=case_id,
                image_path=image_path,
                visualized_dir=visualized_dir,
                passing_netlist_text=current_text,
                passing_validation=original_validation,
                candidate_models=candidate_models,
                timeout_s=timeout_s,
                raw_out_dir=rename_raw_dir,
            )
            renamed_text = rename_result.get("normalized_text", "")
            if renamed_text and rename_result.get("topology_preserved"):
                renamed_path = case_dir / "renamed_original.cir"
                write_netlist(renamed_path, renamed_text)
                renamed_validation = validate_netlist_file(
                    case_id, renamed_path, case_dir / "renamed_original_validation"
                )
                rename_info = {
                    **rename_result,
                    "path": str(renamed_path),
                    "validation": renamed_validation,
                    "ground_truth_evaluation": evaluate_against_ground_truth(renamed_path, gt_path),
                }
                if renamed_validation.get("passes_level_1_3"):
                    final_path_str = str(renamed_path)
                    final_validation = renamed_validation
                    final_gt_eval = rename_info["ground_truth_evaluation"]
        status = "original_passed_level_1_3"
        if rename_info:
            status += "_renamed" if rename_info.get("validation", {}).get("passes_level_1_3") else "_rename_failed"
        final_sim = final_validation.get("simulation", {}) or {}
        functional = {
            "dc_ok": (final_sim.get("dc") or {}).get("ok", False),
            "ac_ok": (final_sim.get("ac") or {}).get("ok", False),
            "tran_ok": (final_sim.get("tran") or {}).get("ok", False),
            "dc_gain": (final_sim.get("dc") or {}).get("gain"),
            "ac_midband_db": (final_sim.get("ac") or {}).get("midband_db"),
            "ac_bw_hz": (final_sim.get("ac") or {}).get("bw_hz"),
        }
        final = {
            "status": status,
            "iteration": 0,
            "candidate_index": None,
            "solution_reason": "original_passed_level_1_3",
            "netlist_path": final_path_str,
            "validation": final_validation,
            "functional": functional,
            "ground_truth_evaluation": final_gt_eval,
        }
        if rename_info:
            final["rename"] = rename_info

    for iteration in range(1, max_iterations + 1):
        if final is not None:
            break
        iter_dir = case_dir / f"iter_{iteration:02d}"
        current_path = iter_dir / f"current_{current_source}.cir"
        write_netlist(current_path, current_text)
        current_validation = validate_netlist_file(case_id, current_path, iter_dir / "current_validation")
        current_gt_eval = evaluate_against_ground_truth(current_path, gt_path)
        current_components = parse_netlist_file(current_path)
        prior_contexts_included = bool(prior_contexts)

        def _validate_candidates(summary: dict[str, Any]) -> list[dict[str, Any]]:
            records: list[dict[str, Any]] = []
            for candidate in summary.get("candidates", []):
                record = dict(candidate)
                normalized = candidate.get("normalized_text") or ""
                if normalized:
                    candidate_path = iter_dir / f"candidate_{candidate['index']:02d}.cir"
                    write_netlist(candidate_path, normalized)
                    validation = validate_netlist_file(
                        case_id,
                        candidate_path,
                        iter_dir / f"candidate_{candidate['index']:02d}_validation",
                    )
                    gt_eval = evaluate_against_ground_truth(candidate_path, gt_path)
                    record.update(
                        {
                            "path": str(candidate_path),
                            "validation": validation,
                            "ground_truth_evaluation": gt_eval,
                        }
                    )
                    prior_contexts.append(
                        {
                            "iteration": iteration,
                            "candidate_index": candidate["index"],
                            "normalized_text": normalized,
                            "reason": candidate.get("reason"),
                            "validation": validation,
                        }
                    )
                records.append(record)
            return records

        if llm_enabled:
            # Fast models first; the judge model is only called when they fail
            # to form a finalizable (priority 1) consensus.
            fast_models = [(m, b) for m, b in candidate_models if m in FAST_MODELS]
            judge_models = [(m, b) for m, b in candidate_models if m not in FAST_MODELS]
            openrouter_summary = call_solution_candidates(
                case_id=case_id,
                image_path=image_path,
                visualized_dir=visualized_dir,
                current_text=current_text,
                current_components=current_components,
                current_validation=current_validation,
                prior_candidates=prior_contexts,
                candidate_models=fast_models,
                timeout_s=timeout_s,
                raw_out_dir=iter_dir / "llm_raw",
            )
            candidate_records = _validate_candidates(openrouter_summary)
            solution = select_solution(
                candidate_records,
                min_consensus=min_solution_consensus,
                original_passes_rule1=original_passes_rule1,
                rule1_mode=rule1_mode,
            )
            fast_consensus_final = bool(solution and solution.get("solution_priority") == 1)
            judge_called = bool(judge_models) and not fast_consensus_final
            if judge_called:
                judge_summary = call_solution_candidates(
                    case_id=case_id,
                    image_path=image_path,
                    visualized_dir=visualized_dir,
                    current_text=current_text,
                    current_components=current_components,
                    current_validation=current_validation,
                    prior_candidates=prior_contexts,
                    candidate_models=judge_models,
                    timeout_s=timeout_s,
                    raw_out_dir=iter_dir / "llm_raw",
                    index_offset=len(fast_models),
                )
                candidate_records.extend(_validate_candidates(judge_summary))
                solution = select_solution(
                    candidate_records,
                    min_consensus=min_solution_consensus,
                    original_passes_rule1=original_passes_rule1,
                    rule1_mode=rule1_mode,
                )
                openrouter_summary = {
                    **openrouter_summary,
                    "candidate_models": candidate_models,
                    "candidate_count": openrouter_summary["candidate_count"] + judge_summary["candidate_count"],
                    "exact_match_candidate_indexes": openrouter_summary["exact_match_candidate_indexes"]
                    + judge_summary["exact_match_candidate_indexes"],
                    "topology_match_candidate_indexes": openrouter_summary["topology_match_candidate_indexes"]
                    + judge_summary["topology_match_candidate_indexes"],
                    "candidates": openrouter_summary["candidates"] + judge_summary["candidates"],
                }
            elif fast_consensus_final:
                print(f"  [solution] case {case_id} iter {iteration}: fast-model consensus, judge call skipped")
            openrouter_summary["judge_called"] = judge_called
        else:
            openrouter_summary = {
                "enabled": False,
                "reason": "openrouter max cases limit reached",
                "candidates": [],
                "exact_match_candidate_indexes": [],
                "topology_match_candidate_indexes": [],
            }
            candidate_records = []
            solution = None
        iteration_record = {
            "iteration": iteration,
            "current_source": current_source,
            "current_path": str(current_path),
            "current_validation": current_validation,
            "current_ground_truth_evaluation": current_gt_eval,
            "llm_context": {
                "validation_included": True,
                "current_netlist_included": True,
                "prior_candidates_included": prior_contexts_included,
                "ground_truth_included": False,
            },
            "openrouter": {
                key: value for key, value in openrouter_summary.items() if key != "candidates"
            },
            "candidates": candidate_records,
            "solution_candidate_index": solution.get("index") if solution else None,
            "solution_reason": solution.get("solution_reason") if solution else None,
            "solution_consensus_size": solution.get("solution_consensus_size") if solution else None,
        }
        iterations.append(iteration_record)

        if solution:
            solution_validation = solution.get("validation")
            if solution_validation and solution_validation.get("passes_level_1_3"):
                final_path = solution["path"]
                final_validation = solution_validation
                final_gt_eval = solution.get("ground_truth_evaluation")
                rename_info: dict[str, Any] = {}

                if rename_on_pass and llm_enabled:
                    rename_raw_dir = iter_dir / "rename_raw"
                    rename_result = call_rename_candidates(
                        case_id=case_id,
                        image_path=image_path,
                        visualized_dir=visualized_dir,
                        passing_netlist_text=solution.get("normalized_text", ""),
                        passing_validation=solution_validation,
                        candidate_models=candidate_models,
                        timeout_s=timeout_s,
                        raw_out_dir=rename_raw_dir,
                    )
                    renamed_text = rename_result.get("normalized_text", "")
                    if renamed_text and rename_result.get("topology_preserved"):
                        renamed_path = iter_dir / "renamed_candidate.cir"
                        write_netlist(renamed_path, renamed_text)
                        renamed_validation = validate_netlist_file(
                            case_id, renamed_path, iter_dir / "renamed_validation"
                        )
                        rename_info = {
                            **rename_result,
                            "path": str(renamed_path),
                            "validation": renamed_validation,
                            "ground_truth_evaluation": evaluate_against_ground_truth(renamed_path, gt_path),
                        }
                        if renamed_validation.get("passes_level_1_3"):
                            final_path = str(renamed_path)
                            final_validation = renamed_validation
                            final_gt_eval = rename_info["ground_truth_evaluation"]

                status = "final_solution_level_1_3_passed"
                if rename_info:
                    status += "_renamed" if rename_info.get("validation", {}).get("passes_level_1_3") else "_rename_failed"

                # Extract DC/AC/TRAN results from the final validation (meaningful after rename)
                final_sim = final_validation.get("simulation", {}) or {}
                functional = {
                    "dc_ok": (final_sim.get("dc") or {}).get("ok", False),
                    "ac_ok": (final_sim.get("ac") or {}).get("ok", False),
                    "tran_ok": (final_sim.get("tran") or {}).get("ok", False),
                    "dc_gain": (final_sim.get("dc") or {}).get("gain"),
                    "ac_midband_db": (final_sim.get("ac") or {}).get("midband_db"),
                    "ac_bw_hz": (final_sim.get("ac") or {}).get("bw_hz"),
                }

                final = {
                    "status": status,
                    "iteration": iteration,
                    "candidate_index": solution["index"],
                    "solution_reason": solution.get("solution_reason"),
                    "netlist_path": final_path,
                    "validation": final_validation,
                    "functional": functional,
                    "ground_truth_evaluation": final_gt_eval,
                }
                if rename_info:
                    final["rename"] = rename_info
                break
            current_text = solution.get("normalized_text") or current_text
            current_source = f"solution_{iteration}_{solution.get('index')}"
            continue

        parsed_candidates = [record for record in candidate_records if record.get("normalized_text")]
        if parsed_candidates:
            # No solution: keep the current repair input stable and feed all candidates/validation into next iteration.
            current_source = f"no_solution_iter_{iteration}"
        else:
            current_source = f"no_candidates_iter_{iteration}"

    if final is None:
        final_path: Path | str = case_dir / "final_unresolved.cir"
        write_netlist(final_path, current_text)
        final_validation = validate_netlist_file(case_id, final_path, case_dir / "final_validation")
        rename_info: dict[str, Any] = {}
        if rename_on_pass and llm_enabled:
            rename_raw_dir = case_dir / "rename_raw_unresolved"
            rename_result = call_rename_candidates(
                case_id=case_id,
                image_path=image_path,
                visualized_dir=visualized_dir,
                passing_netlist_text=current_text,
                passing_validation=final_validation,
                candidate_models=candidate_models,
                timeout_s=timeout_s,
                raw_out_dir=rename_raw_dir,
            )
            renamed_text = rename_result.get("normalized_text", "")
            if renamed_text and rename_result.get("topology_preserved"):
                renamed_path = case_dir / "renamed_unresolved.cir"
                write_netlist(renamed_path, renamed_text)
                renamed_validation = validate_netlist_file(
                    case_id, renamed_path, case_dir / "renamed_unresolved_validation"
                )
                rename_info = {
                    **rename_result,
                    "path": str(renamed_path),
                    "validation": renamed_validation,
                    "ground_truth_evaluation": evaluate_against_ground_truth(renamed_path, gt_path),
                }
                if renamed_validation.get("passes_level_1_3"):
                    final_path = renamed_path
                    final_validation = renamed_validation
        status = "unresolved_after_max_iterations"
        if rename_info:
            status += "_renamed" if rename_info.get("validation", {}).get("passes_level_1_3") else "_rename_attempted"
        final = {
            "status": status,
            "iteration": max_iterations,
            "netlist_path": str(final_path),
            "validation": final_validation,
            "ground_truth_evaluation": evaluate_against_ground_truth(final_path, gt_path),
        }
        if rename_info:
            final["rename"] = rename_info

    result = {
        "id": case_id,
        "original_path": str(original_path),
        "generated": {
            "path": str(original_path),
            "text": read_text_if_exists(original_path),
            "validation": original_validation,
            "ground_truth_evaluation": original_gt_eval,
        },
        "image_path": str(image_path),
        "image_source": image_source,
        "ground_truth": {
            "available": gt_path is not None,
            "path": str(gt_path) if gt_path else None,
            "text": read_text_if_exists(gt_path),
            "included_in_llm_context": False,
        },
        "iterations": iterations,
        "final": final,
    }
    write_json(case_dir / "result.json", result)
    return result


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(result["final"]["status"] for result in results)
    rule_pass_counts = {rule["name"]: 0 for rule in RULES}
    generated_rule_pass_counts = {rule["name"]: 0 for rule in RULES}
    for result in results:
        rules = (result["final"].get("ground_truth_evaluation") or {}).get("rules") or {}
        for rule_name in rule_pass_counts:
            if rules.get(rule_name):
                rule_pass_counts[rule_name] += 1
        generated_rules = (result.get("generated", {}).get("ground_truth_evaluation") or {}).get("rules") or {}
        for rule_name in generated_rule_pass_counts:
            if generated_rules.get(rule_name):
                generated_rule_pass_counts[rule_name] += 1
    return {
        "case_count": len(results),
        "final_status_counts": dict(status_counts),
        "final_level_1_3_pass_count": sum(
            1 for result in results if result["final"]["validation"].get("passes_level_1_3")
        ),
        "generated_level_1_3_pass_count": sum(
            1 for result in results if result.get("generated", {}).get("validation", {}).get("passes_level_1_3")
        ),
        "ground_truth_available_count": sum(1 for result in results if result["ground_truth"]["available"]),
        "generated_rule_pass_counts": generated_rule_pass_counts,
        "final_rule_pass_counts": rule_pass_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Solution-guided LLM repair loop: candidates become solutions by LLM topology consensus, then Level 1-3 checks whether they can be finalized."
    )
    parser.add_argument("ids", nargs="*", help="Optional IDs/ranges. Omit to run all generated netlists.")
    parser.add_argument("--generated-dir", type=Path, default=Path("Data/netlist_generated"))
    parser.add_argument("--ground-truth-dir", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=Path("Data"))
    parser.add_argument(
        "--visualized-dir", type=Path, default=None,
        help="Directory of bbox-annotated images to pass to LLM. "
             "Defaults to Netlistify/test_images/visualized for the original dataset. "
             "Pass the iccad_visualized/ dir for the ICCAD dataset, or omit to disable."
    )
    parser.add_argument("--image-dir", type=Path, default=None,
                        help="Directory of base schematic images. Overrides dataset-root image lookup.")
    parser.add_argument("--id-prefix", type=str, default="",
                        help="Filename prefix for case IDs (e.g. 'a' for a200.cir).")
    parser.add_argument("--ignore-types", type=str, default="",
                        help="Comma-separated component types to ignore in R1-R5 evaluation (e.g. current_source).")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--openrouter-max-cases", type=int, default=100)
    parser.add_argument("--min-solution-consensus", type=int, default=2)
    parser.add_argument("--rename-on-pass", action="store_true", default=False,
                        help="When a candidate passes level 1-3, apply a rename-only LLM pass to give nets semantic names.")
    parser.add_argument(
        "--rule1-mode",
        choices=["none", "skip", "reject"],
        default="none",
        help=(
            "none: ignore rule1 during LLM selection (default). "
            "skip: if original already passes rule1, skip LLM iterations entirely (preserve topology). "
            "reject: if original passes rule1, reject LLM candidates that break rule1."
        ),
    )
    args = parser.parse_args()

    generated_dir = args.generated_dir.resolve()
    ground_truth_dir = args.ground_truth_dir.resolve() if args.ground_truth_dir else None
    dataset_root = args.dataset_root.resolve()
    image_dir = args.image_dir.resolve() if args.image_dir else None
    if args.ignore_types:
        for t in args.ignore_types.split(","):
            IGNORE_TYPES.add(t.strip().lower())
    if args.visualized_dir is not None:
        visualized_dir: Path | None = args.visualized_dir.resolve()
    else:
        visualized_dir = _VISUALIZED_DIR_DEFAULT if _VISUALIZED_DIR_DEFAULT.exists() else None

    ids = parse_id_ranges(args.ids) if args.ids else discover_ids(generated_dir)

    if args.out_dir is not None:
        out_dir = args.out_dir.resolve()
    else:
        id_tag = "_".join(str(i) for i in ids[:6])
        if len(ids) > 6:
            id_tag += f"_and_{len(ids) - 6}_more"
        out_dir = Path("simulation_general") / f"solution_guided_{id_tag}"
        out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    llm_cases_used = 0
    total = len(ids)
    print(f"Running {total} cases  ->  {out_dir}", flush=True)
    print(f"{'#':>4}  {'ID':>6}  {'it':>2}  {'model':<24}  {'L1':>3}  {'L2':>3}  {'L3':>3}  {'DC':>4}  {'AC':>4}  {'TR':>4}  {'R1':>3}  {'R2':>3}  {'R3':>3}  {'R4':>3}  {'R5':>3}", flush=True)
    print("-" * 101, flush=True)
    for case_num, case_id in enumerate(ids, 1):
        llm_enabled = llm_cases_used < args.openrouter_max_cases
        if llm_enabled:
            llm_cases_used += 1
        try:
            result = run_case(
                case_id=case_id,
                generated_dir=generated_dir,
                ground_truth_dir=ground_truth_dir,
                dataset_root=dataset_root,
                image_dir=image_dir,
                visualized_dir=visualized_dir,
                out_dir=out_dir,
                candidate_models=CANDIDATE_MODELS,
                max_iterations=args.max_iterations,
                timeout_s=args.timeout_s,
                llm_enabled=llm_enabled,
                min_solution_consensus=args.min_solution_consensus,
                rename_on_pass=args.rename_on_pass,
                rule1_mode=args.rule1_mode,
                id_prefix=args.id_prefix,
            )
        except FileNotFoundError as exc:
            print(f"[skip] case {case_id}: {exc}", file=sys.stderr)
            skipped.append({"id": case_id, "reason": str(exc)})
            if llm_enabled:
                llm_cases_used -= 1
            continue
        results.append(result)
        # Print original generated netlist as row 0
        gen = result.get("generated", {})
        gen_val = gen.get("validation") or {}
        gen_l1 = "Y" if gen_val.get("syntax", {}).get("ok") else "n"
        gen_l2 = "Y" if gen_val.get("connectivity", {}).get("ok") else "n"
        gen_sim = gen_val.get("simulation") or {}
        gen_op = gen_sim.get("op") or {}
        gen_l3 = "Y" if (gen_sim.get("ok") and gen_op.get("parse_ok") and gen_op.get("valid")) else "n"
        gen_gt = (gen.get("ground_truth_evaluation") or {}).get("rules") or {}
        gen_r1 = "Y" if gen_gt.get("rule1_topology_only") else "n"
        gen_r2 = "Y" if gen_gt.get("rule1_2_supply_nets") else "n"
        gen_r3 = "Y" if gen_gt.get("rule1_3_special_nets") else "n"
        gen_r4 = "Y" if gen_gt.get("rule1_4_transistor_indexes") else "n"
        gen_r5 = "Y" if gen_gt.get("rule1_5_all_component_indexes") else "n"
        print(f"{case_num:>4}  {case_id:>6}  {'0':>2}  {'[generated]':<24}  {gen_l1:>3}  {gen_l2:>3}  {gen_l3:>3}  {'':>4}  {'':>4}  {'':>4}  {gen_r1:>3}  {gen_r2:>3}  {gen_r3:>3}  {gen_r4:>3}  {gen_r5:>3}", flush=True)
        # Print one row per candidate across all iterations
        for iter_record in result.get("iterations", []):
            it_num = iter_record.get("iteration", "?")
            for cand in iter_record.get("candidates", []):
                model_name = (cand.get("model") or "?")[:24]
                val = cand.get("validation") or {}
                err = cand.get("error")
                l1 = "Y" if val.get("syntax", {}).get("ok") else ("E" if err else "n")
                l2 = "Y" if val.get("connectivity", {}).get("ok") else ("E" if err else "n")
                cand_sim = val.get("simulation") or {}
                cand_op = cand_sim.get("op") or {}
                l3 = "Y" if (cand_sim.get("ok") and cand_op.get("parse_ok") and cand_op.get("valid")) else ("E" if err else "n")
                gt_rules = (cand.get("ground_truth_evaluation") or {}).get("rules") or {}
                r1 = "Y" if gt_rules.get("rule1_topology_only") else "n"
                r2 = "Y" if gt_rules.get("rule1_2_supply_nets") else "n"
                r3 = "Y" if gt_rules.get("rule1_3_special_nets") else "n"
                r4 = "Y" if gt_rules.get("rule1_4_transistor_indexes") else "n"
                r5 = "Y" if gt_rules.get("rule1_5_all_component_indexes") else "n"
                print(f"{case_num:>4}  {case_id:>6}  {it_num:>2}  {model_name:<24}  {l1:>3}  {l2:>3}  {l3:>3}  {'':>4}  {'':>4}  {'':>4}  {r1:>3}  {r2:>3}  {r3:>3}  {r4:>3}  {r5:>3}", flush=True)
        # Final summary line with functional results
        final = result["final"]
        status = final["status"]
        final_val = final["validation"]
        final_sim = final_val.get("simulation") or {}
        final_op = final_sim.get("op") or {}
        l1 = "Y" if final_val.get("syntax", {}).get("ok") else "n"
        l2 = "Y" if final_val.get("connectivity", {}).get("ok") else "n"
        l3 = "Y" if (final_sim.get("ok") and final_op.get("parse_ok") and final_op.get("valid")) else "n"
        func = final.get("functional") or {}
        dc = "Y" if func.get("dc_ok") else "n"
        ac = "Y" if func.get("ac_ok") else "n"
        tr = "Y" if func.get("tran_ok") else "n"
        gt_rules = (final.get("ground_truth_evaluation") or {}).get("rules") or {}
        r1 = "Y" if gt_rules.get("rule1_topology_only") else "n"
        r2 = "Y" if gt_rules.get("rule1_2_supply_nets") else "n"
        r3 = "Y" if gt_rules.get("rule1_3_special_nets") else "n"
        r4 = "Y" if gt_rules.get("rule1_4_transistor_indexes") else "n"
        r5 = "Y" if gt_rules.get("rule1_5_all_component_indexes") else "n"
        print(f"{case_num:>4}  {case_id:>6}  {'':>2}  {('=> ' + status):<24}  {l1:>3}  {l2:>3}  {l3:>3}  {dc:>4}  {ac:>4}  {tr:>4}  {r1:>3}  {r2:>3}  {r3:>3}  {r4:>3}  {r5:>3}", flush=True)
        print(flush=True)

    payload = {
        "config": {
            "ids": ids,
            "generated_dir": str(generated_dir),
            "ground_truth_dir": str(ground_truth_dir) if ground_truth_dir else None,
            "ground_truth_included_in_llm_context": False,
            "dataset_root": str(dataset_root),
            "out_dir": str(out_dir),
            "candidate_models": CANDIDATE_MODELS,
            "max_iterations": args.max_iterations,
            "timeout_s": args.timeout_s,
            "openrouter_max_cases": args.openrouter_max_cases,
            "min_solution_consensus": args.min_solution_consensus,
            "rename_on_pass": args.rename_on_pass,
            "rules_affect_logic": False,
        },
        "summary": summarize_results(results),
        "skipped": skipped,
        "cases": results,
    }
    write_json(out_dir / "summary.json", payload)
    print(json.dumps(payload["summary"], indent=2))
    if skipped:
        print(f"Skipped {len(skipped)} case(s) with missing netlist/image: {[s['id'] for s in skipped]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
