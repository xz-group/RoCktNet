#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageFilter, ImageOps

from netlist_parser import (
    Component,
    canonical_graph_token,
    graph_fingerprint_from_components,
    graph_isomorphic,
    is_anonymous_net,
    extract_candidate_reason,
    normalize_candidate_text,
    parse_netlist_file,
    parse_netlist_text,
    topology_fingerprint_from_components,
)

try:
    import certifi
except Exception:  # pragma: no cover
    certifi = None


DEFAULT_OPENROUTER_MODEL = "gemini-3.1-flash-lite"
TOKEN_STRIP_RE = re.compile(r"^[^A-Za-z0-9+\-_]+|[^A-Za-z0-9+\-_]+$")


@dataclass(frozen=True)
class OCRToken:
    text: str
    normalized: str
    confidence: float
    variant: str


@dataclass
class MatchResult:
    expected: str
    found: bool
    match_type: str | None
    matched_text: str | None
    matched_normalized: str | None
    confidence: float | None
    variant: str | None


def canonical_ocr_token(text: str) -> str:
    token = text.strip()
    token = token.replace("–", "-").replace("—", "-").replace("−", "-")
    token = token.replace("|", "I")
    token = token.replace(" ", "")
    token = TOKEN_STRIP_RE.sub("", token)
    token = token.upper()
    return token


def canonical_exactish_token(text: str) -> str:
    return canonical_ocr_token(text).replace("_", "")


def resolve_case_image(dataset_root: Path, dataset_id: int) -> tuple[Path, str]:
    names = [str(dataset_id), f"{dataset_id:06d}"]
    case_dir = dataset_root / str(dataset_id)
    rel_path_file = case_dir / f"RelativePath{dataset_id}.txt"
    if rel_path_file.exists():
        rel = rel_path_file.read_text().strip()
        if rel:
            external = dataset_root.parent / rel
            if external.exists():
                return external.resolve(), "relative_path"
    for name in names:
        local = dataset_root / name / f"Book{name}.png"
        if local.exists():
            return local.resolve(), "book_png"
    image_dir = Path(__file__).resolve().parent / "images"
    for name in names:
        for suffix in (".jpg", ".jpeg", ".png"):
            local = image_dir / f"{name}{suffix}"
            if local.exists():
                return local.resolve(), "flat_images"
    raise FileNotFoundError(f"no image found for case {dataset_id}")


def resolve_case_netlist(dataset_root: Path, dataset_id: int, netlist_dir: Path | None = None) -> Path:
    names = [str(dataset_id), f"{dataset_id:06d}"]
    candidates: list[Path] = []
    if netlist_dir:
        for name in names:
            candidates.append(netlist_dir / f"{name}.cir")
    for name in names:
        candidates.extend(
            [
                dataset_root / name / f"{name}.cir",
                dataset_root / "netlist_generated" / f"{name}.cir",
                dataset_root / "netlist_ground_truth" / f"{name}.cir",
                dataset_root / f"{name}.cir",
            ]
        )
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"no netlist found for case {dataset_id}")


def expected_instance_tokens(components: list[Component]) -> list[str]:
    return sorted(component.inst for component in components)


def expected_named_net_tokens(components: list[Component], port_path: Path | None) -> list[str]:
    names: set[str] = set()
    for component in components:
        for net in component.nodes:
            if not is_anonymous_net(net):
                names.add(net)
    if port_path and port_path.exists():
        for raw in port_path.read_text().splitlines():
            token = canonical_graph_token(raw)
            if token:
                names.add(token)
    return sorted(names)


def build_image_variants(image_path: Path, temp_dir: Path) -> list[tuple[str, Path]]:
    image = Image.open(image_path).convert("L")
    variants: list[tuple[str, Path]] = []

    def save_variant(name: str, img: Image.Image) -> None:
        out_path = temp_dir / f"{image_path.stem}.{name}.png"
        img.save(out_path)
        variants.append((name, out_path))

    save_variant("original", image)

    base = ImageOps.autocontrast(image)
    save_variant("gray_x4", base.resize((base.width * 4, base.height * 4)))

    bw = base.resize((base.width * 5, base.height * 5)).filter(ImageFilter.SHARPEN)
    bw = bw.point(lambda p: 255 if p > 180 else 0)
    save_variant("bw_x5", bw)

    left_crop = image.crop((0, 0, max(1, int(image.width * 0.7)), image.height))
    left_crop = ImageOps.autocontrast(left_crop).resize((left_crop.width * 4, left_crop.height * 4))
    left_crop = left_crop.point(lambda p: 255 if p > 175 else 0)
    save_variant("left_bw_x4", left_crop)

    return variants


def parse_tesseract_tsv(tsv_text: str, variant: str) -> list[OCRToken]:
    lines = [line for line in tsv_text.splitlines() if line.strip()]
    if not lines:
        return []
    header = lines[0].split("\t")
    index_by_name = {name: idx for idx, name in enumerate(header)}
    out: list[OCRToken] = []
    grouped: defaultdict[tuple[str, str, str], list[str]] = defaultdict(list)
    grouped_conf: dict[tuple[str, str, str], float] = {}
    for raw in lines[1:]:
        cols = raw.split("\t")
        if len(cols) != len(header):
            continue
        text = cols[index_by_name["text"]].strip()
        if not text:
            continue
        try:
            conf = float(cols[index_by_name["conf"]])
        except ValueError:
            conf = -1.0
        normalized = canonical_ocr_token(text)
        if normalized:
            out.append(OCRToken(text=text, normalized=normalized, confidence=conf, variant=variant))
        key = (
            cols[index_by_name["block_num"]],
            cols[index_by_name["par_num"]],
            cols[index_by_name["line_num"]],
        )
        grouped[key].append(text)
        grouped_conf[key] = max(grouped_conf.get(key, -1.0), conf)

    # Composite tokens recover cases like "OUT" + "+" or "M" + "1".
    for key, pieces in grouped.items():
        if len(pieces) < 2:
            continue
        joined = canonical_ocr_token("".join(pieces))
        if joined:
            out.append(
                OCRToken(
                    text="".join(pieces),
                    normalized=joined,
                    confidence=grouped_conf.get(key, -1.0),
                    variant=variant,
                )
            )
    return out


def run_tesseract_tsv(image_path: Path, psm: int) -> str:
    command = [
        "tesseract",
        image_path.name,
        "stdout",
        "--psm",
        str(psm),
        "-l",
        "eng",
        "tsv",
        "quiet",
    ]
    proc = subprocess.run(
        command,
        cwd=str(image_path.parent),
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"tesseract failed for {image_path}")
    return proc.stdout


def collect_ocr_tokens(image_path: Path, temp_dir: Path) -> list[OCRToken]:
    tokens: list[OCRToken] = []
    for variant_name, variant_path in build_image_variants(image_path, temp_dir):
        for psm in (6, 11):
            tsv = run_tesseract_tsv(variant_path, psm)
            tokens.extend(parse_tesseract_tsv(tsv, f"{variant_name}:psm{psm}"))
    return tokens


def edit_distance_with_limit(left: str, right: str, limit: int = 1) -> int:
    if abs(len(left) - len(right)) > limit:
        return limit + 1
    prev = list(range(len(right) + 1))
    for i, lch in enumerate(left, 1):
        cur = [i]
        best = cur[0]
        for j, rch in enumerate(right, 1):
            cost = 0 if lch == rch else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
            best = min(best, cur[-1])
        if best > limit:
            return limit + 1
        prev = cur
    return prev[-1]


def match_expected_token(expected: str, tokens: Iterable[OCRToken], *, allow_fuzzy: bool) -> MatchResult:
    expected_exact = canonical_exactish_token(expected)
    best: tuple[int, float, OCRToken] | None = None
    for token in tokens:
        observed = canonical_exactish_token(token.normalized)
        if not observed:
            continue
        if observed == expected_exact:
            rank = 0
        elif allow_fuzzy and expected_exact.endswith(("+", "-")) and observed == expected_exact[:-1]:
            rank = 1
        elif (
            allow_fuzzy
            and len(expected_exact) >= 4
            and token.confidence >= 80.0
            and not any(char.isdigit() for char in expected_exact)
            and edit_distance_with_limit(expected_exact, observed, 1) <= 1
        ):
            rank = 2
        else:
            continue
        score = token.confidence
        if best is None or (rank, -score) < (best[0], -best[1]):
            best = (rank, score, token)

    if best is None:
        return MatchResult(
            expected=expected,
            found=False,
            match_type=None,
            matched_text=None,
            matched_normalized=None,
            confidence=None,
            variant=None,
        )

    rank, score, token = best
    return MatchResult(
        expected=expected,
        found=True,
        match_type="exact" if rank == 0 else ("missing_sign" if rank == 1 else "fuzzy_1edit"),
        matched_text=token.text,
        matched_normalized=token.normalized,
        confidence=score,
        variant=token.variant,
    )


def summarize_matches(expected: list[str], tokens: list[OCRToken], *, allow_fuzzy: bool) -> dict[str, Any]:
    matches = [match_expected_token(token, tokens, allow_fuzzy=allow_fuzzy) for token in expected]
    found = [match for match in matches if match.found]
    missing = [match.expected for match in matches if not match.found]
    recall = (len(found) / len(matches)) if matches else 1.0
    exact = sum(1 for match in found if match.match_type == "exact")
    fuzzy = sum(1 for match in found if match.match_type != "exact")
    return {
        "expected_count": len(matches),
        "found_count": len(found),
        "recall": round(recall, 4),
        "exact_count": exact,
        "fuzzy_count": fuzzy,
        "missing": missing,
        "examples": [asdict(match) for match in found[:12]],
    }


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return ""


def openrouter_chat(
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    timeout_s: int = 120,
    temperature: float = 0.2,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    data = json.dumps(payload)
    try:
        response = subprocess.run(
            [
                "curl",
                "--silent",
                "--show-error",
                "--fail-with-body",
                "--max-time",
                str(timeout_s),
                "--connect-timeout",
                str(min(timeout_s, 10)),
                "-X",
                "POST",
                "-H",
                f"Authorization: Bearer {api_key}",
                "-H",
                "Content-Type: application/json",
                "--data-binary",
                "@-",
                base_url,
            ],
            input=data,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        response = None

    if response is not None:
        if response.returncode != 0:
            detail = (response.stderr or response.stdout or "").strip()
            raise RuntimeError(f"OpenRouter curl failure ({response.returncode}): {detail[:1000]}")
        body = response.stdout
    else:
        request = urllib.request.Request(
            base_url,
            data=data.encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            ssl_context = ssl.create_default_context(cafile=certifi.where()) if certifi else ssl.create_default_context()
            with urllib.request.urlopen(request, timeout=timeout_s, context=ssl_context) as response_obj:
                body = response_obj.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail[:1000]}") from exc
    payload = json.loads(body)
    try:
        content = payload["choices"][0]["message"]["content"]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"unexpected OpenRouter response: {body[:1000]}") from exc
    text = message_text(content)
    if not text:
        raise RuntimeError("OpenRouter returned empty content")
    return text


def validation_context_prompt(validation_context: dict[str, Any] | None) -> str:
    if not validation_context:
        return ""

    syntax = validation_context.get("syntax") or {}
    connectivity = validation_context.get("connectivity") or {}
    simulation = validation_context.get("simulation") or {}
    op = simulation.get("op") or {}
    dc = simulation.get("dc") or {}
    ac = simulation.get("ac") or {}
    tran = simulation.get("tran") or {}
    ports = validation_context.get("inferred_ports") or []

    lines = [
        "",
        "Optional validation context from the current candidate netlist:",
        "Use this only as extra context. Do not copy it blindly; the schematic image remains the source of truth.",
        "Validation context included: true",
        f"Inferred ports: {', '.join(map(str, ports)) if ports else 'none'}",
        f"Level 1 syntax ok: {bool(syntax.get('ok'))}",
        f"Level 2 connectivity ok: {bool(connectivity.get('ok'))}",
    ]
    issues = connectivity.get("issues") or []
    if issues:
        lines.append("Connectivity issues: " + "; ".join(str(item) for item in issues[:6]))
    lines.extend(
        [
            f"Level 3 simulation converged: {bool(simulation.get('ok'))}",
            f"OP parsed: {bool(op.get('parse_ok'))}",
            f"OP physically valid: {bool(op.get('valid'))}",
        ]
    )
    notes = op.get("notes") or []
    if notes:
        lines.append("OP notes: " + "; ".join(str(item) for item in notes[:6]))
    lines.extend(
        [
            f"DC sweep ok: {bool(dc.get('ok'))}",
            f"AC analysis ok: {bool(ac.get('ok'))}",
            f"Transient analysis ok: {bool(tran.get('ok'))}",
        ]
    )
    return "\n".join(lines)


def target_netlist_context_prompt(target_netlist_text: str | None) -> str:
    if not target_netlist_text:
        return ""
    return (
        "\n\nCurrent generated netlist to revise:\n"
        f"{target_netlist_text.strip()}\n\n"
        "Revision instruction:\n"
        "Use the current generated netlist as a starting point. Correct device types, component count, connectivity, and pin order only when the schematic image supports the change. "
        "Keep correct parts unchanged. Return the revised full netlist only, with no explanation."
    )


def candidate_prompt(
    dataset_id: int,
    validation_context: dict[str, Any] | None = None,
    target_netlist_text: str | None = None,
) -> str:
    return (
        "Extract or revise the schematic into plain netlist lines.\n"
        f"Dataset ID: {dataset_id}\n"
        "First check the schematic image against the reference/current generated netlist when one is provided, then output the revised netlist.\n"
        "Topology is the priority: preserve device types, component count, connectivity, and pin order.\n"
        "Use special/external port names when the schematic supports them, especially VDD, VSS, GND, VIN, VOUT, IN, OUT, VREF, VBIAS, IBIAS, VCLK, and similar supply/input/output/bias names.\n"
        "Inferred port names from validation may be inaccurate; rename, add, or remove port-like names when the image and connectivity require it.\n"
        "Text labels are optional evidence only: not every OCR or image text label must appear in the netlist.\n"
        "Avoid floating internal nodes. If a visible text label would create a one-connection internal node, prefer the connected net name or omit that label unless it is a real external port.\n"
        "For evaluation rules, keep topology correct first, then preserve supply/ground names, then preserve special port-like names, then preserve visible component indexes when supported by the image.\n"
        "Return only lines in these exact formats:\n"
        "M*: <InstanceName> <drain> <gate> <source> <bulk> <pmos|nmos> [W] [L]\n"
        "Q*: <InstanceName> <collector> <base> <emitter> <pnp|npn> [value]\n"
        "R*/C*/L*/I*/V*: <InstanceName> <node1> <node2> [value]\n"
        "Preserve component labels from the schematic when they are visible; otherwise use prefix-matched placeholders like M1, Q1, R1, C1, L1, I1, or V1.\n"
        "Preserve visible named nets like VDD, VSS, GND, VOUT, IN+, IN-, VB1 when clear; otherwise use anonymous net names like NET1, NET2.\n"
        "Pin-order conventions must be:\n"
        "- MOS: (drain gate source bulk)\n"
        "- BJT: (collector base emitter)\n"
        "- resistor/capacitor/current_source/isource: (node1 node2)\n"
        "Keep bulk/body terminals explicit for MOS devices.\n"
        "If a visible sizing or element value is present, append it at the end of that line after the required nets/type. Examples: R1 VDD VOUT 10k; C1 VOUT GND 1p; V1 VIN GND DC 0.9; I1 N1 GND 1u; M1 D G S B nmos 1u 180n.\n"
        "Use pmos/nmos for MOS devices and pnp/npn for BJTs.\n"
        "Output format:\n"
        "Reason: one or two short sentences describing the image/reference-netlist checks and the main corrections.\n"
        "Netlist:\n"
        "<revised netlist lines only, with no comments, markdown, code fences, or unsupported extra columns>"
        f"{validation_context_prompt(validation_context)}"
        f"{target_netlist_context_prompt(target_netlist_text)}"
    )


def openrouter_verify(
    dataset_id: int,
    image_path: Path,
    target_components: list[Component],
    *,
    api_key: str,
    model: str,
    retries: int = 1,
    candidate_count: int = 3,
    timeout_s: int = 45,
    validation_context: dict[str, Any] | None = None,
    target_netlist_text: str | None = None,
) -> dict[str, Any]:
    image_bytes = image_path.read_bytes()
    mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    image_data_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    candidates: list[dict[str, Any]] = []
    temperatures = [0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95][:candidate_count]

    for index, temperature in enumerate(temperatures, 1):
        raw = ""
        error = None
        for attempt in range(retries + 1):
            try:
                raw = openrouter_chat(
                    api_key=api_key,
                    model=model,
                    temperature=temperature,
                    timeout_s=timeout_s,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": candidate_prompt(dataset_id, validation_context, target_netlist_text),
                                },
                                {"type": "image_url", "image_url": {"url": image_data_url}},
                            ],
                        }
                    ],
                )
                break
            except Exception as exc:  # pragma: no cover
                error = str(exc)
                if attempt < retries:
                    time.sleep(0.8 * (attempt + 1))
        if not raw:
            candidates.append(
                {
                    "index": index,
                    "temperature": temperature,
                    "error": error or "generation failed",
                }
            )
            continue

        normalized = normalize_candidate_text(raw)
        parsed_error = None
        parsed: list[Component] | None = None
        if normalized:
            try:
                parsed = parse_netlist_text(normalized)
            except Exception as exc:
                parsed_error = str(exc)

        metrics: dict[str, Any] = {
            "index": index,
            "temperature": temperature,
            "reason": extract_candidate_reason(raw),
            "normalized_line_count": len([line for line in normalized.splitlines() if line.strip()]),
            "graph_matches_target": False,
            "topology_matches_target": False,
            "instance_name_recall": 0.0,
            "named_net_recall": 0.0,
            "component_type_counts": {},
        }
        if parsed is not None:
            metrics["graph_matches_target"] = graph_isomorphic(parsed, target_components)
            metrics["topology_matches_target"] = graph_isomorphic(parsed, target_components, ignore_net_labels=True)
            metrics["graph_fingerprint"] = graph_fingerprint_from_components(parsed)
            metrics["topology_fingerprint"] = topology_fingerprint_from_components(parsed)
            target_instances = {component.inst for component in target_components}
            candidate_instances = {component.inst for component in parsed}
            metrics["instance_name_recall"] = round(
                len(target_instances & candidate_instances) / max(1, len(target_instances)), 4
            )
            target_named_nets = {
                net
                for component in target_components
                for net in component.nodes
                if not is_anonymous_net(net)
            }
            candidate_named_nets = {
                net
                for component in parsed
                for net in component.nodes
                if not is_anonymous_net(net)
            }
            metrics["named_net_recall"] = round(
                len(target_named_nets & candidate_named_nets) / max(1, len(target_named_nets)),
                4,
            )
            metrics["component_type_counts"] = dict(Counter(component.ctype for component in parsed))
            metrics["normalized_text"] = normalized
        if parsed_error:
            metrics["parse_error"] = parsed_error
        candidates.append(metrics)

    parsed_candidates = [candidate for candidate in candidates if candidate.get("normalized_line_count", 0) > 0]
    exact_matches = [candidate["index"] for candidate in parsed_candidates if candidate.get("graph_matches_target")]
    topology_matches = [candidate["index"] for candidate in parsed_candidates if candidate.get("topology_matches_target")]
    return {
        "enabled": True,
        "model": model,
        "candidate_count": len(candidates),
        "validation_context_included": validation_context is not None,
        "target_netlist_included_in_llm_context": target_netlist_text is not None,
        "exact_match_candidate_indexes": exact_matches,
        "topology_match_candidate_indexes": topology_matches,
        "candidates": candidates,
    }


def classify_case(
    *,
    instance_summary: dict[str, Any],
    net_summary: dict[str, Any],
    openrouter_summary: dict[str, Any] | None,
) -> tuple[str, str]:
    instance_recall = instance_summary["recall"]
    net_recall = net_summary["recall"]

    if openrouter_summary and openrouter_summary.get("topology_match_candidate_indexes"):
        strict_match_count = len(openrouter_summary.get("exact_match_candidate_indexes") or [])
        topology_match_count = len(openrouter_summary["topology_match_candidate_indexes"])
        if strict_match_count:
            return "verified", "exact labeled graph match from image-generated candidate"
        if topology_match_count >= 2:
            return "verified", "repeated exact topology matches from image-generated candidates"
        return "verified", "exact topology match from image-generated candidate"

    if openrouter_summary and openrouter_summary.get("exact_match_candidate_indexes"):
        exact_match_count = len(openrouter_summary["exact_match_candidate_indexes"])
        if instance_recall >= 0.5 and net_recall >= 0.35:
            return "verified", "exact graph match from image-generated candidate plus OCR support"
        if exact_match_count >= 2 and instance_recall >= 0.4 and net_recall >= 0.5:
            return "verified", "repeated exact graph matches from image-generated candidates plus moderate OCR support"
        if exact_match_count >= 2 and (instance_recall + net_recall) >= 0.8:
            return "verified", "repeated exact graph matches from image-generated candidates plus combined OCR support"
        return "consistent_but_unverified", "image-generated candidate matches graph but OCR support is weak"

    if openrouter_summary:
        good_candidates = [
            candidate
            for candidate in openrouter_summary["candidates"]
            if candidate.get("normalized_line_count", 0) >= 2
        ]
        strong_disagreement = [
            candidate
            for candidate in good_candidates
            if candidate.get("instance_name_recall", 0.0) <= 0.2
            and candidate.get("named_net_recall", 0.0) <= 0.2
        ]
        if len(strong_disagreement) >= 2 and instance_recall < 0.25 and net_recall < 0.25:
            return "mismatch", "multiple image-generated candidates disagree materially and OCR support is weak"

    if instance_recall >= 0.6 or net_recall >= 0.5:
        return "consistent_but_unverified", "OCR labels are broadly consistent but topology was not confirmed"
    return "uncertain", "insufficient image evidence for a high-precision automatic decision"


def summarize_case_from_precomputed(
    *,
    dataset_root: Path,
    dataset_id: int,
    netlist_path: Path,
    components: list[Component],
    image_path: Path,
    image_source: str,
    port_path: Path,
    ocr_tokens: list[OCRToken],
    instance_summary: dict[str, Any],
    net_summary: dict[str, Any],
    openrouter_model: str | None,
    openrouter_candidate_count: int,
    openrouter_timeout_s: int,
    run_openrouter: bool,
    validation_context: dict[str, Any] | None,
    validation_context_source: str | None,
    include_target_netlist_in_context: bool,
) -> dict[str, Any]:
    openrouter_summary = None
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if openrouter_model and run_openrouter:
        openrouter_summary = openrouter_verify(
            dataset_id,
            image_path,
            components,
            api_key=api_key,
            model=openrouter_model,
            candidate_count=openrouter_candidate_count,
            timeout_s=openrouter_timeout_s,
            validation_context=validation_context,
            target_netlist_text=netlist_path.read_text() if include_target_netlist_in_context else None,
        )
    elif openrouter_model:
        openrouter_summary = {
            "enabled": False,
            "reason": "case did not meet OCR gating thresholds for OpenRouter verification",
        }

    status, reason = classify_case(
        instance_summary=instance_summary,
        net_summary=net_summary,
        openrouter_summary=openrouter_summary if openrouter_summary and openrouter_summary.get("enabled") else None,
    )

    return {
        "id": dataset_id,
        "status": status,
        "reason": reason,
        "paths": {
            "netlist": str(netlist_path.resolve()),
            "image": str(image_path),
            "image_source": image_source,
            "port": str(port_path.resolve()) if port_path.exists() else None,
            "validation_context": validation_context_source,
        },
        "netlist": {
            "component_count": len(components),
            "component_type_counts": dict(Counter(component.ctype for component in components)),
            "instance_count": len(expected_instance_tokens(components)),
            "named_net_count": len(expected_named_net_tokens(components, port_path)),
            "graph_fingerprint": graph_fingerprint_from_components(components),
            "topology_fingerprint": topology_fingerprint_from_components(components),
        },
        "ocr": {
            "raw_token_count": len(ocr_tokens),
            "instance_summary": instance_summary,
            "named_net_summary": net_summary,
        },
        "validation_context": {
            "included_in_llm_context": validation_context is not None,
            "source": validation_context_source,
        },
        "target_netlist_context": {
            "included_in_llm_context": include_target_netlist_in_context,
        },
        "openrouter": openrouter_summary,
    }


def load_validation_contexts(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    data = json.loads(path.read_text())
    rows = data.get("circuits", data) if isinstance(data, dict) else data
    contexts: dict[int, dict[str, Any]] = {}
    if not isinstance(rows, list):
        return contexts
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_id = row.get("testcase_id", row.get("id"))
        if raw_id is None:
            continue
        try:
            contexts[int(str(raw_id))] = row
        except ValueError:
            continue
    return contexts


def parse_id_ranges(values: list[str]) -> list[int]:
    out: list[int] = []
    for value in values:
        if "-" in value:
            start_raw, end_raw = value.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if end < start:
                raise ValueError(f"invalid range: {value}")
            out.extend(range(start, end + 1))
        else:
            out.append(int(value))
    return sorted(dict.fromkeys(out))


def write_json_report(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2))


def write_csv_report(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "status",
                "reason",
                "image_source",
                "component_count",
                "ocr_instance_recall",
                "ocr_named_net_recall",
                "openrouter_exact_match",
                "openrouter_topology_match",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "status": row["status"],
                    "reason": row["reason"],
                    "image_source": row["paths"]["image_source"],
                    "component_count": row["netlist"]["component_count"],
                    "ocr_instance_recall": row["ocr"]["instance_summary"]["recall"],
                    "ocr_named_net_recall": row["ocr"]["named_net_summary"]["recall"],
                    "openrouter_exact_match": bool(
                        row.get("openrouter", {}).get("exact_match_candidate_indexes")
                        if row.get("openrouter")
                        else False
                    ),
                    "openrouter_topology_match": bool(
                        row.get("openrouter", {}).get("topology_match_candidate_indexes")
                        if row.get("openrouter")
                        else False
                    ),
                }
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Conservative verifier for Book<ID>.png / RelativePath<ID>.txt against <ID>.cir",
    )
    parser.add_argument(
        "ids",
        nargs="+",
        help="IDs or ranges like 3551 3556 3701-3816",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Dataset root directory",
    )
    parser.add_argument(
        "--netlist-dir",
        type=Path,
        help="Optional flat directory containing <id>.cir files, e.g. Data/netlist_generated",
    )
    parser.add_argument(
        "--temp-root",
        type=Path,
        default=Path(__file__).resolve().parent / ".verifier_tmp",
        help="Temporary directory for OCR variants",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional JSON report output path",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        help="Optional CSV report output path",
    )
    parser.add_argument(
        "--openrouter-model",
        default=DEFAULT_OPENROUTER_MODEL,
        help="Optional OpenRouter multimodal model name for candidate-netlist corroboration",
    )
    parser.add_argument(
        "--openrouter-candidate-count",
        type=int,
        default=3,
        help="Number of multimodal candidate netlists to generate per case",
    )
    parser.add_argument(
        "--openrouter-min-instance-recall",
        type=float,
        default=0.5,
        help="Minimum OCR instance-label recall required before running OpenRouter for a case",
    )
    parser.add_argument(
        "--openrouter-min-net-recall",
        type=float,
        default=0.35,
        help="Minimum OCR named-net recall required before running OpenRouter for a case",
    )
    parser.add_argument(
        "--openrouter-max-cases",
        type=int,
        help="Optional cap on how many gated cases will run through OpenRouter",
    )
    parser.add_argument(
        "--openrouter-timeout-s",
        type=int,
        default=45,
        help="Per-request timeout for OpenRouter API calls",
    )
    parser.add_argument(
        "--validation-context-report",
        type=Path,
        help="Optional run_general_validation JSON; when set, matching case summaries are added only to the LLM prompt context",
    )
    parser.add_argument(
        "--include-target-netlist-in-context",
        action="store_true",
        help="Add the current generated netlist to the LLM prompt so the model revises it instead of generating independently",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    netlist_dir = args.netlist_dir.resolve() if args.netlist_dir else None
    temp_root = args.temp_root.resolve()
    ids = parse_id_ranges(args.ids)
    validation_contexts = load_validation_contexts(args.validation_context_report.resolve() if args.validation_context_report else None)
    validation_context_source = str(args.validation_context_report.resolve()) if args.validation_context_report else None

    rows: list[dict[str, Any]] = []
    failures: list[tuple[int, str]] = []
    openrouter_cases_run = 0
    for dataset_id in ids:
        try:
            case_dir = dataset_root / str(dataset_id)
            netlist_path = resolve_case_netlist(dataset_root, dataset_id, netlist_dir)
            port_path = case_dir / f"Port{dataset_id}.txt"
            components = parse_netlist_file(netlist_path)
            image_path, image_source = resolve_case_image(dataset_root, dataset_id)
            case_temp = temp_root / str(dataset_id)
            case_temp.mkdir(parents=True, exist_ok=True)
            ocr_tokens = collect_ocr_tokens(image_path, case_temp)

            instance_summary = summarize_matches(expected_instance_tokens(components), ocr_tokens, allow_fuzzy=False)
            net_summary = summarize_matches(expected_named_net_tokens(components, port_path), ocr_tokens, allow_fuzzy=True)

            run_openrouter = False
            if args.openrouter_model:
                run_openrouter = (
                    instance_summary["recall"] >= args.openrouter_min_instance_recall
                    and net_summary["recall"] >= args.openrouter_min_net_recall
                )
                if args.openrouter_max_cases is not None and openrouter_cases_run >= args.openrouter_max_cases:
                    run_openrouter = False

            rows.append(
                summarize_case_from_precomputed(
                    dataset_root=dataset_root,
                    dataset_id=dataset_id,
                    netlist_path=netlist_path,
                    components=components,
                    image_path=image_path,
                    image_source=image_source,
                    port_path=port_path,
                    ocr_tokens=ocr_tokens,
                    instance_summary=instance_summary,
                    net_summary=net_summary,
                    openrouter_model=args.openrouter_model,
                    openrouter_candidate_count=args.openrouter_candidate_count,
                    openrouter_timeout_s=args.openrouter_timeout_s,
                    run_openrouter=run_openrouter,
                    validation_context=validation_contexts.get(dataset_id),
                    validation_context_source=validation_context_source,
                    include_target_netlist_in_context=args.include_target_netlist_in_context,
                )
            )
            if run_openrouter:
                openrouter_cases_run += 1
        except Exception as exc:
            failures.append((dataset_id, str(exc)))

    if args.json_out:
        write_json_report(args.json_out.resolve(), rows)
    if args.csv_out:
        write_csv_report(args.csv_out.resolve(), rows)

    print(json.dumps(rows, indent=2))
    if failures:
        print(json.dumps({"failures": failures}, indent=2), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
