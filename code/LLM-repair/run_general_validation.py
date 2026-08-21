#!/usr/bin/env python3
"""
General validation pipeline (Level 1–3) for all circuits in the dataset.

Level 1: Syntax        — parse_cir(), detect malformed netlists
Level 2: Connectivity  — floating nodes, no ground, duplicate names, etc.
Level 3: Simulation    — OP convergence, OP validity, DC sweep, AC analysis, TRAN

Does NOT run type-specific (Level 4) evaluation.
Runs on every circuit in the dataset, regardless of circuit type.

Usage:
  python3 run_general_validation.py [--workers 8] [--limit N] [--source paper|book]

Outputs:
  simulation_general/general_validation_YYYYMMDD_HHMMSS.json
"""

import argparse
import datetime
import json
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import validate as V

from netlist_parser import Component as ParsedComponent
from netlist_parser import parse_netlist_file

_tls = threading.local()  # per-thread component cache for topology-aware bias

_SUPPLY_NODES = frozenset({"VDD", "AVDD", "DVDD", "VSS", "AVSS", "DVSS", "GND", "GNDA", "VGND", "0"})

DATASET_DIR = Path(__file__).resolve().parent / "Data" / "netlist_generated"
VALIDATION_TC_BASE = DATASET_DIR
SIM_OUT_DIR = Path(__file__).resolve().parent / "simulation_general"
ALLMAP_PATH = Path("allmap.json")

VALIDATE_TYPE_ALIASES = {
    "nmos": "nmos4",
    "pmos": "pmos4",
    "current_source": "isource",
    "voltage_source": "vsource",
}


def _validate_dev_type(ctype: str) -> str:
    return VALIDATE_TYPE_ALIASES.get(ctype, ctype)


def _parse_cir_with_shared_parser(cir_path: Path) -> tuple[list[V.Component], list[str]]:
    """Adapter from the shared parser's Component format to validate.py's format."""
    try:
        parsed = parse_netlist_file(cir_path)
    except Exception as exc:
        _tls.components = []
        return [], [str(exc)]

    components: list[V.Component] = []
    for lineno, comp in enumerate(parsed, 1):
        dev_type = _validate_dev_type(comp.ctype)
        validate_comp = V.Component(comp.inst, list(comp.nodes), dev_type, lineno)
        validate_comp.params = list(comp.params)
        components.append(validate_comp)
    _tls.components = components
    return components, []


V.parse_cir = _parse_cir_with_shared_parser


PORT_EXACT_NAMES = {
    "VDD",
    "AVDD",
    "DVDD",
    "VSS",
    "AVSS",
    "DVSS",
    "GND",
    "GNDA",
    "VGND",
}
PORT_PREFIXES = (
    "VIN",
    "IN",
    "VOUT",
    "OUT",
    "IIN",
    "IOUT",
    "VB",
    "IB",
    "VBIAS",
    "IBIAS",
    "VREF",
    "VCM",
    "VCONT",
    "VCTRL",
    "VCLK",
    "LABEL_NET_",
)


def _infer_ports_from_components(components: list[ParsedComponent]) -> list[str]:
    seen: set[str] = set()
    ports: list[str] = []

    def add(node: str) -> None:
        if node == "0" or node in seen:
            return
        seen.add(node)
        ports.append(node)

    all_nodes: list[str] = []
    for comp in components:
        for node in comp.nodes:
            if node not in all_nodes:
                all_nodes.append(node)

    for preferred in ("VDD", "VSS", "GND", "GNDA", "VGND"):
        if preferred in all_nodes:
            add(preferred)

    for node in all_nodes:
        upper = node.upper()
        if upper in PORT_EXACT_NAMES or upper.startswith(PORT_PREFIXES):
            add(node)

    return ports


def _infer_ports_from_cir_path(cir_path: Path) -> list[str]:
    return _infer_ports_from_components(parse_netlist_file(cir_path))


def _read_ports_inferred(port_path: Path) -> list[str]:
    tc_id = port_path.name.removeprefix("Port").removesuffix(".txt")
    cir_path = port_path.parent / f"{tc_id}.cir"
    if not cir_path.exists():
        return []
    return _infer_ports_from_cir_path(cir_path)


V.read_ports = _read_ports_inferred
V.OUTPUT_PREFIXES = ("VOUT", "IOUT", "OUT")
V.VIN_PREFIXES = ("VIN", "IIN", "IN")


def _check_connectivity_with_inferred_ports(components: list[V.Component], ports: list[str]) -> list[str]:
    issues: list[str] = []
    port_set = set(ports)

    node_count: dict[str, int] = {}
    for comp in components:
        for node in comp.nodes:
            node_count[node] = node_count.get(node, 0) + 1

    all_nodes = set(node_count)
    if not ({"GND", "VSS", "0"} & all_nodes):
        issues.append("No ground reference (GND, VSS, or 0) in netlist")
    if "VDD" not in all_nodes:
        issues.append("No power supply (VDD) in netlist")

    for node, count in node_count.items():
        if node in {"0", "GND", "VSS", "VDD"}:
            continue
        if node not in port_set and count < 2:
            issues.append(f"Floating internal node '{node}' (only 1 connection)")

    seen: set[str] = set()
    for comp in components:
        if comp.name in seen:
            issues.append(f"Duplicate component name '{comp.name}'")
        seen.add(comp.name)

    for port in ports:
        if port not in all_nodes:
            issues.append(f"Port '{port}' inferred but not found in netlist")

    return issues


V.check_connectivity = _check_connectivity_with_inferred_ports


def _topology_aware_bias(port: str, components: list[V.Component]) -> float:
    """For VB*/IB*/VBIAS*/IBIAS* ports, pick bias based on whether the port drives nMOS or pMOS gates."""
    p = port.upper()
    nmos_count = 0
    pmos_count = 0
    for comp in components:
        if len(comp.nodes) >= 2 and comp.nodes[1].upper() == p:
            if comp.dev_type == "nmos4":
                nmos_count += 1
            elif comp.dev_type == "pmos4":
                pmos_count += 1
    if nmos_count > 0 and pmos_count == 0:
        return V.VDD_V * 0.3    # ~0.54 V — nMOS cascode / tail bias
    if pmos_count > 0 and nmos_count == 0:
        return V.VDD_V * 0.55   # ~0.99 V — pMOS cascode bias
    return 0.75                 # mixed or no gate connection → default


def _undriven_vsource_pseudo_ports(
    components: list[V.Component], existing_ports: list[str]
) -> list[str]:
    """Return the n+ node of every zero-value V* whose both terminals are internal nodes.

    These nodes need an external DC bias because their only path to a defined
    voltage is through infinite-impedance MOS gates. Driving n+ to VDD/2 lets
    ngspice find a valid OP; n- is then fixed by the 0-V source itself.
    """
    port_set = {p.upper() for p in existing_ports}
    extra: list[str] = []
    seen: set[str] = set()
    for comp in components:
        if comp.dev_type != "vsource" or len(comp.nodes) < 2:
            continue
        n_pos = comp.nodes[0].upper()
        n_neg = comp.nodes[1].upper()
        raw_params = [str(p).strip() for p in getattr(comp, "params", []) if str(p).strip()]
        has_numeric = any(_is_numeric_spice_value(p) for p in raw_params)
        if (
            n_pos not in port_set
            and n_pos not in _SUPPLY_NODES
            and n_neg not in port_set
            and n_neg not in _SUPPLY_NODES
            and not has_numeric
            and n_pos not in seen
        ):
            extra.append(comp.nodes[0])
            seen.add(n_pos)
    return extra


def _emit_source_line(port: str, bias: float | None, full: bool) -> str | None:
    """Return the SPICE voltage source line for a single port, or None if skipped."""
    if bias is None:
        return None
    p = port.upper()
    sp = port.replace('+', 'p').replace('-', 'n')
    if p in {"VSS", "VGND"}:
        return f"V{sp} {sp} 0 0"
    if full and V._is_vin(port):
        return f"V{sp} {sp} 0 DC {bias} AC 1 SIN({bias} 0.1 1MEG 0 0)"
    return f"V{sp} {sp} 0 {bias}"


def _source_lines_op_inferred(ports: list[str]) -> list[str]:
    components: list[V.Component] = getattr(_tls, "components", [])
    pseudo = _undriven_vsource_pseudo_ports(components, ports)
    lines = []
    for port in list(ports) + pseudo:
        if port.upper() in {"GND", "GNDA", "0"}:
            continue
        if port in pseudo:
            bias: float | None = V.VDD_V * 0.5
        elif port.upper().startswith(("VB", "IB", "VBIAS", "IBIAS")):
            bias = _topology_aware_bias(port, components)
        else:
            bias = V._port_bias(port)
        line = _emit_source_line(port, bias, full=False)
        if line:
            lines.append(line)
    return lines


def _source_lines_full_inferred(ports: list[str]) -> list[str]:
    components: list[V.Component] = getattr(_tls, "components", [])
    pseudo = _undriven_vsource_pseudo_ports(components, ports)
    lines = []
    for port in list(ports) + pseudo:
        if port.upper() in {"GND", "GNDA", "0"}:
            continue
        if port in pseudo:
            bias: float | None = V.VDD_V * 0.5
        elif port.upper().startswith(("VB", "IB", "VBIAS", "IBIAS")):
            bias = _topology_aware_bias(port, components)
        else:
            bias = V._port_bias(port)
        line = _emit_source_line(port, bias, full=True)
        if line:
            lines.append(line)
    return lines


V._source_lines_op = _source_lines_op_inferred
V._source_lines_full = _source_lines_full_inferred


def _mos_sizing_from_params(params: list[str]) -> str:
    if not params:
        return f"W={V.IDEAL_W} L={V.IDEAL_L}"
    keyed = [item for item in params if "=" in item]
    if keyed:
        has_w = any(item.lower().startswith("w=") for item in keyed)
        has_l = any(item.lower().startswith("l=") for item in keyed)
        defaults = []
        if not has_w:
            defaults.append(f"W={V.IDEAL_W}")
        if not has_l:
            defaults.append(f"L={V.IDEAL_L}")
        return " ".join(keyed + defaults)
    if len(params) >= 2:
        return f"W={params[0]} L={params[1]}"
    return f"W={params[0]} L={V.IDEAL_L}"


_SPICE_NUM_RE = re.compile(
    r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?(MEG|[TGMKUNPFAa])?$",
    re.IGNORECASE,
)


def _is_numeric_spice_value(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    upper = s.upper()
    if upper.startswith("DC") or upper.startswith("AC") or upper.startswith("SIN"):
        return True
    return bool(_SPICE_NUM_RE.match(s))


def _numeric_params(params: list[str]) -> list[str]:
    return [p for p in params if _is_numeric_spice_value(p)]


_LEADING_COEFF_RE = re.compile(r"^([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)\s*[A-Za-z]", re.IGNORECASE)
_ISOURCE_SWEEP_BASES: list[float] = [1e-6, 2e-6, 5e-6, 10e-6, 20e-6, 50e-6]


def _symbolic_leading_coeff(s: str) -> float:
    """Extract leading numeric coefficient from symbolic expressions like '0.5ISS' → 0.5, 'ISS' → 1.0."""
    m = _LEADING_COEFF_RE.match(s.strip())
    return float(m.group(1)) if m else 1.0


def _amps_to_spice(a: float) -> str:
    if a >= 1e-3:
        return f"{a * 1e3:.4g}m"
    if a >= 1e-6:
        return f"{a * 1e6:.4g}u"
    if a >= 1e-9:
        return f"{a * 1e9:.4g}n"
    return f"{a:.4g}"


def _make_element_lines_fn(isource_base_a: float = 1e-6):
    def _fn(components: list[V.Component]) -> list[str]:
        lines = []
        for comp in components:
            n = [x.replace('+', 'p').replace('-', 'n') for x in comp.nodes]
            raw_params = [str(item) for item in getattr(comp, "params", []) if str(item).strip()]
            numeric_params = _numeric_params(raw_params)
            value = " ".join(numeric_params)
            if comp.dev_type == "nmos4":
                sizing = _mos_sizing_from_params(numeric_params)
                lines.append(f"{comp.name} {n[0]} {n[1]} {n[2]} {n[3]} NFET {sizing}")
            elif comp.dev_type == "pmos4":
                sizing = _mos_sizing_from_params(numeric_params)
                lines.append(f"{comp.name} {n[0]} {n[1]} {n[2]} {n[3]} PFET {sizing}")
            elif comp.dev_type == "resistor":
                lines.append(f"{comp.name} {n[0]} {n[1]} {value or V.DEFAULT_R}")
            elif comp.dev_type == "capacitor":
                lines.append(f"{comp.name} {n[0]} {n[1]} {value or V.DEFAULT_C}")
            elif comp.dev_type == "inductor":
                lines.append(f"{comp.name} {n[0]} {n[1]} {value or V.DEFAULT_L}")
            elif comp.dev_type == "diode":
                lines.append(f"{comp.name} {n[0]} {n[1]} DIDEAL")
            elif comp.dev_type == "npn":
                lines.append(f"{comp.name} {n[0]} {n[1]} {n[2]} NPNQ")
            elif comp.dev_type == "pnp":
                lines.append(f"{comp.name} {n[0]} {n[1]} {n[2]} PNPQ")
            elif comp.dev_type == "isource":
                if value:
                    lines.append(f"{comp.name} {n[0]} {n[1]} {value}")
                else:
                    symbolic = raw_params[0] if raw_params else ""
                    coeff = _symbolic_leading_coeff(symbolic) if symbolic else 1.0
                    lines.append(f"{comp.name} {n[0]} {n[1]} {_amps_to_spice(coeff * isource_base_a)}")
            elif comp.dev_type == "vsource":
                lines.append(f"{comp.name} {n[0]} {n[1]} {value or '0'}")
        return lines
    return _fn


V._element_lines_ideal = _make_element_lines_fn(1e-6)


def _load_source_map() -> dict:
    if not ALLMAP_PATH.exists():
        return {}
    return {k: v.get("datasource", "").lower()
            for k, v in json.loads(ALLMAP_PATH.read_text()).items()}


def _load_type_map() -> dict:
    if not ALLMAP_PATH.exists():
        return {}
    return {k: v.get("type", "").lower()
            for k, v in json.loads(ALLMAP_PATH.read_text()).items()}


def _discover_circuits(dataset_dir: Path, filter_source=None) -> list[str]:
    """Return sorted list of tc_ids for flat *.cir or nested <id>/<id>.cir layouts."""
    source_map = _load_source_map() if filter_source else {}
    ids = []
    flat_cirs = sorted(dataset_dir.glob("*.cir"), key=lambda p: (len(p.stem), p.stem))
    if flat_cirs:
        for cir_path in flat_cirs:
            tc_id = cir_path.stem
            if filter_source and source_map.get(tc_id, "") != filter_source:
                continue
            ids.append(tc_id)
    else:
        for d in sorted(dataset_dir.iterdir(), key=lambda p: (len(p.name), p.name)):
            if not d.is_dir():
                continue
            tc_id = d.name
            if not (d / f"{tc_id}.cir").exists():
                continue
            if filter_source and source_map.get(tc_id, "") != filter_source:
                continue
            ids.append(tc_id)
    return ids


def _prepare_validation_layout(dataset_dir: Path, ids: list[str]) -> Path:
    if all((dataset_dir / tc_id / f"{tc_id}.cir").exists() for tc_id in ids):
        return dataset_dir

    staged = SIM_OUT_DIR / "_validation_cases"
    staged.mkdir(parents=True, exist_ok=True)
    for tc_id in ids:
        src = dataset_dir / f"{tc_id}.cir"
        dst_dir = staged / tc_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst_dir / f"{tc_id}.cir")
    return staged


def _case_cir_path(tc_base: Path, tc_id: str) -> Path:
    nested = tc_base / tc_id / f"{tc_id}.cir"
    if nested.exists():
        return nested
    return tc_base / f"{tc_id}.cir"


def _ports_for_tc_id(tc_id: str) -> list[str]:
    cir_path = _case_cir_path(VALIDATION_TC_BASE, tc_id)
    if not cir_path.exists():
        return []
    return _infer_ports_from_cir_path(cir_path)


def _write_inferred_port_files(ids: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, list[str]] = {}
    for tc_id in ids:
        ports = _ports_for_tc_id(tc_id)
        manifest[tc_id] = ports
        (out_dir / f"Port{tc_id}.txt").write_text("\n".join(ports) + ("\n" if ports else ""))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _run_one(tc_id: str) -> V.TCResult:
    return V.validate(tc_id, mode="ideal",
                      tc_base=VALIDATION_TC_BASE, sim_base=SIM_OUT_DIR)


def _tc_to_dict(r: V.TCResult, type_map: dict) -> dict:
    return {
        "testcase_id": str(r.tc_id),
        "circuit_type_label": type_map.get(str(r.tc_id), ""),
        "inferred_ports": _ports_for_tc_id(str(r.tc_id)),
        "syntax": {
            "ok":     r.syntax_ok,
            "errors": r.syntax_errors,
        },
        "connectivity": {
            "ok":     r.connectivity_ok,
            "issues": r.connectivity_issues,
        },
        "simulation": {
            "ok":       r.sim_ok,
            "error":    r.sim_error if not r.sim_ok else "",
            "op": {
                "parse_ok":  r.sim_parse_ok,
                "valid":     r.op_valid,
                "notes":     r.op_notes,
                "voltages":  {k: round(v, 6) for k, v in r.op_voltages.items()},
            },
            "dc": {
                "ok":        r.dc_ok,
                "gain":      round(r.dc_gain, 4),
                "inverting": r.dc_inverting,
                "swing_low":  round(r.dc_swing_low, 4),
                "swing_high": round(r.dc_swing_high, 4),
                "error":     r.dc_error,
            },
            "ac": {
                "ok":          r.ac_ok,
                "midband_db":  round(r.ac_midband_db, 3),
                "bw_hz":       round(r.ac_bw_hz, 2),
                "phase_deg":   round(r.ac_phase_deg, 2),
                "error":       r.ac_error,
            },
            "tran": {
                "ok":      r.tran_ok,
                "pp_norm": round(r.tran_pp_norm, 4),
                "error":   r.tran_error,
            },
        },
    }


def _build_summary(results: list[V.TCResult], elapsed: float, args, filter_source) -> dict:
    n = len(results)
    if n == 0:
        return {}

    def pct(x): return round(100 * x / n, 1) if n else 0

    syntax_ok   = sum(r.syntax_ok for r in results)
    conn_ok     = sum(r.connectivity_ok for r in results)
    sim_ok      = sum(r.sim_ok for r in results)
    op_parse_ok = sum(r.sim_parse_ok for r in results)
    op_valid    = sum(r.op_valid for r in results)
    dc_ok       = sum(r.dc_ok for r in results)
    ac_ok       = sum(r.ac_ok for r in results)
    tran_ok     = sum(r.tran_ok for r in results)

    # Failure breakdown — assigned to the first failing level
    fail_syntax  = sum(1 for r in results if not r.syntax_ok)
    fail_conn    = sum(1 for r in results if r.syntax_ok and not r.connectivity_ok)
    fail_conv    = sum(1 for r in results if r.syntax_ok and r.connectivity_ok and not r.sim_ok)
    fail_parse   = sum(1 for r in results if r.sim_ok and not r.sim_parse_ok)
    fail_op      = sum(1 for r in results if r.sim_parse_ok and not r.op_valid)

    return {
        "generated_at":   datetime.datetime.now().isoformat(timespec="seconds"),
        "elapsed_s":      round(elapsed, 1),
        "workers":        args.workers,
        "dataset_dir":    str(args.dataset_dir),
        "port_out_dir":   str(args.port_out_dir),
        "limit":          args.limit,
        "filter_source":  filter_source,
        "ports":          "inferred from netlist node names",
        "mode":           "ideal level-1 CMOS",
        "vdd_v":          V.VDD_V,
        "total_circuits": n,
        "rates": {
            "syntax_ok":       {"n": syntax_ok,   "pct": pct(syntax_ok)},
            "connectivity_ok": {"n": conn_ok,      "pct": pct(conn_ok)},
            "sim_converged":   {"n": sim_ok,       "pct": pct(sim_ok)},
            "op_parse_ok":     {"n": op_parse_ok,  "pct": pct(op_parse_ok)},
            "op_valid":        {"n": op_valid,      "pct": pct(op_valid)},
            "dc_ok":           {"n": dc_ok,         "pct": pct(dc_ok)},
            "ac_ok":           {"n": ac_ok,         "pct": pct(ac_ok)},
            "tran_ok":         {"n": tran_ok,       "pct": pct(tran_ok)},
        },
        "failure_breakdown": {
            "syntax":      fail_syntax,
            "connectivity": fail_conn,
            "convergence": fail_conv,
            "op_parse":    fail_parse,
            "op_invalid":  fail_op,
        },
    }


def print_summary(summary: dict) -> None:
    n = summary["total_circuits"]
    r = summary["rates"]
    f = summary["failure_breakdown"]

    print(f"\n{'═'*56}")
    print(f"  GENERAL VALIDATION SUMMARY  ({n} circuits, {summary['elapsed_s']:.0f}s)")
    print(f"{'═'*56}")
    print(f"  {'Metric':<30} {'Pass':>6} {'%':>7}")
    print(f"  {'-'*44}")
    labels = [
        ("syntax_ok",       "Syntax valid"),
        ("connectivity_ok", "Connectivity valid"),
        ("sim_converged",   "Simulation converged"),
        ("op_parse_ok",     "OP parsed"),
        ("op_valid",        "OP physically valid"),
        ("dc_ok",           "DC sweep succeeded"),
        ("ac_ok",           "AC analysis succeeded"),
        ("tran_ok",         "Transient succeeded"),
    ]
    for key, label in labels:
        v = r[key]
        print(f"  {label:<30} {v['n']:>6} {v['pct']:>6.1f}%")

    print(f"\n  Failure breakdown (first failing level):")
    for k, cnt in f.items():
        if cnt:
            print(f"    {k:<20}: {cnt}")
    print()


def main():
    global DATASET_DIR, VALIDATION_TC_BASE

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit",   type=int, default=None)
    parser.add_argument("--source",  default=None, help="paper | book")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--port-out-dir", type=Path, default=SIM_OUT_DIR / "inferred_ports")
    args = parser.parse_args()

    args.dataset_dir = args.dataset_dir.resolve()
    args.port_out_dir = args.port_out_dir.resolve()
    DATASET_DIR = args.dataset_dir
    filter_source = args.source.lower().strip() if args.source else None

    circuits = _discover_circuits(DATASET_DIR, filter_source)
    if args.limit:
        circuits = circuits[:args.limit]
    VALIDATION_TC_BASE = _prepare_validation_layout(DATASET_DIR, circuits) if circuits else DATASET_DIR
    if circuits:
        _write_inferred_port_files(circuits, args.port_out_dir)

    n = len(circuits)
    print(f"Dataset dir   : {DATASET_DIR}")
    print(f"Validation dir: {VALIDATION_TC_BASE}")
    print(f"Port out dir  : {args.port_out_dir}")
    print(f"Circuits found: {n}")
    if filter_source:
        print(f"Source filter : {filter_source}")
    print(f"Workers       : {args.workers}")
    SIM_OUT_DIR.mkdir(parents=True, exist_ok=True)

    if n == 0:
        print("Nothing to run.")
        return

    results: list[V.TCResult] = []
    done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_one, tc_id): tc_id for tc_id in circuits}
        for future in as_completed(futures):
            results.append(future.result())
            done += 1
            if done % 50 == 0 or done == n:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta  = (n - done) / rate if rate > 0 else 0
                print(f"  {done}/{n}  ({100*done//n}%)  "
                      f"{elapsed:.0f}s elapsed  ETA {eta:.0f}s", flush=True)

    results.sort(key=lambda r: (len(str(r.tc_id)), str(r.tc_id)))
    elapsed = time.time() - t0

    type_map = _load_type_map()
    summary  = _build_summary(results, elapsed, args, filter_source)

    print_summary(summary)

    # Build output
    circuits_out = [_tc_to_dict(r, type_map) for r in results]
    output = {"config": summary, "circuits": circuits_out}

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = ["general_validation", ts]
    if filter_source:
        parts.append(filter_source)
    out_path = SIM_OUT_DIR / f"{'_'.join(parts)}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"  JSON → {out_path}")


if __name__ == "__main__":
    main()
