#!/usr/bin/env python3
"""
Netlist Validation Pipeline — Extended
Stages: syntax → connectivity → ngspice simulation (OP + DC + AC + TRAN)
Modes : 'ideal'  — generic SPICE level-1 CMOS (default)
        'sky130' — SkyWater 130nm PDK BSIM4
"""

import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
TESTCASE_DIR = BASE_DIR / "testcases"
SIM_DIR = BASE_DIR / "simulation"

SKY130_TT = Path(
   
)

TESTCASE_IDS = [3, 4, 5, 6, 8, 9, 10, 14, 15, 17, 18, 20, 21, 100]

# ── Constants ─────────────────────────────────────────────────────────────────
VDD_V = 1.8
DEFAULT_R = "10k"
DEFAULT_C = "1p"

IDEAL_L = "0.5u"
IDEAL_W = "5u"
NMOS_MODEL_IDEAL  = ".model NFET  nmos level=1 vto=0.5  kp=120e-6 lambda=0.02 gamma=0.4 phi=0.6"
PMOS_MODEL_IDEAL  = ".model PFET  pmos level=1 vto=-0.5 kp=60e-6  lambda=0.02 gamma=0.4 phi=0.6"
NPN_MODEL_IDEAL   = ".model NPNQ  npn  bf=100 is=1e-15 vaf=100"
PNP_MODEL_IDEAL   = ".model PNPQ  pnp  bf=100 is=1e-15 vaf=100"
DIODE_MODEL_IDEAL = ".model DIDEAL D   is=1e-14 n=1"
DEFAULT_L = "10n"  # inductor default value

# sky130: L/W in µm (sizing_mix.py convention: L ∈ [0.15,1.0], W ∈ [0.5,10.0])
SKY130_L = "0.5"
SKY130_W = "5"
SKY130_NFET = "sky130_fd_pr__nfet_01v8"
SKY130_PFET = "sky130_fd_pr__pfet_01v8"

PORT_BIAS: Dict[str, float] = {
    "VDD":  VDD_V,
    "VSS":  0.0,
    "VIN1": VDD_V * 0.5,
    "VIN2": VDD_V * 0.5,
    "VB1":  0.75,
    "VB2":  1.05,
}
OUTPUT_PREFIXES: Tuple[str, ...] = ("VOUT", "IOUT")
VIN_PREFIXES:    Tuple[str, ...] = ("VIN", "IIN")


def _port_bias(port: str) -> Optional[float]:
    """Return DC bias for a port; None means it is an output (leave floating)."""
    p = port.upper()
    if any(p.startswith(op) for op in OUTPUT_PREFIXES):
        return None
    exact = PORT_BIAS.get(port)
    if exact is not None:
        return exact
    if p in ("VSS", "GND", "GNDA", "VGND") or p.startswith("VSS"):
        return 0.0
    if p.startswith("VDD") or p.startswith("AVDD") or p.startswith("DVDD"):
        return VDD_V
    if any(p.startswith(vp) for vp in VIN_PREFIXES):
        return VDD_V * 0.5
    if p.startswith(("VB", "IB", "VBIAS", "IBIAS")):
        return 0.75
    if p.startswith(("VREF", "VCM", "VCONT", "VCTRL")):
        return VDD_V * 0.5
    if p.startswith("VCLK"):
        return 0.0
    return VDD_V * 0.5


# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class Component:
    name: str
    nodes: List[str]
    dev_type: str
    line: int


@dataclass
class TCResult:
    tc_id: int
    mode: str
    # Stage 1 — Syntax
    syntax_ok: bool = False
    syntax_errors: List[str] = field(default_factory=list)
    # Stage 2 — Connectivity
    connectivity_ok: bool = False
    connectivity_issues: List[str] = field(default_factory=list)
    # Stage 3a — Operating Point
    sim_ok: bool = False
    sim_error: str = ""
    sim_parse_ok: bool = False
    op_voltages: Dict[str, float] = field(default_factory=dict)
    op_valid: bool = False
    op_notes: List[str] = field(default_factory=list)
    # Stage 3b — DC Sweep
    dc_ok: bool = False
    dc_gain: float = 0.0
    dc_inverting: Optional[bool] = None
    dc_swing_low: float = 0.0
    dc_swing_high: float = 0.0
    dc_error: str = ""
    # Stage 3c — AC Analysis
    ac_ok: bool = False
    ac_midband_db: float = 0.0
    ac_bw_hz: float = 0.0
    ac_phase_deg: float = 0.0
    ac_error: str = ""
    # Stage 3d — Transient
    tran_ok: bool = False
    tran_pp_norm: float = 0.0
    tran_error: str = ""
    # Circuit type + functional
    circuit_type: str = "unknown"
    functional_valid: bool = False
    functional_notes: List[str] = field(default_factory=list)


# ── 1. Syntax parsing ─────────────────────────────────────────────────────────
def parse_cir(cir_path: Path) -> Tuple[List[Component], List[str]]:
    components: List[Component] = []
    errors: List[str] = []

    with open(cir_path) as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("*"):
                continue

            m = re.match(r"^(\w+)\s*\(([^)]+)\)\s+(\w+)\s*$", line)
            if not m:
                errors.append(f"L{lineno}: cannot parse '{line}'")
                continue

            name, nodes_str, dev_type = m.groups()
            nodes = nodes_str.split()
            dt = dev_type.lower()

            if dt in ("nmos4", "pmos4"):
                if len(nodes) != 4:
                    errors.append(f"L{lineno}: {dev_type} needs 4 nodes, got {len(nodes)}")
                    continue
            elif dt in ("resistor", "capacitor", "inductor", "diode"):
                if len(nodes) != 2:
                    errors.append(f"L{lineno}: {dev_type} needs 2 nodes, got {len(nodes)}")
                    continue
            elif dt in ("npn", "pnp"):
                if len(nodes) != 4:
                    errors.append(f"L{lineno}: {dev_type} needs 4 nodes (c b e s), got {len(nodes)}")
                    continue
            elif dt == "inverter":
                if len(nodes) != 4:
                    errors.append(f"L{lineno}: INVERTER needs 4 nodes (in out VDD VSS), got {len(nodes)}")
                    continue
            elif dt == "transmission_gate":
                if len(nodes) != 5:
                    errors.append(f"L{lineno}: TRANSMISSION_GATE needs 5 nodes (in out ctrl VDD VSS), got {len(nodes)}")
                    continue
            else:
                errors.append(f"L{lineno}: unknown device type '{dev_type}'")
                continue

            components.append(Component(name, nodes, dt, lineno))

    return components, errors


def read_ports(port_path: Path) -> List[str]:
    return port_path.read_text().split()


# ── 2. Connectivity validation ────────────────────────────────────────────────
def check_connectivity(components: List[Component], ports: List[str]) -> List[str]:
    issues: List[str] = []
    port_set = set(ports)

    node_count: Dict[str, int] = {}
    for comp in components:
        for n in comp.nodes:
            node_count[n] = node_count.get(n, 0) + 1

    all_nodes = set(node_count)

    if "VSS" not in all_nodes and "0" not in all_nodes:
        issues.append("No ground reference (VSS or 0) in netlist")
    if "VDD" not in all_nodes:
        issues.append("No power supply (VDD) in netlist")

    for node, cnt in node_count.items():
        # '0' is SPICE intrinsic ground (used as BJT substrate); never flag it
        if node == "0":
            continue
        # V.../I... nets are ports (voltage/current sources or named signals)
        if node[0].upper() in ("V", "I"):
            continue
        if node not in port_set and cnt < 2:
            issues.append(f"Floating internal node '{node}' (only 1 connection)")

    seen: set = set()
    for comp in components:
        if comp.name in seen:
            issues.append(f"Duplicate component name '{comp.name}'")
        seen.add(comp.name)

    for p in ports:
        if p not in all_nodes:
            issues.append(f"Port '{p}' declared but not found in netlist")

    return issues


# ── 3. Circuit type classification ────────────────────────────────────────────
def _is_output(port: str) -> bool:
    return any(port.startswith(op) for op in OUTPUT_PREFIXES)


def _is_vin(port: str) -> bool:
    return any(port.upper().startswith(vp) for vp in VIN_PREFIXES)


def classify_circuit_topology(components: List[Component], ports: List[str]) -> str:
    """Topology-based classification — used for sky130 and fallback cases."""
    port_set = set(ports)
    output_ports = {p for p in port_set if _is_output(p)}

    if not output_ports:
        return "no_output"
    vin_ports = [p for p in ports if _is_vin(p)]
    if not vin_ports:
        return "static_bias"

    vin1 = vin_ports[0]
    nmos = [c for c in components if c.dev_type == "nmos4"]
    pmos = [c for c in components if c.dev_type == "pmos4"]
    has_both = bool(nmos) and bool(pmos)
    has_vb = any(p.startswith("VB") for p in port_set)

    for c in nmos + pmos:
        gate, drain, source = c.nodes[1], c.nodes[0], c.nodes[2]
        if gate == vin1:
            if drain in output_ports:
                if has_both:
                    return "complementary_cs"
                if has_vb:
                    return "cascode"
                return "common_source"
            if source in output_ports:
                return "source_follower"

    if has_vb:
        return "cascode"
    return "unknown"


def classify_from_measurements(
    dc_gain: float, dc_inverting: Optional[bool], ports: List[str]
) -> str:
    """Data-driven classification from DC sweep results."""
    port_set = set(ports)
    if not any(_is_output(p) for p in port_set):
        return "no_output"
    if not any(_is_vin(p) for p in port_set):
        return "static_bias"
    if dc_inverting is None:
        return "unknown"
    if dc_inverting and dc_gain > 1.5:
        return "inverting_amplifier"
    if dc_inverting and dc_gain <= 1.5:
        return "weak_inverter"
    if not dc_inverting and 0.3 <= dc_gain <= 1.5:
        return "source_follower"
    if not dc_inverting and dc_gain > 1.5:
        return "non_inv_amplifier"
    return "unknown"


# ── 4. SPICE generation ───────────────────────────────────────────────────────
def _spice_net(net: str) -> str:
    """Sanitize a net name for SPICE: replace + and - with p and n."""
    return net.replace('+', 'p').replace('-', 'n')


def _ordered_net_nodes(components: List[Component]) -> List[str]:
    seen: set = set()
    ordered: List[str] = []
    for comp in components:
        for n in comp.nodes:
            if n not in seen:
                seen.add(n)
                ordered.append(n)
    return ordered


def _source_lines_op(ports: List[str]) -> List[str]:
    """Sources for OP-only simulation (DC values only)."""
    lines = []
    for port in ports:
        v = _port_bias(port)
        if v is None:
            continue  # output — leave floating
        sp = _spice_net(port)
        if port.upper() in ("VSS", "GND", "GNDA"):
            lines.append(f"V{sp} {sp} 0 0")
        else:
            lines.append(f"V{sp} {sp} 0 {v}")
    return lines


def _source_lines_full(ports: List[str]) -> List[str]:
    """Sources for full analysis; VIN ports get AC + SIN stimulus."""
    lines = []
    for port in ports:
        v = _port_bias(port)
        if v is None:
            continue  # output — leave floating
        sp = _spice_net(port)
        if port.upper() in ("VSS", "GND", "GNDA"):
            lines.append(f"V{sp} {sp} 0 0")
        elif _is_vin(port):
            lines.append(f"V{sp} {sp} 0 DC {v} AC 1 SIN({v} 0.1 1MEG 0 0)")
        else:
            lines.append(f"V{sp} {sp} 0 {v}")
    return lines


def _element_lines_ideal(components: List[Component]) -> List[str]:
    lines = []
    for comp in components:
        n = [_spice_net(x) for x in comp.nodes]
        if comp.dev_type == "nmos4":
            lines.append(f"{comp.name} {n[0]} {n[1]} {n[2]} {n[3]} NFET W={IDEAL_W} L={IDEAL_L}")
        elif comp.dev_type == "pmos4":
            lines.append(f"{comp.name} {n[0]} {n[1]} {n[2]} {n[3]} PFET W={IDEAL_W} L={IDEAL_L}")
        elif comp.dev_type == "resistor":
            lines.append(f"{comp.name} {n[0]} {n[1]} {DEFAULT_R}")
        elif comp.dev_type == "capacitor":
            lines.append(f"{comp.name} {n[0]} {n[1]} {DEFAULT_C}")
        elif comp.dev_type == "inductor":
            lines.append(f"{comp.name} {n[0]} {n[1]} {DEFAULT_L}")
        elif comp.dev_type == "diode":
            lines.append(f"{comp.name} {n[0]} {n[1]} DIDEAL")
        elif comp.dev_type == "npn":
            # nodes: c b e s — use c b e only (substrate '0' is SPICE intrinsic)
            lines.append(f"{comp.name} {n[0]} {n[1]} {n[2]} NPNQ")
        elif comp.dev_type == "pnp":
            lines.append(f"{comp.name} {n[0]} {n[1]} {n[2]} PNPQ")
        elif comp.dev_type == "inverter":
            # nodes: in out VDD VSS → expand to PMOS + NMOS
            lines.append(f"M{comp.name}p {n[1]} {n[0]} {n[2]} {n[2]} PFET W={IDEAL_W} L={IDEAL_L}")
            lines.append(f"M{comp.name}n {n[1]} {n[0]} {n[3]} {n[3]} NFET W={IDEAL_W} L={IDEAL_L}")
        elif comp.dev_type == "transmission_gate":
            # nodes: in out ctrl VDD VSS → NMOS + PMOS in parallel
            lines.append(f"M{comp.name}n {n[1]} {n[2]} {n[0]} {n[4]} NFET W={IDEAL_W} L={IDEAL_L}")
            lines.append(f"M{comp.name}p {n[1]} {n[2]} {n[0]} {n[3]} PFET W={IDEAL_W} L={IDEAL_L}")
    return lines


def _element_lines_sky130(components: List[Component]) -> List[str]:
    lines = []
    for comp in components:
        n = comp.nodes
        if comp.dev_type == "nmos4":
            lines.append(
                f"X{comp.name} {n[0]} {n[1]} {n[2]} {n[3]}"
                f" {SKY130_NFET} l={SKY130_L} w={SKY130_W}"
            )
        elif comp.dev_type == "pmos4":
            lines.append(
                f"X{comp.name} {n[0]} {n[1]} {n[2]} {n[3]}"
                f" {SKY130_PFET} l={SKY130_L} w={SKY130_W}"
            )
        elif comp.dev_type == "resistor":
            lines.append(f"{comp.name} {n[0]} {n[1]} {DEFAULT_R}")
        elif comp.dev_type == "capacitor":
            lines.append(f"{comp.name} {n[0]} {n[1]} {DEFAULT_C}")
    return lines


def generate_spice_ideal_full(
    tc_id: str, components: List[Component], ports: List[str], workdir: Path
) -> Path:
    """Combined OP + DC + AC + TRAN SPICE for ideal mode."""
    vin_ports = [p for p in ports if _is_vin(p)]
    has_vout  = any(_is_output(p) for p in ports)

    net_nodes = _ordered_net_nodes(components)
    op_print  = " ".join(f"v({_spice_net(n).lower()})" for n in net_nodes)

    # First output node (lowercase) for DC/AC/TRAN
    vout_node = next(
        (_spice_net(n).lower() for n in net_nodes if _is_output(n)),
        None,
    )

    lines = [
        f"* Testcase {tc_id} — Full Analysis (ideal level-1 CMOS)",
        NMOS_MODEL_IDEAL,
        PMOS_MODEL_IDEAL,
        NPN_MODEL_IDEAL,
        PNP_MODEL_IDEAL,
        DIODE_MODEL_IDEAL,
        ".options reltol=1e-3 itl1=500 itl2=100",
        "",
        "* Sources",
        *_source_lines_full(ports),
        "",
        "* Circuit",
        *_element_lines_ideal(components),
        "",
        ".control",
        "set filetype=ascii",
        "",
        "op",
        "echo OP_START",
        f"print {op_print}",
        "echo OP_END",
    ]

    if vin_ports and has_vout and vout_node:
        vin_src = f"V{_spice_net(vin_ports[0])}"
        lines += [
            "",
            f"dc {vin_src} 0 {VDD_V} {VDD_V/20:.4f}",
            "echo DC_START",
            f"print v({vout_node})",
            "echo DC_END",
            "",
            "ac dec 10 1 1G",
            "echo AC_START",
            f"print frequency vdb({vout_node}) vp({vout_node})",
            "echo AC_END",
            "",
            "tran 5n 2u",
            "echo TRAN_START",
            f"print time v({vout_node})",
            "echo TRAN_END",
        ]

    lines += ["", ".endc", "", ".end"]

    workdir.mkdir(parents=True, exist_ok=True)
    spice_file = workdir / f"tc{tc_id}_full_ideal.spice"
    spice_file.write_text("\n".join(lines) + "\n")
    return spice_file


def generate_spice_sky130(
    tc_id: int, components: List[Component], ports: List[str], workdir: Path
) -> Path:
    """Sky130 OP-only SPICE. Wraps DUT in .subckt to resolve BSIM4 models in ngspice-46."""
    port_set = set(ports)
    net_nodes = _ordered_net_nodes(components)
    subckt_ports = " ".join(ports)

    print_parts = []
    for n in net_nodes:
        if n in port_set:
            print_parts.append(f"v({n.lower()})")
        else:
            print_parts.append(f"v(xdut.{n.lower()})")
    print_targets = " ".join(print_parts)

    lines = [
        f"* Testcase {tc_id} — OP Analysis (sky130 PDK)",
        f".include {SKY130_TT}",
        ".param mc_mm_switch=0 mc_pr_switch=0",
        ".options reltol=1e-3 itl1=500 itl2=100",
        "",
        f".subckt dut {subckt_ports}",
        *_element_lines_sky130(components),
        ".ends dut",
        "",
        *_source_lines_op(ports),
        "",
        f"XDUT {subckt_ports} dut",
        "",
        ".control",
        "set filetype=ascii",
        "op",
        "echo OP_START",
        f"print {print_targets}",
        "echo OP_END",
        ".endc",
        "",
        ".end",
    ]

    workdir.mkdir(parents=True, exist_ok=True)
    spice_file = workdir / f"tc{tc_id}_op_sky130.spice"
    spice_file.write_text("\n".join(lines) + "\n")
    return spice_file


# ── 5. Simulation runner ──────────────────────────────────────────────────────
def run_ngspice(spice_file: Path) -> Tuple[bool, str]:
    log_file = spice_file.with_suffix(".log")
    try:
        proc = subprocess.run(
            ["ngspice", "-b", str(spice_file), "-o", str(log_file)],
            capture_output=True, text=True, timeout=120,
        )
        log = log_file.read_text() if log_file.exists() else ""
        log += proc.stderr
        return proc.returncode == 0, log
    except subprocess.TimeoutExpired:
        return False, "ngspice timed out (>120s)"
    except FileNotFoundError:
        return False, "ngspice not found in PATH"


# ── 6. Result parsing ─────────────────────────────────────────────────────────
def parse_op_voltages(log: str) -> Optional[Dict[str, float]]:
    m = re.search(r"OP_START(.*?)OP_END", log, re.DOTALL)
    if not m:
        return None
    voltages: Dict[str, float] = {}
    for vm in re.finditer(
        r"v\((?:xdut\.)?(\w+)\)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        m.group(1), re.IGNORECASE,
    ):
        voltages[vm.group(1).upper()] = float(vm.group(2))
    return voltages


def _parse_table(log: str, start: str, end: str) -> Optional[List[List[float]]]:
    """Extract rows from ngspice tabular output between markers.
    ngspice format:  Index<tab>col0<tab>col1...   (Index is an integer)
    Returns list of [col0, col1, ...] float rows (Index stripped).
    """
    m = re.search(re.escape(start) + r"(.*?)" + re.escape(end), log, re.DOTALL)
    if not m:
        return None

    rows: List[List[float]] = []
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line or "Index" in line or line.startswith("-") or line.startswith("*"):
            continue
        parts = re.split(r"\s+", line)
        try:
            int(parts[0])                          # first field = row index
            vals = [float(p) for p in parts[1:] if p]
            if vals:
                rows.append(vals)
        except (ValueError, IndexError):
            continue

    return rows or None


def _compute_dc_metrics(
    rows: List[List[float]],
) -> Optional[Dict]:
    """Compute gain and swing from DC sweep table.
    Column 0 is the v-sweep (= VIN), column 1 is v(vout).
    """
    if not rows or len(rows) < 3:
        return None
    vin  = [r[0] for r in rows if len(r) >= 2]
    vout = [r[1] for r in rows if len(r) >= 2]

    # Peak signed gain over consecutive steps
    max_g = 0.0
    for i in range(len(vin) - 1):
        dv = vin[i + 1] - vin[i]
        if abs(dv) < 1e-9:
            continue
        g = (vout[i + 1] - vout[i]) / dv
        if abs(g) > abs(max_g):
            max_g = g

    return {
        "gain":      abs(max_g),
        "inverting": max_g < 0,
        "vout_min":  min(vout),
        "vout_max":  max(vout),
        "swing":     max(vout) - min(vout),
    }


def _compute_ac_metrics(rows: List[List[float]]) -> Optional[Dict]:
    """Compute midband gain (dB), bandwidth, and phase from AC table.
    Columns: [frequency, vdb(vout), vp(vout)]
    """
    if not rows or len(rows) < 2:
        return None
    valid = [r for r in rows if len(r) >= 3]
    if not valid:
        return None

    freqs    = [r[0] for r in valid]
    gains_db = [r[1] for r in valid]
    phases   = [r[2] for r in valid]

    midband_db = sum(gains_db[:5]) / min(5, len(gains_db))
    midband_phase_deg = math.degrees(phases[0])

    # -3 dB bandwidth
    bw = freqs[-1]
    for i, g in enumerate(gains_db):
        if g < midband_db - 3.0:
            bw = freqs[i]
            break

    return {
        "midband_db":  midband_db,
        "bw_hz":       bw,
        "phase_deg":   midband_phase_deg,
    }


def _compute_tran_metrics(rows: List[List[float]]) -> Optional[Dict]:
    """Peak-to-peak VOUT from transient table.  Columns: [time, v(vout)]"""
    if not rows or len(rows) < 2:
        return None
    vout = [r[1] for r in rows if len(r) >= 2]
    if not vout:
        return None
    pp = max(vout) - min(vout)
    return {"pp_norm": pp / VDD_V, "pp_abs": pp}


# ── 7. Validity checks ────────────────────────────────────────────────────────
def check_op_validity(
    voltages: Dict[str, float], ports: List[str]
) -> Tuple[bool, List[str]]:
    notes: List[str] = []
    ok = True

    for node, v in voltages.items():
        if v < -0.1 or v > VDD_V + 0.1:
            notes.append(f"V({node})={v:.4f} outside supply range")
            ok = False

    for p in ports:
        if _is_output(p):
            sp = _spice_net(p).upper()
            v = voltages.get(sp) if sp != p else voltages.get(p)
            if v is None:
                notes.append(f"V({p}) missing from OP results")
                ok = False
            elif abs(v) < 0.01:
                notes.append(f"V({p})={v:.4f} at VSS rail")
            elif abs(v - VDD_V) < 0.01:
                notes.append(f"V({p})={v:.4f} at VDD rail")

    return ok, notes


def check_functional(
    circuit_type: str,
    op_valid: bool,
    dc: Optional[Dict],
    ac: Optional[Dict],
    tran: Optional[Dict],
) -> Tuple[bool, List[str]]:
    notes: List[str] = []
    valid = True

    if not op_valid:
        notes.append("OP not physically valid")
        return False, notes

    if circuit_type in ("no_output", "static_bias"):
        return op_valid, notes

    if dc is None:
        notes.append("DC analysis unavailable")
        valid = False
    else:
        gain  = dc["gain"]
        swing = dc["swing"]

        if circuit_type == "inverting_amplifier":
            if not dc["inverting"]:
                notes.append(f"Expected inverting gain; got +{gain:.2f}")
                valid = False
            if gain < 1.0:
                notes.append(f"Gain {gain:.2f} < 1 — no amplification")
                valid = False
            if swing < 0.1 * VDD_V:
                notes.append(f"Output swing {swing:.3f} V < 10 % VDD")
                valid = False

        elif circuit_type in ("source_follower", "non_inv_amplifier"):
            if dc["inverting"]:
                notes.append(f"Expected non-inverting; got inverting gain -{gain:.2f}")
                valid = False
            if circuit_type == "source_follower" and not (0.3 <= gain <= 1.5):
                notes.append(f"Follower gain {gain:.2f} outside [0.3, 1.5]")
                valid = False

        elif circuit_type == "weak_inverter":
            if not dc["inverting"]:
                notes.append(f"Expected inverting gain; got +{gain:.2f}")
                valid = False
            if swing < 0.05 * VDD_V:
                notes.append(f"Output swing {swing:.3f} V — circuit barely active")
                valid = False

    if ac is not None and circuit_type == "inverting_amplifier":
        ph = ac["phase_deg"]
        if abs(abs(ph) - 180) > 45:
            notes.append(f"AC phase {ph:.1f}° unexpected (expected ≈180°)")

    if tran is not None and tran["pp_norm"] < 0.01:
        notes.append(
            f"Transient output flat ({tran['pp_norm']*100:.1f}% pp/VDD) — no signal response"
        )

    return valid, notes


# ── 8. Main pipeline ──────────────────────────────────────────────────────────
def validate(
    tc_id: str,
    mode: str = "ideal",
    tc_base: Optional[Path] = None,
    sim_base: Optional[Path] = None,
) -> TCResult:
    if tc_base is None:
        tc_base = TESTCASE_DIR
    if sim_base is None:
        sim_base = SIM_DIR

    result = TCResult(tc_id=tc_id, mode=mode)
    tc_dir    = tc_base / str(tc_id)
    cir_file  = tc_dir / f"{tc_id}.cir"
    port_file = tc_dir / f"Port{tc_id}.txt"

    # Stage 1 — Syntax
    components, errors = parse_cir(cir_file)
    result.syntax_errors = errors
    result.syntax_ok = not errors and bool(components)
    if not result.syntax_ok:
        return result

    ports = read_ports(port_file)

    # Stage 2 — Connectivity
    issues = check_connectivity(components, ports)
    result.connectivity_issues = issues
    result.connectivity_ok = not issues

    has_vin  = any(_is_vin(p)    for p in ports)
    has_vout = any(_is_output(p) for p in ports)

    # Stage 3 — Simulation
    workdir = sim_base / f"tc{tc_id}"
    workdir.mkdir(parents=True, exist_ok=True)

    if mode == "sky130":
        spice_file = generate_spice_sky130(tc_id, components, ports, workdir)
    else:
        spice_file = generate_spice_ideal_full(tc_id, components, ports, workdir)

    sim_ok, log = run_ngspice(spice_file)
    result.sim_ok = sim_ok
    if not sim_ok:
        tail = log.strip().splitlines()
        result.sim_error = "\n".join(tail[-6:]) if tail else log

    (workdir / f"tc{tc_id}_full_{mode}.log").write_text(log)

    # ── 3a. OP ──────────────────────────────────────────────────────────────
    voltages = parse_op_voltages(log)
    result.sim_parse_ok = voltages is not None
    if voltages:
        result.op_voltages = voltages
        valid, notes = check_op_validity(voltages, ports)
        result.op_valid = valid
        result.op_notes = notes

    # ── 3b–3d. DC / AC / TRAN (ideal mode + has signal path) ───────────────
    dc_m = ac_m = tran_m = None

    if mode == "ideal" and has_vin and has_vout and result.sim_ok:
        dc_rows = _parse_table(log, "DC_START", "DC_END")
        if dc_rows:
            result.dc_ok = True
            dc_m = _compute_dc_metrics(dc_rows)
            if dc_m:
                result.dc_gain       = dc_m["gain"]
                result.dc_inverting  = dc_m["inverting"]
                result.dc_swing_low  = dc_m["vout_min"]
                result.dc_swing_high = dc_m["vout_max"]
        else:
            result.dc_error = "DC table not found in log"

        ac_rows = _parse_table(log, "AC_START", "AC_END")
        if ac_rows:
            result.ac_ok = True
            ac_m = _compute_ac_metrics(ac_rows)
            if ac_m:
                result.ac_midband_db = ac_m["midband_db"]
                result.ac_bw_hz      = ac_m["bw_hz"]
                result.ac_phase_deg  = ac_m["phase_deg"]
        else:
            result.ac_error = "AC table not found in log"

        tran_rows = _parse_table(log, "TRAN_START", "TRAN_END")
        if tran_rows:
            result.tran_ok = True
            tran_m = _compute_tran_metrics(tran_rows)
            if tran_m:
                result.tran_pp_norm = tran_m["pp_norm"]
        else:
            result.tran_error = "TRAN table not found in log"

        result.circuit_type = classify_from_measurements(
            result.dc_gain, result.dc_inverting, ports
        )
    else:
        result.circuit_type = classify_circuit_topology(components, ports)

    # ── Functional validation ────────────────────────────────────────────────
    fv, fn = check_functional(
        result.circuit_type, result.op_valid, dc_m, ac_m, tran_m
    )
    result.functional_valid = fv
    result.functional_notes = fn

    return result


# ── 9. Reporting ──────────────────────────────────────────────────────────────
def print_report(results: List[TCResult]) -> None:
    if not results:
        return
    mode = results[0].mode
    n    = len(results)

    syntax_ok  = sum(r.syntax_ok       for r in results)
    struct_ok  = sum(r.syntax_ok and r.connectivity_ok for r in results)
    sim_ok     = sum(r.sim_ok          for r in results)
    parse_ok   = sum(r.sim_parse_ok    for r in results)
    op_valid   = sum(r.op_valid        for r in results)
    dc_ok      = sum(r.dc_ok           for r in results)
    ac_ok      = sum(r.ac_ok           for r in results)
    tran_ok    = sum(r.tran_ok         for r in results)
    func_ok    = sum(r.functional_valid for r in results)

    SEP = "─" * 68

    print(f"\n{'═'*68}")
    print(f"  MODE: {mode.upper()}")
    print(f"{'═'*68}")

    for r in results:
        print(f"\n{SEP}")
        print(f"  Testcase {r.tc_id}  [{r.circuit_type}]")
        print(SEP)

        print(f"  [{'✓' if r.syntax_ok else '✗'}] Syntax")
        for e in r.syntax_errors:
            print(f"        {e}")

        print(f"  [{'✓' if r.connectivity_ok else '✗'}] Connectivity")
        for i in r.connectivity_issues:
            print(f"        {i}")

        print(f"  [{'✓' if r.sim_ok else '✗'}] Simulation (ngspice)")
        if not r.sim_ok and r.sim_error:
            for line in r.sim_error.splitlines():
                print(f"        {line}")

        if r.sim_ok:
            print(f"  [{'✓' if r.sim_parse_ok else '✗'}] OP parsed")

        if r.op_voltages:
            print(f"  [{'✓' if r.op_valid else '!'}] OP physical validity")
            for note in r.op_notes:
                print(f"        {note}")
            vout = {k: v for k, v in r.op_voltages.items() if _is_output(k)}
            if vout:
                print("        " + ", ".join(f"V({k})={v:.4f}" for k, v in vout.items()))

        if r.dc_ok:
            inv = "inv" if r.dc_inverting else "non-inv"
            swing = r.dc_swing_high - r.dc_swing_low
            print(f"  [✓] DC sweep  gain={r.dc_gain:.2f} ({inv})  "
                  f"swing=[{r.dc_swing_low:.3f},{r.dc_swing_high:.3f}]V  Δ={swing:.3f}V")
        elif r.dc_error:
            print(f"  [✗] DC sweep  {r.dc_error}")

        if r.ac_ok:
            bw_str = f"{r.ac_bw_hz:.3g} Hz" if r.ac_bw_hz < 1e9 else ">1 GHz"
            print(f"  [✓] AC        midband={r.ac_midband_db:.1f} dB  "
                  f"phase={r.ac_phase_deg:.0f}°  BW={bw_str}")
        elif r.ac_error:
            print(f"  [✗] AC        {r.ac_error}")

        if r.tran_ok:
            print(f"  [✓] Transient  pp={r.tran_pp_norm*100:.1f}% VDD")
        elif r.tran_error:
            print(f"  [✗] Transient  {r.tran_error}")

        print(f"  [{'✓' if r.functional_valid else '✗'}] Functional validity")
        for fn in r.functional_notes:
            print(f"        {fn}")

    print(f"\n{'═'*68}")
    print(f"  SUMMARY  [{mode.upper()}]")
    print(f"{'═'*68}")
    print(f"  Total testcases          : {n}")
    print(f"  Syntax valid             : {syntax_ok}/{n}  ({100*syntax_ok//n}%)")
    print(f"  Structural valid         : {struct_ok}/{n}  ({100*struct_ok//n}%)")
    print(f"  Simulation success       : {sim_ok}/{n}  ({100*sim_ok//n}%)")
    print(f"  OP parse success         : {parse_ok}/{n}  ({100*parse_ok//n}%)")
    print(f"  OP physically valid      : {op_valid}/{n}  ({100*op_valid//n}%)")
    if dc_ok:
        print(f"  DC sweep success         : {dc_ok}/{n}  ({100*dc_ok//n}%)")
    if ac_ok:
        print(f"  AC analysis success      : {ac_ok}/{n}  ({100*ac_ok//n}%)")
    if tran_ok:
        print(f"  Transient success        : {tran_ok}/{n}  ({100*tran_ok//n}%)")
    print(f"  Functional validity      : {func_ok}/{n}  ({100*func_ok//n}%)")

    # Failure breakdown
    fail: Dict[str, int] = {
        "syntax": 0, "connectivity": 0, "convergence": 0,
        "parse": 0, "functional": 0,
    }
    for r in results:
        if not r.syntax_ok:
            fail["syntax"] += 1
        if not r.connectivity_ok:
            fail["connectivity"] += 1
        if r.sim_ok and not r.sim_parse_ok:
            fail["parse"] += 1
        if r.syntax_ok and r.sim_ok and not r.sim_parse_ok:
            fail["convergence"] += 1
        if not r.functional_valid:
            fail["functional"] += 1

    if any(fail.values()):
        print("  Failure breakdown:")
        for k, v in fail.items():
            if v:
                print(f"    {k:<20}: {v}")

    # Circuit type breakdown
    types: Dict[str, int] = {}
    for r in results:
        types[r.circuit_type] = types.get(r.circuit_type, 0) + 1
    print("  Circuit type breakdown:")
    for ct, cnt in sorted(types.items()):
        print(f"    {ct:<24}: {cnt}")
    print()


def save_json(results: List[TCResult], mode: str) -> Path:
    data = []
    for r in results:
        data.append({
            "testcase_id": r.tc_id,
            "mode": r.mode,
            "circuit_type": r.circuit_type,
            # Stage 1
            "syntax_ok": r.syntax_ok,
            "syntax_errors": r.syntax_errors,
            # Stage 2
            "connectivity_ok": r.connectivity_ok,
            "connectivity_issues": r.connectivity_issues,
            # Stage 3a
            "sim_ok": r.sim_ok,
            "sim_error": r.sim_error,
            "sim_parse_ok": r.sim_parse_ok,
            "op_voltages": r.op_voltages,
            "op_valid": r.op_valid,
            "op_notes": r.op_notes,
            # Stage 3b
            "dc_ok": r.dc_ok,
            "dc_gain": r.dc_gain,
            "dc_inverting": r.dc_inverting,
            "dc_swing_low": r.dc_swing_low,
            "dc_swing_high": r.dc_swing_high,
            "dc_error": r.dc_error,
            # Stage 3c
            "ac_ok": r.ac_ok,
            "ac_midband_db": r.ac_midband_db,
            "ac_bw_hz": r.ac_bw_hz,
            "ac_phase_deg": r.ac_phase_deg,
            "ac_error": r.ac_error,
            # Stage 3d
            "tran_ok": r.tran_ok,
            "tran_pp_norm": r.tran_pp_norm,
            "tran_error": r.tran_error,
            # Functional
            "functional_valid": r.functional_valid,
            "functional_notes": r.functional_notes,
        })
    out = SIM_DIR / f"validation_report_{mode}.json"
    out.write_text(json.dumps(data, indent=2))
    return out


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    modes = sys.argv[1:] if len(sys.argv) > 1 else ["ideal"]

    SIM_DIR.mkdir(parents=True, exist_ok=True)

    for mode in modes:
        if mode not in ("ideal", "sky130"):
            print(f"Unknown mode '{mode}'. Use 'ideal' or 'sky130'.")
            continue

        print(f"\n[{mode.upper()}] Running validation pipeline ...")
        results: List[TCResult] = []
        for tc_id in TESTCASE_IDS:
            print(f"  → tc{tc_id} ...", end=" ", flush=True)
            r = validate(str(tc_id), mode)
            results.append(r)
            flags = []
            if r.syntax_ok:         flags.append("syn")
            if r.connectivity_ok:   flags.append("conn")
            if r.sim_ok:            flags.append("sim")
            if r.op_valid:          flags.append("op")
            if r.dc_ok:             flags.append("dc")
            if r.ac_ok:             flags.append("ac")
            if r.tran_ok:           flags.append("tran")
            if r.functional_valid:  flags.append("func")
            print(", ".join(flags) if flags else "FAILED")

        print_report(results)
        report_path = save_json(results, mode)
        print(f"  JSON → {report_path}")
