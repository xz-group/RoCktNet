from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx


PAREN_LINE_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*\(([^()]*)\)\s*([A-Za-z][A-Za-z0-9_\-]*)(?:\s+(.*))?\s*$"
)
ANON_NET_RE = re.compile(r"^net[0-9A-Za-z_]*$", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")
PORT_EXACT_NETS = {
    "0",
    "GND",
    "GNDA",
    "VGND",
    "VSS",
    "AVSS",
    "DVSS",
    "VDD",
    "AVDD",
    "DVDD",
}
PORT_PREFIXES = (
    "VIN",
    "VOUT",
    "IN",
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
)

TYPE_ALIASES = {
    "pmos": "pmos",
    "pmos4": "pmos4",
    "nmos": "nmos",
    "nmos4": "nmos4",
    "resistor": "resistor",
    "res": "resistor",
    "capacitor": "capacitor",
    "cap": "capacitor",
    "inductor": "inductor",
    "ind": "inductor",
    "diode": "diode",
    "pnp": "pnp",
    "npn": "npn",
    "currentsource": "current_source",
    "current_source": "current_source",
    "current-source": "current_source",
    "isource": "current_source",
    "current": "current_source",
    "vsource": "voltage_source",
    "voltage_source": "voltage_source",
    "voltage-source": "voltage_source",
    "voltage": "voltage_source",
    "amplifier": "amplifier",
    "amp": "amplifier",
    "opamp": "amplifier",
    "op_amp": "amplifier",
    "op": "amplifier",
    "switch_ideal": "switch_ideal",
    "switch": "switch_ideal",
    "transmission_gate": "transmission_gate",
    "tgate": "transmission_gate",
    "short": "short",
    "inverter": "inverter",
    "inv": "inverter",
    "current_source": "current_source",
    "voltage_source": "voltage_source",
    # single-letter SPICE aliases
    "r": "resistor",
    "c": "capacitor",
    "l": "inductor",
    "i": "current_source",
    "v": "voltage_source",
    "d": "diode",
}

PREFIX_TYPES = {
    "R": "resistor",
    "C": "capacitor",
    "L": "inductor",
    "I": "current_source",
    "V": "voltage_source",
    "D": "diode",
}


@dataclass(frozen=True)
class Component:
    inst: str
    nodes: tuple[str, ...]
    ctype: str
    params: tuple[str, ...] = ()

    @property
    def prefix(self) -> str:
        return self.inst[0].upper() if self.inst else ""


def canonical_component_type(raw: str) -> str:
    lowered = raw.strip().lower()
    return TYPE_ALIASES.get(lowered, lowered)


def canonical_graph_token(text: str) -> str:
    token = text.strip()
    token = token.replace("–", "-").replace("—", "-").replace("−", "-")
    token = token.replace(" ", "")
    return token.upper()


def is_anonymous_net(net: str) -> bool:
    return bool(ANON_NET_RE.fullmatch(net.strip()))


def is_port_like_net(net: str) -> bool:
    token = canonical_graph_token(net)
    return token in PORT_EXACT_NETS or token.startswith(PORT_PREFIXES)


def topology_net_label(net: str) -> str:
    token = canonical_graph_token(net)
    return token if is_port_like_net(token) else ""


def strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned
    lines = cleaned.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_netlist_text(text: str) -> list[Component]:
    components: list[Component] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = strip_inline_comment(raw).strip()
        if not line:
            continue
        try:
            components.append(parse_netlist_line(line, lineno=lineno))
        except ValueError:
            # Try stripping trailing SPICE size/model parameters (e.g. "8u 2u")
            # by removing tokens after the known component type token.
            parts = SPACE_RE.split(line.strip())
            parsed = False
            for end in range(len(parts), 1, -1):
                try:
                    components.append(parse_netlist_line(" ".join(parts[:end]), lineno=lineno))
                    parsed = True
                    break
                except ValueError:
                    continue
            # If still can't parse, skip the line silently
    if not components:
        raise ValueError("empty netlist")
    return components


def parse_netlist_file(path: Path) -> list[Component]:
    return parse_netlist_text(path.read_text())


def parse_netlist_line(line: str, *, lineno: int | None = None) -> Component:
    location = f"line {lineno}: " if lineno is not None else ""
    paren_component = parse_parenthesized_line(line)
    if paren_component:
        return paren_component

    parts = SPACE_RE.split(line.strip())
    if not parts or not parts[0]:
        raise ValueError(f"{location}invalid netlist syntax: {line!r}")

    inst = parts[0]
    prefix = inst[0].upper()
    # X prefix: subckt instantiation — treat as generic multi-pin component
    if prefix == "X":
        ctype = canonical_component_type(parts[-1])
        if ctype in TYPE_ALIASES.values() and len(parts) >= 3:
            nodes = tuple(parts[1:-1])
            return make_component(parts[0], nodes, ctype)
        raise ValueError(f"{location}unknown subckt type: {line!r}")
    if prefix == "M":
        last = canonical_component_type(parts[-1])
        if last in {"npn", "pnp"}:
            return parse_fixed_arity_line(
                parts,
                node_count=3,
                allowed_types={"pnp", "npn"},
                fallback_type=None,
                location=location,
                kind="BJT",
                line=line,
            )
        if last in {"pmos", "nmos", "pmos4", "nmos4"}:
            return parse_fixed_arity_line(
                parts,
                node_count=4,
                allowed_types={"pmos", "nmos", "pmos4", "nmos4"},
                fallback_type=None,
                location=location,
                kind="MOS",
                line=line,
            )
        # M prefix with non-MOS type → fall through to generic handler
        ctype = last
        if ctype in TYPE_ALIASES.values() and len(parts) >= 3:
            nodes = tuple(parts[1:-1])
            return make_component(parts[0], nodes, ctype)
        raise ValueError(f"{location}invalid netlist syntax: {line!r}")
    if prefix == "Q":
        return parse_fixed_arity_line(
            parts,
            node_count=3,
            allowed_types={"pnp", "npn"},
            fallback_type=None,
            location=location,
            kind="BJT",
            line=line,
        )
    if prefix in PREFIX_TYPES:
        last = canonical_component_type(parts[-1])
        # If last token is an explicit multi-pin type, use generic handler
        if last in {"transmission_gate", "inverter", "amplifier", "switch_ideal", "short"}:
            pass  # fall through to generic handler
        else:
            return parse_fixed_arity_line(
                parts,
                node_count=2,
                allowed_types=None,
                fallback_type=PREFIX_TYPES[prefix],
                location=location,
                kind=prefix,
                line=line,
            )
    # Amplifier / generic multi-pin component: last token is the type, rest are nodes
    ctype = canonical_component_type(parts[-1])
    if ctype in TYPE_ALIASES.values() and len(parts) >= 3:
        nodes = tuple(parts[1:-1])
        return make_component(parts[0], nodes, ctype)
    raise ValueError(f"{location}invalid netlist syntax: {line!r}")


def strip_inline_comment(raw: str) -> str:
    line = raw.strip()
    if not line or line.startswith(("*", "#", "//")):
        return ""
    for marker in ("//", "#"):
        before, sep, _after = line.partition(marker)
        if sep:
            line = before.rstrip()
    return line


def parse_parenthesized_line(line: str) -> Component | None:
    match = PAREN_LINE_RE.match(line)
    if not match:
        return None
    inst, nodes_raw, ctype_raw, params_raw = match.groups()
    nodes = tuple(tok for tok in SPACE_RE.split(nodes_raw.strip()) if tok)
    if len(nodes) < 2:
        raise ValueError(f"expected at least two nodes: {line!r}")
    params = tuple(tok for tok in SPACE_RE.split((params_raw or "").strip()) if tok)
    return make_component(inst, nodes, canonical_component_type(ctype_raw), params=params)


def parse_fixed_arity_line(
    parts: list[str],
    *,
    node_count: int,
    allowed_types: set[str] | None,
    fallback_type: str | None,
    location: str,
    kind: str,
    line: str,
) -> Component:
    expected_without_type = 1 + node_count
    expected_with_type = expected_without_type + 1

    if fallback_type is None:
        if len(parts) < expected_with_type:
            raise ValueError(f"{location}{kind} line must have {node_count} nodes followed by a type: {line!r}")
        ctype = canonical_component_type(parts[expected_without_type])
        params = parts[expected_with_type:]
    elif len(parts) == expected_without_type:
        ctype = fallback_type
        params = []
    elif len(parts) > expected_without_type:
        maybe_type = canonical_component_type(parts[expected_without_type])
        if maybe_type == fallback_type:
            ctype = maybe_type
            params = parts[expected_with_type:]
        else:
            ctype = fallback_type
            params = parts[expected_without_type:]
    else:
        raise ValueError(f"{location}{kind} line must have {node_count} nodes: {line!r}")

    if allowed_types is not None and ctype not in allowed_types:
        raise ValueError(f"{location}{kind} type must be one of {sorted(allowed_types)}: {line!r}")

    return make_component(parts[0], parts[1 : 1 + node_count], ctype, params=params)


def make_component(
    inst: str,
    nodes: tuple[str, ...] | list[str],
    ctype: str,
    *,
    params: tuple[str, ...] | list[str] = (),
) -> Component:
    return Component(
        inst=canonical_graph_token(inst),
        nodes=tuple(canonical_graph_token(tok) for tok in nodes),
        ctype=canonical_component_type(ctype),
        params=tuple(str(tok).strip() for tok in params if str(tok).strip()),
    )


def component_to_netlist_line(component: Component) -> str:
    suffix = f" {' '.join(component.params)}" if component.params else ""
    if component.prefix in {"M", "Q"}:
        return f"{component.inst} {' '.join(component.nodes)} {component.ctype}{suffix}"
    if component.prefix in PREFIX_TYPES and len(component.nodes) == 2:
        return f"{component.inst} {' '.join(component.nodes)}{suffix}"
    return f"{component.inst} ({' '.join(component.nodes)}) {component.ctype}{suffix}"


def normalize_netlist_text(text: str) -> str:
    return "\n".join(component_to_netlist_line(component) for component in parse_netlist_text(text))


def normalize_candidate_text(text: str) -> str:
    valid_lines: list[str] = []
    in_reason = False
    for raw in strip_code_fence(text).splitlines():
        line = strip_inline_comment(raw).strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith("reason:"):
            in_reason = True
            continue
        if lowered.startswith("netlist:"):
            in_reason = False
            continue
        if in_reason:
            try:
                parse_netlist_line(line)
            except ValueError:
                continue
            in_reason = False
        try:
            component = parse_netlist_line(line)
        except ValueError:
            continue
        valid_lines.append(component_to_netlist_line(component))
    return "\n".join(valid_lines)


def extract_candidate_reason(text: str) -> str:
    lines = strip_code_fence(text).splitlines()
    reason_lines: list[str] = []
    collecting = False
    for raw in lines:
        line = raw.strip()
        if not line:
            if collecting:
                break
            continue
        lowered = line.lower()
        if lowered.startswith("netlist:"):
            break
        if lowered.startswith("reason:"):
            collecting = True
            first = line.split(":", 1)[1].strip()
            if first:
                reason_lines.append(first)
            continue
        if collecting:
            try:
                parse_netlist_line(line)
                break
            except ValueError:
                reason_lines.append(line)
    return " ".join(reason_lines).strip()


def graph_from_components(components: list[Component], *, ignore_net_labels: bool = False) -> nx.Graph:
    graph = nx.Graph()
    for index, component in enumerate(components):
        comp_node = f"comp:{index}"
        graph.add_node(comp_node, kind="component", ctype=component.ctype)
        for pin_index, net in enumerate(component.nodes):
            pin_node = f"pin:{index}:{pin_index}"
            graph.add_node(pin_node, kind="pin", pin=pin_index)
            graph.add_edge(comp_node, pin_node)
            net_node = f"net:{net}"
            if not graph.has_node(net_node):
                label = topology_net_label(net) if ignore_net_labels else net
                graph.add_node(net_node, kind="net", label=label)
            graph.add_edge(pin_node, net_node)
    return graph


def graph_fingerprint_from_components(components: list[Component], *, ignore_net_labels: bool = False) -> str:
    graph = graph_from_components(components, ignore_net_labels=ignore_net_labels)
    hashed = nx.Graph()
    for node, attrs in graph.nodes(data=True):
        if attrs["kind"] == "component":
            wl_label = f"component:{attrs['ctype']}"
        elif attrs["kind"] == "pin":
            wl_label = f"pin:{attrs['pin']}"
        else:
            wl_label = f"net:{attrs['label']}"
        hashed.add_node(node, wl_label=wl_label)
    hashed.add_edges_from(graph.edges())
    return nx.weisfeiler_lehman_graph_hash(hashed, node_attr="wl_label")


def topology_fingerprint_from_components(components: list[Component]) -> str:
    return graph_fingerprint_from_components(components, ignore_net_labels=True)


def graph_isomorphic(
    components_a: list[Component],
    components_b: list[Component],
    *,
    ignore_net_labels: bool = False,
) -> bool:
    graph_a = graph_from_components(components_a, ignore_net_labels=ignore_net_labels)
    graph_b = graph_from_components(components_b, ignore_net_labels=ignore_net_labels)

    def node_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
        if left["kind"] != right["kind"]:
            return False
        if left["kind"] == "component":
            return left["ctype"] == right["ctype"]
        if left["kind"] == "pin":
            return left["pin"] == right["pin"]
        return left["label"] == right["label"]

    matcher = nx.algorithms.isomorphism.GraphMatcher(graph_a, graph_b, node_match=node_match)
    return matcher.is_isomorphic()
