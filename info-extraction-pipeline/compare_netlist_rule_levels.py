#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx

from netlist_parser import Component, is_port_like_net, parse_netlist_file


INDEX_RE = re.compile(r"(\d+)$")
TRANSISTOR_TYPES = {"nmos", "pmos", "nmos4", "pmos4", "npn", "pnp"}
_MOSFET_TYPES = {"nmos", "pmos", "nmos4", "pmos4"}

# When True, MOSFET bulk (4th pin) is excluded from graph isomorphism checks.
# Set to False to restore strict bulk-pin matching.
IGNORE_MOSFET_BULK = True
SUPPLY_NETS = {"VDD", "AVDD", "DVDD", "GND", "GNDA", "VGND", "VSS", "AVSS", "DVSS", "0"}
_VDD_NETS = {"VDD", "AVDD", "DVDD"}
_GND_NETS = {"GND", "GNDA", "VGND", "VSS", "AVSS", "DVSS", "0"}


def _canonical_supply(token: str) -> str:
    if token in _VDD_NETS:
        return "VDD"
    if token in _GND_NETS:
        return "GND"
    return token


RULES = [
    {
        "level": 1,
        "name": "rule1_topology_only",
        "description": "Only topology; ignore all net names and all component instance indexes.",
        "net_mode": "none",
        "component_index_mode": "none",
    },
    {
        "level": 2,
        "name": "rule1_2_supply_nets",
        "description": "Topology plus VDD/GND/VSS-like supply and ground net labels.",
        "net_mode": "supply",
        "component_index_mode": "none",
    },
    {
        "level": 3,
        "name": "rule1_3_special_nets",
        "description": "Topology plus all special/port-like net labels such as VIN, VOUT, bias, clock, supply, ground.",
        "net_mode": "special",
        "component_index_mode": "none",
    },
    {
        "level": 4,
        "name": "rule1_4_transistor_indexes",
        "description": "Rule 3 plus MOSFET/BJT instance indexes.",
        "net_mode": "special",
        "component_index_mode": "transistors",
    },
    {
        "level": 5,
        "name": "rule1_5_all_component_indexes",
        "description": "Rule 4 plus all component instance indexes.",
        "net_mode": "special",
        "component_index_mode": "all",
    },
]


def component_index(inst: str) -> str:
    match = INDEX_RE.search(inst)
    return match.group(1) if match else inst


def net_label(net: str, mode: str) -> str:
    token = net.upper()
    if mode == "none":
        return ""
    if mode == "supply":
        return _canonical_supply(token) if token in SUPPLY_NETS else ""
    if mode == "special":
        if token in SUPPLY_NETS:
            return _canonical_supply(token)
        return token if is_port_like_net(token) else ""
    raise ValueError(f"unknown net label mode: {mode}")


def should_label_component_index(component: Component, mode: str) -> bool:
    if mode == "none":
        return False
    if mode == "all":
        return True
    if mode == "transistors":
        return component.ctype in TRANSISTOR_TYPES
    raise ValueError(f"unknown component index mode: {mode}")


def component_label(component: Component, index_mode: str) -> str:
    label = f"component:{component.ctype}"
    if should_label_component_index(component, index_mode):
        label += f":idx:{component_index(component.inst)}"
    return label


def graph_for_rule(components: list[Component], rule: dict[str, str]) -> nx.Graph:
    graph = nx.Graph()
    for comp_position, component in enumerate(components):
        comp_node = f"comp:{comp_position}"
        graph.add_node(
            comp_node,
            kind="component",
            label=component_label(component, rule["component_index_mode"]),
        )
        pins = component.nodes
        if IGNORE_MOSFET_BULK and component.ctype in _MOSFET_TYPES and len(pins) == 4:
            pins = pins[:3]  # skip bulk when flag is set
        for pin_index, net in enumerate(pins):
            pin_node = f"pin:{comp_position}:{pin_index}"
            graph.add_node(pin_node, kind="pin", label=f"pin:{pin_index}")
            graph.add_edge(comp_node, pin_node)

            net_node = f"net:{net}"
            if not graph.has_node(net_node):
                graph.add_node(net_node, kind="net", label=f"net:{net_label(net, rule['net_mode'])}")
            graph.add_edge(pin_node, net_node)
    return graph


def graphs_match(left: list[Component], right: list[Component], rule: dict[str, str]) -> bool:
    graph_left = graph_for_rule(left, rule)
    graph_right = graph_for_rule(right, rule)

    def node_match(a: dict[str, Any], b: dict[str, Any]) -> bool:
        return a["kind"] == b["kind"] and a["label"] == b["label"]

    return nx.algorithms.isomorphism.GraphMatcher(graph_left, graph_right, node_match=node_match).is_isomorphic()


def component_counts(components: list[Component]) -> dict[str, int]:
    return dict(Counter(component.ctype for component in components))


def compare_case(generated_path: Path, ground_truth_path: Path) -> dict[str, Any]:
    generated = parse_netlist_file(generated_path)
    ground_truth = parse_netlist_file(ground_truth_path)
    rule_results = {}
    for rule in RULES:
        rule_results[rule["name"]] = graphs_match(generated, ground_truth, rule)

    return {
        "id": generated_path.stem,
        "generated_path": str(generated_path),
        "ground_truth_path": str(ground_truth_path),
        "generated_component_count": len(generated),
        "ground_truth_component_count": len(ground_truth),
        "generated_component_type_counts": component_counts(generated),
        "ground_truth_component_type_counts": component_counts(ground_truth),
        "rules": rule_results,
    }


def discover_pairs(generated_dir: Path, ground_truth_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for generated_path in sorted(generated_dir.glob("*.cir"), key=lambda p: (len(p.stem), p.stem)):
        ground_truth_path = ground_truth_dir / generated_path.name
        if ground_truth_path.exists():
            pairs.append((generated_path, ground_truth_path))
    return pairs


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare generated and ground-truth netlists under cumulative rule levels.")
    parser.add_argument("--generated-dir", type=Path, default=Path("Data/netlist_generated"))
    parser.add_argument("--ground-truth-dir", type=Path, default=Path("Data/netlist_ground_truth"))
    parser.add_argument("--json-out", type=Path, default=Path("simulation_general/netlist_rule_level_comparison.json"))
    parser.add_argument("--show-failures", type=int, default=12)
    args = parser.parse_args()

    generated_dir = args.generated_dir.resolve()
    ground_truth_dir = args.ground_truth_dir.resolve()
    pairs = discover_pairs(generated_dir, ground_truth_dir)
    rows = [compare_case(generated_path, ground_truth_path) for generated_path, ground_truth_path in pairs]

    total = len(rows)
    summary = {
        "generated_dir": str(generated_dir),
        "ground_truth_dir": str(ground_truth_dir),
        "total_pairs": total,
        "rules": RULES,
        "pass_counts": {},
        "fail_counts": {},
    }
    for rule in RULES:
        name = rule["name"]
        passed = sum(1 for row in rows if row["rules"][name])
        summary["pass_counts"][name] = passed
        summary["fail_counts"][name] = total - passed

    payload = {"summary": summary, "cases": rows}
    write_json(args.json_out.resolve(), payload)

    print(f"Generated dir   : {generated_dir}")
    print(f"Ground-truth dir: {ground_truth_dir}")
    print(f"Pairs compared  : {total}")
    print()
    print(f"{'Level':<6} {'Rule':<34} {'Pass':>6} {'Fail':>6} {'Pass %':>8}")
    print("-" * 62)
    for rule in RULES:
        name = rule["name"]
        passed = summary["pass_counts"][name]
        failed = summary["fail_counts"][name]
        pct = (100 * passed / total) if total else 0
        print(f"{rule['level']:<6} {name:<34} {passed:>6} {failed:>6} {pct:>7.1f}%")

    if args.show_failures:
        print()
        for rule in RULES:
            name = rule["name"]
            failures = [row["id"] for row in rows if not row["rules"][name]]
            shown = failures[: args.show_failures]
            suffix = f" ... +{len(failures) - len(shown)}" if len(failures) > len(shown) else ""
            print(f"{name} failures: {', '.join(shown) if shown else 'none'}{suffix}")

    print()
    print(f"JSON -> {args.json_out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
