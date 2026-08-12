"""
Level-1 graph-level netlist comparison using NetworkX.

Each netlist is converted to a topology graph with three node kinds:
  - component nodes, labeled by device type (nmos, pmos, npn, resistor, ...)
  - pin nodes, labeled only by pin position
  - net nodes, with names ignored during matching

This matches the level-1 rule in compare_netlist_rule_levels.py: compare only
topology and component type, while ignoring all net names and component instance
indexes. MOSFET bulk pins are ignored, matching that script's default behavior.
For non-isomorphic pairs, NetworkX graph_edit_distance is used to produce a
topology distance.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
from networkx.algorithms import isomorphism as iso

IGNORE_MOSFET_BULK = True
MOSFET_TYPES = {"nmos", "pmos"}
NONPOLAR_TWO_TERMINAL_TYPES = {"resistor", "capacitor", "inductor"}
SUPPLY_NETS = {"VDD", "GND"}

AMP_ORDERS = {
    "Siso_amp": ["In", "Out"],
    "Diso_amp": ["InN", "InP", "Out"],
    "Dido_amp": ["InN", "InP", "OutN", "OutP"],
}


@dataclass(frozen=True)
class Device:
    name: str
    device_type: str
    pins: list[tuple[str, str]]


def parse_netlist(path: Path) -> list[Device]:
    if path.suffix.lower() == ".json":
        return parse_json_netlist(path)
    return parse_spice_netlist(path)


def parse_spice_netlist(path: Path) -> list[Device]:
    devices: list[Device] = []

    for line_no, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if (
            not line
            or line.startswith("*")
            or line.startswith("#")
            or line.startswith(".")
        ):
            continue

        parts = line.split()
        name = parts[0]
        prefix = name[0].upper()

        try:
            device = parse_device(name, prefix, parts[1:])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: {exc}: {line}") from exc
        devices.append(device)

    return devices


def two_terminal_source_ports(ports: dict) -> tuple[str, str]:
    """Return (positive, negative) nets for a 2-terminal source, accepting the
    several key namings seen in GT files (Positive/Negative, In/Out, Pos/Neg)."""
    for pos_key, neg_key in (("Positive", "Negative"), ("In", "Out"), ("Pos", "Neg")):
        if pos_key in ports and neg_key in ports:
            return ports[pos_key], ports[neg_key]
    raise ValueError(f"source ports not recognized: {sorted(ports)}")


def parse_json_netlist(path: Path) -> list[Device]:
    data = json.loads(path.read_text(encoding="utf-8"))
    devices: list[Device] = []
    counters: dict[str, int] = {}

    for comp in data.get("ckt_netlist", []):
        ctype = comp["component_type"]
        ports = comp.get("port_connection", {})
        counters[ctype] = counters.get(ctype, 0) + 1
        name = f"{ctype}{counters[ctype]}"

        if ctype == "NMOS":
            devices.append(
                Device(
                    name,
                    "nmos",
                    [
                        ("D", ports["Drain"]),
                        ("G", ports["Gate"]),
                        ("S", ports["Source"]),
                        ("B", ports.get("Body", "GND")),
                    ],
                )
            )
        elif ctype == "PMOS":
            devices.append(
                Device(
                    name,
                    "pmos",
                    [
                        ("D", ports["Drain"]),
                        ("G", ports["Gate"]),
                        ("S", ports["Source"]),
                        ("B", ports.get("Body", "VDD")),
                    ],
                )
            )
        elif ctype == "NPN":
            devices.append(
                Device(
                    name,
                    "npn",
                    [
                        ("C", ports["Collector"]),
                        ("B", ports["Base"]),
                        ("E", ports["Emitter"]),
                    ],
                )
            )
        elif ctype == "PNP":
            devices.append(
                Device(
                    name,
                    "pnp",
                    [
                        ("C", ports["Collector"]),
                        ("B", ports["Base"]),
                        ("E", ports["Emitter"]),
                    ],
                )
            )
        elif ctype == "Res":
            devices.append(
                Device(name, "resistor", [("P", ports["Pos"]), ("P", ports["Neg"])])
            )
        elif ctype == "Cap":
            devices.append(
                Device(name, "capacitor", [("P", ports["Pos"]), ("P", ports["Neg"])])
            )
        elif ctype in ("Voltage", "Battery", "AC"):
            # Battery / AC collapse to voltage_src, matching the predicted side
            # (the netlist builder writes all of them with a "V" prefix).
            pos, neg = two_terminal_source_ports(ports)
            devices.append(Device(name, "voltage_src", [("+", pos), ("-", neg)]))
        elif ctype == "Current":
            pos, neg = two_terminal_source_ports(ports)
            devices.append(Device(name, "current_src", [("+", pos), ("-", neg)]))
        elif ctype == "Diode":
            devices.append(
                Device(name, "diode", [("A", ports["In"]), ("K", ports["Out"])])
            )
        elif ctype in AMP_ORDERS:
            devices.append(
                Device(
                    name,
                    "amplifier",
                    [
                        (f"pin{i}", ports[key])
                        for i, key in enumerate(AMP_ORDERS[ctype], start=1)
                    ],
                )
            )
        else:
            raise ValueError(f"unsupported component_type {ctype} in {path}")

    return devices


def parse_device(name: str, prefix: str, rest: list[str]) -> Device:
    if prefix == "M":
        if len(rest) < 5:
            raise ValueError("MOS line needs D G S B model")
        d, g, s, b, model = rest[:5]
        return Device(
            name,
            normalize_device_type(model),
            [("D", d), ("G", g), ("S", s), ("B", b)],
        )

    if prefix == "Q":
        if len(rest) < 4:
            raise ValueError("BJT line needs C B E model")
        c, b, e, model = rest[:4]
        return Device(
            name, normalize_device_type(model), [("C", c), ("B", b), ("E", e)]
        )

    if prefix == "D":
        if len(rest) < 2:
            raise ValueError("diode line needs A K")
        return Device(name, "diode", [("A", rest[0]), ("K", rest[1])])

    if prefix in ("R", "C", "L"):
        if len(rest) < 2:
            raise ValueError(f"{prefix} line needs two pins")
        type_name = {"R": "resistor", "C": "capacitor", "L": "inductor"}[prefix]
        return Device(name, type_name, [("P", rest[0]), ("P", rest[1])])

    if prefix in ("V", "I"):
        if len(rest) < 2:
            raise ValueError(f"{prefix} source line needs + -")
        type_name = "voltage_src" if prefix == "V" else "current_src"
        return Device(name, type_name, [("+", rest[0]), ("-", rest[1])])

    if prefix == "S":
        if len(rest) < 2:
            raise ValueError("switch line needs two pins")
        model = rest[2].lower() if len(rest) >= 3 else "switch"
        return Device(
            name, normalize_device_type(model), [("P", rest[0]), ("P", rest[1])]
        )

    if prefix == "X":
        if len(rest) < 2:
            raise ValueError("subckt line needs at least one net and model")
        *nets, model = rest
        pins = [(f"pin{i}", net) for i, net in enumerate(nets, start=1)]
        return Device(name, normalize_device_type(model), pins)

    if len(rest) >= 2:
        pins = [(f"pin{i}", net) for i, net in enumerate(rest, start=1)]
        return Device(name, prefix.lower(), pins)

    raise ValueError("unrecognized or too-short device line")


def normalize_device_type(device_type: str) -> str:
    lowered = device_type.lower()
    if lowered in ("nmos4", "nmos_4"):
        return "nmos"
    if lowered in ("pmos4", "pmos_4"):
        return "pmos"
    return lowered


def normalize_pin_role(device_type: str, role: str, mos_ds_symmetric: bool) -> str:
    if mos_ds_symmetric and device_type in MOSFET_TYPES and role in ("D", "S"):
        return "DS"
    return role


def canonical_net(net: str) -> str:
    upper = net.upper()
    if upper == "VDD":
        return "VDD"
    if upper in ("GND", "VSS"):
        return "GND"
    return net


def net_node_id(net: str) -> str:
    return f"net:{net}"


def topology_pin_label(device_type: str, pin_index: int) -> str:
    if device_type == "amplifier":
        return "pin:amplifier"
    if device_type in NONPOLAR_TWO_TERMINAL_TYPES:
        return "pin:nonpolar"
    return f"pin:{pin_index}"


def build_topology_graph(devices: list[Device]) -> nx.Graph:
    graph = nx.Graph()

    for index, device in enumerate(devices):
        comp_id = f"comp:{index}"
        graph.add_node(
            comp_id,
            kind="component",
            label=device.device_type,
            original_name=device.name,
        )

        pins = device.pins
        if IGNORE_MOSFET_BULK and device.device_type in MOSFET_TYPES and len(pins) == 4:
            pins = pins[:3]

        for pin_index, (_role, net) in enumerate(pins):
            pin_id = f"pin:{index}:{pin_index}"
            net_id = net_node_id(net)

            graph.add_node(
                pin_id,
                kind="pin",
                label=topology_pin_label(device.device_type, pin_index),
            )
            graph.add_node(net_id, kind="net", label="net:", original_name=net)
            graph.add_edge(comp_id, pin_id, relation="has_pin")
            graph.add_edge(pin_id, net_id, relation="connects")

    return graph


def node_match(a: dict, b: dict) -> bool:
    return a.get("kind") == b.get("kind") and a.get("label") == b.get("label")


def edge_match(a: dict, b: dict) -> bool:
    return a.get("relation") == b.get("relation")


def node_subst_cost(a: dict, b: dict) -> float:
    if node_match(a, b):
        return 0.0
    if a.get("kind") != b.get("kind"):
        return 3.0
    return 1.0


def edge_subst_cost(a: dict, b: dict) -> float:
    return 0.0 if edge_match(a, b) else 1.0


def graph_distance(
    pred_graph: nx.Graph, gt_graph: nx.Graph, timeout: float | None
) -> float | None:
    return nx.graph_edit_distance(
        pred_graph,
        gt_graph,
        node_subst_cost=node_subst_cost,
        node_del_cost=lambda _: 1.0,
        node_ins_cost=lambda _: 1.0,
        edge_subst_cost=edge_subst_cost,
        edge_del_cost=lambda _: 1.0,
        edge_ins_cost=lambda _: 1.0,
        timeout=timeout,
    )


def normalized_distance(
    distance: float | None, pred_graph: nx.Graph, gt_graph: nx.Graph
) -> float | None:
    if distance is None:
        return None
    denom = max(
        1,
        pred_graph.number_of_nodes()
        + pred_graph.number_of_edges()
        + gt_graph.number_of_nodes()
        + gt_graph.number_of_edges(),
    )
    return distance / denom


def normalized_stem(path: Path, strip_suffix: str = "") -> str:
    stem = path.stem
    if strip_suffix and stem.endswith(strip_suffix):
        return stem[: -len(strip_suffix)]
    return stem


def collect_netlists(
    directory: Path, pattern: str, strip_suffix: str = ""
) -> dict[str, Path]:
    return {
        normalized_stem(path, strip_suffix): path
        for path in sorted(directory.glob(pattern))
        if path.is_file()
    }


def common_stems(
    pred_dir: Path,
    gt_dir: Path,
    pred_pattern: str,
    gt_pattern: str,
    pred_strip_suffix: str,
    gt_strip_suffix: str,
) -> tuple[list[str], list[str], list[str], dict[str, Path], dict[str, Path]]:
    pred = collect_netlists(pred_dir, pred_pattern, pred_strip_suffix)
    gt = collect_netlists(gt_dir, gt_pattern, gt_strip_suffix)
    common = sorted(set(pred) & set(gt))
    missing_pred = sorted(set(gt) - set(pred))
    missing_gt = sorted(set(pred) - set(gt))
    return common, missing_pred, missing_gt, pred, gt


def compare_pair(
    stem: str, pred_path: Path, gt_path: Path, args: argparse.Namespace
) -> dict:
    pred_devices = parse_netlist(pred_path)
    gt_devices = parse_netlist(gt_path)
    pred_graph = build_topology_graph(pred_devices)
    gt_graph = build_topology_graph(gt_devices)

    matcher = iso.GraphMatcher(
        pred_graph, gt_graph, node_match=node_match, edge_match=edge_match
    )
    isomorphic = matcher.is_isomorphic()

    distance = 0.0 if isomorphic else None
    if not isomorphic and not args.skip_ged:
        distance = graph_distance(pred_graph, gt_graph, args.timeout)

    norm = normalized_distance(distance, pred_graph, gt_graph)
    return {
        "stem": stem,
        "isomorphic": isomorphic,
        "ged": distance,
        "normalized_ged": norm,
        "pred_devices": len(pred_devices),
        "gt_devices": len(gt_devices),
        "pred_nodes": pred_graph.number_of_nodes(),
        "gt_nodes": gt_graph.number_of_nodes(),
        "pred_edges": pred_graph.number_of_edges(),
        "gt_edges": gt_graph.number_of_edges(),
        "pred_path": str(pred_path),
        "gt_path": str(gt_path),
    }


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare predicted and GT netlists using level-1 topology-only graph matching."
    )
    parser.add_argument(
        "--pred-dir",
        default="resultImage2Net/netlist",
        help="directory containing predicted netlists",
    )
    parser.add_argument(
        "--gt-dir",
        default="Image2Net/netlistGT",
        help="directory containing ground-truth netlists",
    )
    parser.add_argument(
        "--pred-pattern", default="*.cir", help="glob pattern for predicted netlists"
    )
    parser.add_argument(
        "--gt-pattern", default="*.json", help="glob pattern for GT netlists"
    )
    parser.add_argument(
        "--pred-strip-suffix",
        default="",
        help="suffix to strip from predicted file stems",
    )
    parser.add_argument(
        "--gt-strip-suffix", default="", help="suffix to strip from GT file stems"
    )
    parser.add_argument(
        "--out-csv",
        default="resultImage2Net/netlist_graph_compare_level1.csv",
        help="CSV report path",
    )
    parser.add_argument(
        "--timeout", type=float, default=2.0, help="GED timeout per pair in seconds"
    )
    parser.add_argument(
        "--skip-ged", action="store_true", help="only run graph isomorphism, no GED"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="compare at most this many common stems"
    )
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    out_csv = Path(args.out_csv)

    common, missing_pred, missing_gt, pred_files, gt_files = common_stems(
        pred_dir,
        gt_dir,
        args.pred_pattern,
        args.gt_pattern,
        args.pred_strip_suffix,
        args.gt_strip_suffix,
    )
    if args.limit is not None:
        common = common[: args.limit]

    rows = []
    failures = []
    for stem in common:
        pred_path = pred_files[stem]
        gt_path = gt_files[stem]
        try:
            row = compare_pair(stem, pred_path, gt_path, args)
            rows.append(row)
        except Exception as exc:  # keep batch comparison moving
            failures.append((stem, str(exc)))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "stem",
        "isomorphic",
        "ged",
        "normalized_ged",
        "pred_devices",
        "gt_devices",
        "pred_nodes",
        "gt_nodes",
        "pred_edges",
        "gt_edges",
        "pred_path",
        "gt_path",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(row[k]) for k in fields})

    matched = sum(1 for row in rows if row["isomorphic"])
    mismatched = len(rows) - matched
    print(f"common pairs      : {len(rows)}")
    print(f"isomorphic        : {matched}")
    print(f"non-isomorphic    : {mismatched}")
    print(f"missing predicted : {len(missing_pred)}")
    print(f"missing GT        : {len(missing_gt)}")
    print(f"parse failures    : {len(failures)}")
    print(f"csv report        : {out_csv}")

    if mismatched and args.skip_ged:
        print("\nMismatches:")
        for row in [r for r in rows if not r["isomorphic"]][:10]:
            print(f"  {row['stem']}")
    elif mismatched:
        print("\nTop mismatches by GED:")
        sortable = [r for r in rows if not r["isomorphic"]]
        sortable.sort(
            key=lambda r: float("inf") if r["ged"] is None else r["ged"], reverse=True
        )
        for row in sortable[:10]:
            ged = fmt(row["ged"]) or "timeout"
            norm = fmt(row["normalized_ged"])
            print(f"  {row['stem']}: GED={ged}, normalized={norm}")

    if missing_pred:
        print("\nMissing predicted:")
        for stem in missing_pred[:20]:
            print(f"  {stem}")
        if len(missing_pred) > 20:
            print(f"  ... {len(missing_pred) - 20} more")

    if missing_gt:
        print("\nMissing GT:")
        for stem in missing_gt[:20]:
            print(f"  {stem}")
        if len(missing_gt) > 20:
            print(f"  ... {len(missing_gt) - 20} more")

    if failures:
        print("\nParse/compare failures:")
        for stem, message in failures[:20]:
            print(f"  {stem}: {message}")
        if len(failures) > 20:
            print(f"  ... {len(failures) - 20} more")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
