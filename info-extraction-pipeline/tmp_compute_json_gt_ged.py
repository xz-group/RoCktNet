from __future__ import annotations

import json
import time
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from collections import Counter

from networkx.algorithms import isomorphism as iso
import networkx as nx

import compare_netlists_graph as cmp

ROOT = Path(__file__).parent
PRED_DIR = ROOT / "result" / "netlist"
GT_DIR_CANDIDATES = [
    ROOT / "netlistGT",
    ROOT / "previousTest" / "netlistGTImage2Net",
    Path(
        r"D:\InfoExtractPipeline\ComponentDetection\train_data\Image2Net\ci2n_datasets\validationNetlist\golden"
    ),
]
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_WORKERS = 4

AMP_ORDERS = {
    "Siso_amp": ["In", "Out"],
    "Diso_amp": ["InN", "InP", "Out"],
    "Dido_amp": ["InN", "InP", "OutN", "OutP"],
}

SUPPLY_LOW_ALIASES = {"GND", "VSS", "VSS!", "0"}
SUPPLY_HIGH_ALIASES = {"VDD", "VCC", "VDD!", "VCC!"}
SUPPLY_ALIASES = SUPPLY_LOW_ALIASES | SUPPLY_HIGH_ALIASES


def gt_dir(override: str | None = None) -> Path:
    if override:
        path = Path(override)
        json_count = len(list(path.glob("*.json"))) if path.exists() else 0
        if not path.exists():
            raise FileNotFoundError(f"GT directory does not exist: {path}")
        if json_count == 0:
            raise FileNotFoundError(f"GT directory has no *.json files: {path}")
        return path

    for path in GT_DIR_CANDIDATES:
        if path.exists() and any(path.glob("*.json")):
            return path
    raise FileNotFoundError("No GT json directory found")


def json_devices(path: Path) -> list[cmp.Device]:
    data = json.loads(path.read_text(encoding="utf-8"))
    devices: list[cmp.Device] = []
    counters: dict[str, int] = {}
    for comp in data.get("ckt_netlist", []):
        ctype = comp["component_type"]
        ports = comp.get("port_connection", {})
        counters[ctype] = counters.get(ctype, 0) + 1
        name = f"{ctype}{counters[ctype]}"

        if ctype == "NMOS":
            pins = [
                ("D", ports["Drain"]),
                ("G", ports["Gate"]),
                ("S", ports["Source"]),
                ("B", ports.get("Body", "GND")),
            ]
            devices.append(cmp.Device(name, "nmos", pins))
        elif ctype == "PMOS":
            pins = [
                ("D", ports["Drain"]),
                ("G", ports["Gate"]),
                ("S", ports["Source"]),
                ("B", ports.get("Body", "VDD")),
            ]
            devices.append(cmp.Device(name, "pmos", pins))
        elif ctype == "NPN":
            devices.append(
                cmp.Device(
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
                cmp.Device(
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
                cmp.Device(name, "resistor", [("P", ports["Pos"]), ("P", ports["Neg"])])
            )
        elif ctype == "Cap":
            devices.append(
                cmp.Device(
                    name, "capacitor", [("P", ports["Pos"]), ("P", ports["Neg"])]
                )
            )
        elif ctype == "Voltage":
            devices.append(
                cmp.Device(
                    name,
                    "voltage_src",
                    [("+", ports["Positive"]), ("-", ports["Negative"])],
                )
            )
        elif ctype == "Current":
            devices.append(
                cmp.Device(
                    name, "current_src", [("+", ports["In"]), ("-", ports["Out"])]
                )
            )
        elif ctype == "Diode":
            devices.append(
                cmp.Device(name, "diode", [("A", ports["In"]), ("K", ports["Out"])])
            )
        elif ctype in AMP_ORDERS:
            order = AMP_ORDERS[ctype]
            devices.append(
                cmp.Device(
                    name,
                    "amplifier",
                    [(f"pin{i}", ports[key]) for i, key in enumerate(order, start=1)],
                )
            )
        else:
            raise ValueError(f"unsupported component_type {ctype} in {path}")
    return devices


def normalized_distance(distance: float | None, pred_graph, gt_graph) -> float | None:
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


def golden_counts(devices: list[cmp.Device]) -> dict[str, int]:
    nets = {
        cmp.canonical_net(net)
        for device in devices
        for _, net in device.pins
    }
    ports = sum(len(device.pins) for device in devices)
    return {
        "golden_devices": len(devices),
        "golden_nets": len(nets),
        "golden_ports": ports,
        "golden_denominator": len(devices) + len(nets) + ports,
    }


def ned(distance: float | None, denominator: int) -> float | None:
    if distance is None:
        return None
    if denominator <= 0:
        return None
    return distance / denominator


def normalized_eval_net(net: str, ignore_supply: bool, extended_supply_aliases: bool) -> str:
    upper = net.upper()
    if extended_supply_aliases:
        if ignore_supply:
            # Keep aliases distinct as nodes during loose matching. This lets
            # VSS/VCC act as renameable supply-like nets without accidentally
            # merging VSS with GND or VCC with VDD inside one circuit.
            return upper if upper in SUPPLY_ALIASES else net
        if upper in SUPPLY_LOW_ALIASES:
            return "GND"
        if upper in SUPPLY_HIGH_ALIASES:
            return "VDD"
        return net
    return cmp.canonical_net(net)


def eval_net_label(net: str, ignore_supply: bool, extended_supply_aliases: bool) -> str:
    upper = net.upper()
    if ignore_supply:
        return "NET"
    if extended_supply_aliases:
        if upper in SUPPLY_LOW_ALIASES:
            return "GND"
        if upper in SUPPLY_HIGH_ALIASES:
            return "VDD"
        return "NET"
    canonical = cmp.canonical_net(net)
    return canonical if canonical in cmp.SUPPLY_NETS else "NET"


def build_eval_graph(
    devices: list[cmp.Device],
    ignore_supply: bool,
    mos_ds_symmetric: bool,
    amp_pins_symmetric: bool,
    extended_supply_aliases: bool,
    ignore_amplifier_pins: bool,
) -> nx.Graph:
    graph = nx.Graph()

    for index, device in enumerate(devices):
        comp_id = f"comp:{index}"
        graph.add_node(
            comp_id,
            kind="component",
            label=device.device_type,
            original_name=device.name,
        )

        for pin_index, (role, net) in enumerate(device.pins):
            if ignore_amplifier_pins and device.device_type == "amplifier":
                continue
            pin_role = cmp.normalize_pin_role(device.device_type, role, mos_ds_symmetric)
            if amp_pins_symmetric and device.device_type == "amplifier":
                pin_role = "AMP"
            pin_id = f"pin:{index}:{pin_index}"
            net_id = f"net:{normalized_eval_net(net, ignore_supply, extended_supply_aliases)}"

            graph.add_node(pin_id, kind="pin", label=pin_role)
            graph.add_node(
                net_id,
                kind="net",
                label=eval_net_label(net, ignore_supply, extended_supply_aliases),
                original_name=net,
            )
            graph.add_edge(comp_id, pin_id, relation="has_pin")
            graph.add_edge(pin_id, net_id, relation="connects")

    return graph


def is_supply_alias(net: str) -> bool:
    return net.upper() in SUPPLY_ALIASES


def structural_pin_label(device_type: str, role: str, mos_ds_symmetric: bool, amp_pins_symmetric: bool) -> str:
    if amp_pins_symmetric and device_type == "amplifier":
        return "amplifier.AMP"
    return f"{device_type}.{cmp.normalize_pin_role(device_type, role, mos_ds_symmetric)}"


def net_signature_multiset(
    devices: list[cmp.Device],
    mos_ds_symmetric: bool,
    amp_pins_symmetric: bool,
    ignore_amplifier_pins: bool,
) -> tuple[Counter[tuple[str, ...]], Counter[str]]:
    nets: dict[str, list[str]] = {}
    supply_nets: set[str] = set()
    for device in devices:
        for role, net in device.pins:
            if ignore_amplifier_pins and device.device_type == "amplifier":
                continue
            nets.setdefault(net, []).append(
                structural_pin_label(device.device_type, role, mos_ds_symmetric, amp_pins_symmetric)
            )
            if is_supply_alias(net):
                supply_nets.add(net)

    ordinary = Counter()
    supply = Counter()
    for net, pins in nets.items():
        signature = tuple(sorted(pins))
        if net in supply_nets:
            supply.update(signature)
        else:
            ordinary[signature] += 1
    return ordinary, supply


def supply_partition_relaxed_match(
    pred_devices: list[cmp.Device],
    gt_devices: list[cmp.Device],
    mos_ds_symmetric: bool,
    amp_pins_symmetric: bool,
    ignore_amplifier_pins: bool,
) -> bool:
    pred_sigs, pred_supply = net_signature_multiset(
        pred_devices,
        mos_ds_symmetric,
        amp_pins_symmetric,
        ignore_amplifier_pins,
    )
    gt_sigs, gt_supply = net_signature_multiset(
        gt_devices,
        mos_ds_symmetric,
        amp_pins_symmetric,
        ignore_amplifier_pins,
    )

    common = pred_sigs & gt_sigs
    pred_left = pred_sigs - common
    gt_left = gt_sigs - common

    pred_supply_total = pred_supply.copy()
    for signature, count in pred_left.items():
        for _ in range(count):
            pred_supply_total.update(signature)

    gt_supply_total = gt_supply.copy()
    for signature, count in gt_left.items():
        for _ in range(count):
            gt_supply_total.update(signature)

    return pred_supply_total == gt_supply_total


def compare_one(task: tuple[str, str, str, bool, float, bool, bool, bool, bool, bool]) -> dict:
    (
        stem,
        pred_path_s,
        gt_path_s,
        ignore_supply,
        timeout,
        mos_ds_symmetric,
        amp_pins_symmetric,
        extended_supply_aliases,
        supply_partition_relax,
        ignore_amplifier_pins,
    ) = task
    pred_path = Path(pred_path_s)
    gt_path = Path(gt_path_s)
    pred_devices = cmp.parse_netlist(pred_path)
    gt_devices = json_devices(gt_path)
    counts = golden_counts(gt_devices)
    pred_graph = build_eval_graph(
        pred_devices,
        ignore_supply=ignore_supply,
        mos_ds_symmetric=mos_ds_symmetric,
        amp_pins_symmetric=amp_pins_symmetric,
        extended_supply_aliases=extended_supply_aliases,
        ignore_amplifier_pins=ignore_amplifier_pins,
    )
    gt_graph = build_eval_graph(
        gt_devices,
        ignore_supply=ignore_supply,
        mos_ds_symmetric=mos_ds_symmetric,
        amp_pins_symmetric=amp_pins_symmetric,
        extended_supply_aliases=extended_supply_aliases,
        ignore_amplifier_pins=ignore_amplifier_pins,
    )
    isomorphic = iso.GraphMatcher(
        pred_graph,
        gt_graph,
        node_match=cmp.node_match,
        edge_match=cmp.edge_match,
    ).is_isomorphic()
    relaxed_supply_match = False

    if (
        not isomorphic
        and ignore_supply
        and extended_supply_aliases
        and supply_partition_relax
        and supply_partition_relaxed_match(
            pred_devices,
            gt_devices,
            mos_ds_symmetric,
            amp_pins_symmetric,
            ignore_amplifier_pins,
        )
    ):
        isomorphic = True
        relaxed_supply_match = True

    if isomorphic:
        ged = 0.0
    else:
        ged = cmp.graph_distance(pred_graph, gt_graph, timeout)

    return {
        "stem": stem,
        "ignore_supply": ignore_supply,
        "isomorphic": isomorphic,
        "relaxed_supply_match": relaxed_supply_match,
        "ged": ged,
        "normalized_ged": normalized_distance(ged, pred_graph, gt_graph),
        "ned": ned(ged, counts["golden_denominator"]),
        **counts,
    }


def summarize(rows: list[dict], ignore_supply: bool) -> None:
    mode = "loose_supply" if ignore_supply else "strict_supply"
    mode_rows = [row for row in rows if row["ignore_supply"] == ignore_supply]
    completed = [row for row in mode_rows if row["ged"] is not None]
    timed_out = [row for row in mode_rows if row["ged"] is None]
    non_iso_completed = [row for row in completed if not row["isomorphic"]]
    total = len(mode_rows)
    success = sum(1 for row in mode_rows if row["isomorphic"])
    ned_completed = [row["ned"] for row in completed if row["ned"] is not None]
    ned_non_iso = [row["ned"] for row in non_iso_completed if row["ned"] is not None]
    paper_mean_ned = (
        sum(row["ned"] for row in mode_rows) / total
        if total and not timed_out and all(row["ned"] is not None for row in mode_rows)
        else None
    )

    print(f"\nRESULT {mode}")
    print(f"total: {total}")
    print(f"isomorphic: {success}")
    print(f"successful_rate: {success / total if total else ''}")
    print(f"ged_completed_including_iso_zero: {len(completed)}")
    print(f"ged_timeouts: {len(timed_out)}")
    print(
        f"mean_ged_including_iso_zero: {mean(row['ged'] for row in completed) if completed else ''}"
    )
    print(
        f"mean_normalized_ged_including_iso_zero: {mean(row['normalized_ged'] for row in completed) if completed else ''}"
    )
    print(
        f"mean_ged_non_iso_only: {mean(row['ged'] for row in non_iso_completed) if non_iso_completed else ''}"
    )
    print(
        f"mean_normalized_ged_non_iso_only: {mean(row['normalized_ged'] for row in non_iso_completed) if non_iso_completed else ''}"
    )
    print(
        f"mean_ned_image2net_formula: {paper_mean_ned if paper_mean_ned is not None else ''}"
    )
    print(
        f"mean_ned_completed_including_iso_zero: {mean(ned_completed) if ned_completed else ''}"
    )
    print(
        f"mean_ned_non_iso_only: {mean(ned_non_iso) if ned_non_iso else ''}"
    )
    print("timeout_stems: " + ",".join(row["stem"] for row in timed_out))


def write_csv(rows: list[dict], out_csv: Path) -> None:
    fields = [
        "stem",
        "mode",
        "isomorphic",
        "relaxed_supply_match",
        "ged",
        "ned",
        "normalized_ged",
        "golden_devices",
        "golden_nets",
        "golden_ports",
        "golden_denominator",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["ignore_supply"], r["stem"])):
            writer.writerow({
                "stem": row["stem"],
                "mode": "loose_supply" if row["ignore_supply"] else "strict_supply",
                "isomorphic": row["isomorphic"],
                "relaxed_supply_match": row["relaxed_supply_match"],
                "ged": "" if row["ged"] is None else row["ged"],
                "ned": "" if row["ned"] is None else row["ned"],
                "normalized_ged": "" if row["normalized_ged"] is None else row["normalized_ged"],
                "golden_devices": row["golden_devices"],
                "golden_nets": row["golden_nets"],
                "golden_ports": row["golden_ports"],
                "golden_denominator": row["golden_denominator"],
            })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", default=None, help="directory containing GT JSON files")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="GED timeout per pair in seconds")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="parallel worker count")
    parser.add_argument(
        "--strict-mos-ds",
        dest="mos_ds_symmetric",
        action="store_false",
        default=True,
        help="keep MOS D/S roles distinct instead of treating them as interchangeable",
    )
    parser.add_argument(
        "--strict-amp-pin-order",
        dest="amp_pins_symmetric",
        action="store_false",
        default=True,
        help="keep amplifier pin order distinct instead of treating amplifier ports as interchangeable",
    )
    parser.add_argument(
        "--count-amplifier-pins",
        dest="ignore_amplifier_pins",
        action="store_false",
        default=True,
        help="count amplifier pin number/connectivity instead of only matching the amplifier component",
    )
    parser.add_argument(
        "--basic-supply-aliases",
        dest="extended_supply_aliases",
        action="store_false",
        default=True,
        help="disable VSS/VCC/0 supply aliases and use the legacy VDD/GND handling",
    )
    parser.add_argument(
        "--no-supply-partition-relax",
        dest="supply_partition_relax",
        action="store_false",
        default=True,
        help="disable the loose-supply fallback that ignores GT/pred supply net partition differences",
    )
    parser.add_argument(
        "--out-csv",
        default="result/json_gt_ged_ned_image2net.csv",
        help="CSV path for per-circuit GED/NED rows",
    )
    args = parser.parse_args()

    gt = gt_dir(args.gt_dir)
    pred_files = {
        p.stem: p
        for p in sorted(PRED_DIR.glob("*.cir"))
        if not p.stem.startswith("Book")
    }
    gt_files = {p.stem: p for p in sorted(gt.glob("*.json"))}
    common = sorted(set(pred_files) & set(gt_files))
    print(
        f"pred_non_book={len(pred_files)} gt_json={len(gt_files)} common={len(common)} gt_dir={gt}"
    )
    tasks = [
        (
            stem,
            str(pred_files[stem]),
            str(gt_files[stem]),
            ignore_supply,
            args.timeout,
            args.mos_ds_symmetric,
            args.amp_pins_symmetric,
            args.extended_supply_aliases,
            args.supply_partition_relax,
            args.ignore_amplifier_pins,
        )
        for ignore_supply in (False, True)
        for stem in common
    ]

    started = time.time()
    rows: list[dict] = []
    if args.max_workers <= 1:
        for index, task in enumerate(tasks, start=1):
            row = compare_one(task)
            rows.append(row)
            print(
                f"done {index}/{len(tasks)} "
                f"{'loose' if row['ignore_supply'] else 'strict'} {row['stem']} "
                f"iso={row['isomorphic']} ged={row['ged']} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(compare_one, task) for task in tasks]
            for index, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                rows.append(row)
                print(
                    f"done {index}/{len(tasks)} "
                    f"{'loose' if row['ignore_supply'] else 'strict'} {row['stem']} "
                    f"iso={row['isomorphic']} ged={row['ged']} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )

    summarize(rows, False)
    summarize(rows, True)
    if args.out_csv:
        out_csv = Path(args.out_csv)
        write_csv(rows, out_csv)
        print(f"\ncsv report: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
