"""
Create a human-readable report for graph-level netlist differences.

The report is intentionally diagnostic rather than pass/fail: it prints each
net as the set of device pins attached to it, so net-name differences are much
less distracting while manual topology differences are easier to spot.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import compare_netlists_graph as cmp


def load_mismatch_stems(csv_path: Path) -> list[str]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return [row["stem"] for row in csv.DictReader(f) if row["isomorphic"] == "False"]


def device_label(device: cmp.Device) -> str:
    return f"{device.name}:{device.device_type}"


def net_signatures(devices: list[cmp.Device], ignore_supply_names: bool) -> dict[str, list[str]]:
    nets = defaultdict(list)
    for device in devices:
        for role, net in device.pins:
            display_net = cmp.canonical_net(net)
            if ignore_supply_names and display_net in cmp.SUPPLY_NETS:
                display_net = "NET_SUPPLY_IGNORED"
            nets[display_net].append(f"{device_label(device)}.{role}")
    return {net: sorted(pins) for net, pins in nets.items()}


def structural_net_signatures(devices: list[cmp.Device], ignore_supply_names: bool) -> Counter:
    """Net signatures that ignore concrete device and net names."""
    counter = Counter()
    for _, pins in net_signatures(devices, ignore_supply_names).items():
        structural = []
        for pin in pins:
            left, role = pin.rsplit(".", 1)
            _, dev_type = left.split(":", 1)
            structural.append(f"{dev_type}.{role}")
        counter[tuple(sorted(structural))] += 1
    return counter


def device_type_counts(devices: list[cmp.Device]) -> Counter:
    return Counter(device.device_type for device in devices)


def raw_lines(path: Path) -> list[str]:
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line:
            lines.append(line)
    return lines


def format_counter_delta(a: Counter, b: Counter, left_name: str, right_name: str) -> list[str]:
    lines = []
    all_keys = sorted(set(a) | set(b), key=str)
    for key in all_keys:
        av = a.get(key, 0)
        bv = b.get(key, 0)
        if av != bv:
            lines.append(f"- `{key}`: {left_name}={av}, {right_name}={bv}")
    return lines


def format_net_signatures(title: str, signatures: dict[str, list[str]]) -> list[str]:
    lines = [f"**{title} Net Pin Sets**"]
    for net, pins in sorted(signatures.items(), key=lambda item: (len(item[1]), item[0])):
        lines.append(f"- `{net}`: " + ", ".join(f"`{pin}`" for pin in pins))
    return lines


def write_report(args: argparse.Namespace) -> None:
    mismatch_stems = load_mismatch_stems(Path(args.compare_csv))
    left_dir = Path(args.left_dir)
    right_dir = Path(args.right_dir)

    out_lines = [
        "# Netlist Difference Report",
        "",
        f"Left: `{left_dir}`",
        f"Right: `{right_dir}`",
        f"Compare CSV: `{args.compare_csv}`",
        f"Ignore supply names in structural summary: `{args.ignore_supply_names}`",
        "",
        f"Mismatch count: {len(mismatch_stems)}",
        "",
    ]

    for stem in mismatch_stems:
        left_path = left_dir / f"{stem}.cir"
        right_path = right_dir / f"{stem}.cir"
        left_devices = cmp.parse_netlist(left_path)
        right_devices = cmp.parse_netlist(right_path)

        left_counts = device_type_counts(left_devices)
        right_counts = device_type_counts(right_devices)
        left_struct = structural_net_signatures(left_devices, args.ignore_supply_names)
        right_struct = structural_net_signatures(right_devices, args.ignore_supply_names)
        left_nets = net_signatures(left_devices, args.ignore_supply_names)
        right_nets = net_signatures(right_devices, args.ignore_supply_names)

        out_lines.extend([
            f"## {stem}",
            "",
            "**Raw Netlists**",
            "",
            f"`{left_path}`",
            "```spice",
            *raw_lines(left_path),
            "```",
            f"`{right_path}`",
            "```spice",
            *raw_lines(right_path),
            "```",
            "",
            "**Device Count Differences**",
        ])

        count_delta = format_counter_delta(left_counts, right_counts, "left", "right")
        out_lines.extend(count_delta if count_delta else ["- none"])
        out_lines.extend(["", "**Structural Net Signature Differences**"])

        struct_delta = format_counter_delta(left_struct, right_struct, "left", "right")
        out_lines.extend(struct_delta if struct_delta else ["- none by type/pin multiset; difference may be device-to-device arrangement"])
        out_lines.append("")
        out_lines.extend(format_net_signatures("Left", left_nets))
        out_lines.append("")
        out_lines.extend(format_net_signatures("Right", right_nets))
        out_lines.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-dir", default="netlistAllTopoCorrect")
    parser.add_argument("--right-dir", default="netlistGT")
    parser.add_argument("--compare-csv", default="result/netlist_graph_compare_gt_vs_alltopo_ignore_supply.csv")
    parser.add_argument("--out", default="result/netlist_gt_vs_alltopo_diff_report.md")
    parser.add_argument("--ignore-supply-names", action="store_true")
    args = parser.parse_args()
    write_report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
