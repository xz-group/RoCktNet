"""
build_netlist.py

Reads incidence_matrix/<image>.json (produced by build_incidence_matrix.py).
For each image with zero red_flags across all components, emits a SPICE-style
netlist to netlist/<image>.cir. Any image with at least one red_flag is
skipped and reported in the batch summary.

SPICE pin orders (no parameter values, model name = class):
  M: D G S B  (model = nmos / pmos; plain MOS B comes from default GND/VDD)
  Q: C B E    (model = npn / pnp)
  D: A K      (model = diode)
  R/C/L:      pin1 pin2
  V/I (incl. ac_src, battery): + -
  S (switch_ideal): pin1 pin2 switch_ideal   (non-standard 2-pin form, basic
                                              connectivity only)
  X (amplifier): pin1 pin2 ... amplifier

Usage:
  python build_netlist.py                  # batch over all incidence_matrix/*.json
  python build_netlist.py --image 000058   # single image
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent
IN_DIR = ROOT / "incidence_matrix"
OUT_DIR = ROOT / "netlist"


def order_symmetric(a, b):
    """For symmetric 2-pin devices (R/C/L/switch): put VDD on the high side
    (pin 1) and GND on the low side (pin 2), matching the convention that
    current flows top-down from VDD to GND. Non-supply nets keep their order."""
    if b == "VDD" and a != "VDD":
        return b, a
    if a == "GND" and b != "GND":
        return b, a
    return a, b


def pin_order_for(comp):
    """
    Return (ordered_node_list, model_name_or_None) for a component.
    Pulls node names from comp['pins'] in SPICE-canonical pin order.
    """
    cls = comp["class"]
    p = comp["pins"]

    if cls in ("nmos", "nmos-bulk"):
        return [p["D"], p["G"], p["S"], p["B"]], "nmos"
    if cls in ("pmos", "pmos-bulk"):
        return [p["D"], p["G"], p["S"], p["B"]], "pmos"
    if cls == "npn":
        return [p["C"], p["B"], p["E"]], "npn"
    if cls == "pnp":
        return [p["C"], p["B"], p["E"]], "pnp"
    if cls == "diode":
        return [p["A"], p["K"]], "diode"
    if cls in ("voltage_src", "ac_src", "battery", "current_src"):
        return [p["+"], p["-"]], None
    if cls in ("resistor", "capacitor", "inductor"):
        a, b = order_symmetric(p["1"], p["2"])
        return [a, b], None
    if cls == "switch_ideal":
        a, b = order_symmetric(p["1"], p["2"])
        return [a, b], "switch_ideal"
    if cls == "amplifier":
        # pin1, pin2, ... in numerical order
        pin_keys = sorted(
            (k for k in p if k.startswith("pin")),
            key=lambda k: int(k[3:]),
        )
        return [p[k] for k in pin_keys], "amplifier"
    # Fallback: dump pins in dict order
    return list(p.values()), cls


def generate_netlist(data):
    image = data["image"]
    components = data["components"]
    nodes = data["nodes"]

    lines = []
    # Title line (SPICE treats the first line as the title)
    # lines.append(f"* netlist for {image}")
    # lines.append(f"* source: incidence_matrix/{image}.json")
    # lines.append(f"* nodes ({len(nodes)}): {', '.join(nodes)}")
    # lines.append(f"* components: {len(components)}")
    # lines.append("")

    for c in components:
        node_list, model = pin_order_for(c)
        parts = [c["name"], *node_list]
        if model is not None:
            parts.append(model)
        lines.append(" ".join(parts))

    lines.append("")
    # lines.append(".end")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--image",
        default=None,
        help="single image id (e.g. 000058); if omitted, batch all",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.image:
        image_ids = [args.image]
    else:
        image_ids = sorted(p.stem for p in IN_DIR.glob("*.json"))

    generated = []
    skipped = []  # (image_id, [component names with red_flags])
    missing = []  # incidence json not found

    for img in image_ids:
        in_path = IN_DIR / f"{img}.json"
        if not in_path.exists():
            missing.append(img)
            continue
        with open(in_path) as f:
            data = json.load(f)
        red = [c["name"] for c in data["components"] if c.get("red_flags")]
        if red:
            skipped.append((img, red))
            continue
        out_path = OUT_DIR / f"{img}.cir"
        with open(out_path, "w") as f:
            f.write(generate_netlist(data))
        generated.append(img)

    # ---- summary ----
    total = len(image_ids)
    print("=== Netlist generation summary ===")
    print(f"Total images scanned : {total}")
    print(f"Generated            : {len(generated)}")
    for img in generated:
        print(f"  [OK] {img}.cir")
    print(f"Skipped (red_flags)  : {len(skipped)}")
    for img, red_comps in skipped:
        print(f"  [SKIP] {img} -- red_flags on: {', '.join(red_comps)}")
    if missing:
        print(f"Missing input        : {len(missing)}")
        for img in missing:
            print(f"  [--] {img}.json not found")


if __name__ == "__main__":
    main()
