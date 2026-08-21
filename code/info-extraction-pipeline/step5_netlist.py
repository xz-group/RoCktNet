def order_symmetric(a, b):
    """For symmetric 2-pin devices (R/C/L/switch): put VDD on the high side
    (pin 1) and GND on the low side (pin 2), matching the convention that
    current flows top-down from VDD to GND. Non-supply nets keep their order."""
    if b == "VDD" and a != "VDD":
        return b, a
    if a == "GND" and b != "GND":
        return b, a
    return a, b


# SPICE pin orders (no parameter values, model name = class):
#   M: D G S B  (model = nmos / pmos; plain MOS B comes from default GND/VDD)
#   Q: C B E    (model = npn / pnp)
#   D: A K      (model = diode)
#   R/C/L:      pin1 pin2
#   V/I (incl. ac_src, battery): + -
#   S (switch_ideal): pin1 pin2 switch_ideal   (non-standard 2-pin form, basic
#                                               connectivity only)
#   X (amplifier): pin1 pin2 ... amplifier
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

