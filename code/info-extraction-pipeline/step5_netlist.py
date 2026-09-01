def order_symmetric(a, b):
    if b == "VDD" and a != "VDD":
        return b, a
    if a == "GND" and b != "GND":
        return b, a
    return a, b


def pin_order_for(comp):
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

    for c in components:
        node_list, model = pin_order_for(c)
        parts = [c["name"], *node_list]
        if model is not None:
            parts.append(model)
        lines.append(" ".join(parts))

    lines.append("")
    # lines.append(".end")
    return "\n".join(lines) + "\n"

