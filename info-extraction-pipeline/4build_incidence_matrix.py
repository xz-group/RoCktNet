"""
build_incidence_matrix.py

Reads three per-image input artifacts and emits a structured JSON describing
the incidence between components and electrical nodes:

  component_bbox/<image>.txt    -> bbox + class for each component
  orientation/<image>.json      -> orientation 'u'/'r'/'d'/'l' for required classes
  node_touches/<image>.json     -> per-node touches (component_bbox_idx, edge, contact_xy)
                                   + removed_single_bbox_nodes (potential ports)

Writes:
  incidence_matrix/<image>.json

Usage:
  python build_incidence_matrix.py                  # batch over all images
  python build_incidence_matrix.py --image 000058   # single image
"""

import argparse
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parent
BBOX_DIR = ROOT / "component_bbox"
ORIENT_DIR = ROOT / "orientation"
TOUCH_DIR = ROOT / "node_touches"
IMAGE_DIR = ROOT / "images"
OUT_DIR = ROOT / "incidence_matrix"
TEXT_BBOX_DIR = ROOT / "masked_no_text_images" / "_text_bboxes"
COMBINED_LINES_DIR = ROOT / "combined_lines"
NODE_DATA_DIR = ROOT / "nodes" / "data"

CLASS_NAMES = {
    0: "gnd",
    1: "vdd",
    2: "nmos",
    3: "nmos-bulk",
    4: "pmos",
    5: "pmos-bulk",
    6: "npn",
    7: "pnp",
    8: "resistor",
    9: "capacitor",
    10: "inductor",
    11: "diode",
    12: "voltage_src",
    13: "ac_src",
    14: "current_src",
    15: "battery",
    16: "amplifier",
    17: "switch_ideal",
}

PREFIX = {
    "nmos": "M",
    "nmos-bulk": "M",
    "pmos": "M",
    "pmos-bulk": "M",
    "npn": "Q",
    "pnp": "Q",
    "resistor": "R",
    "capacitor": "C",
    "inductor": "L",
    "diode": "D",
    "voltage_src": "V",
    "ac_src": "V",
    "battery": "V",
    "current_src": "I",
    "amplifier": "X",
    "switch_ideal": "S",
}

# G-side occupies this fraction of bbox along the orientation axis.
G_SPLIT_RATIO = 0.30

# Incidence-time safety merge for duplicate corner contacts from the same node.
CLOSE_SAME_NODE_TOUCH_MERGE_PX = 20

# Pixel checks ignore the endpoint-adjacent pixel on each side of the segment.
SHORT_ENDPOINT_MARGIN_PX = 1

# Scoped dangling-pin rescue: a MOS/BJT non-gate pin left on a label_net usually
# means its wire was broken upstream. Extend that pin's stub tip outward up to
# this many pixels and, if it reaches a real node's line within the tolerance,
# adopt that node. Only ever applied to already-broken (label_net) non-gate pins.
EXTEND_PIN_PROBE_PX = 50.0
EXTEND_PIN_PROBE_TOL_PX = 6.0

# Broken-wire bridge for a dangling pin stub: even when the gap to a real node
# is NOT inked (HAWP dropped the segment entirely), bridge it when the stub and
# the target node's dangling end are mutually collinear and face each other
# across a short gap -- the unmistakable signature of one wire split in two.
ENDPOINT_BRIDGE_MAX_GAP_PX = 16.0
ENDPOINT_BRIDGE_COLLINEAR_COS = 0.95

# A "black line" between two contacts may contain tiny bright breaks (anti-
# aliasing, a junction marker, a crossing wire). Tolerate a bright run up to this
# many pixels; a genuine non-connection leaves a much longer bright stretch.
# (3 catches real shorts like 027/M14's 3px gap without the false shorts seen at 4.)
SHORT_LINE_MAX_GAP_PX = 3

# A wire drawn straight through a MOS body lands a non-G touch on the edge
# opposite the gate, aligned with the gate contact on the perpendicular axis.
# This is the max allowed offset (px) on that axis to treat it as a pass-through.
MOS_PASSTHROUGH_ALIGN_PX = 8.0

# Amplifier differential-input detection. A single (Siso) connection puts the
# input and output pins on the symmetry axis (same x for u/d, same y for l/r);
# a differential input offsets the lone detected input pin from the output by a
# sizeable fraction of the symbol. If the offset along that axis exceeds this
# fraction of the perpendicular bbox dimension, treat the input side as
# differential (2 pins). Measured: genuine Siso ~0.005, Diso >= 0.17.
AMP_DIFF_ALIGN_RATIO = 0.10

# If a missing-pin red flag is caused by overlapping/nearby component boxes,
# rescue the pin only when the expected edge touches another bbox this closely.
CLOSE_BBOX_PIN_RESCUE_PX = 30

# Last-resort G/B pin rescue for MOS/BJT devices: if the expected gate/base
# edge has a very short wire that never became a line segment, look for black
# pixels just outside the ideal edge-center contact.
IDEAL_GB_BLACK_PIXEL_RESCUE_RADIUS_PX = 4

# Red-flag rescue: drop extra touches caused by OCR text boxes and recover
# missing pins from nearby line segments.
TEXT_TOUCH_MARGIN_PX = 3.0
NEARBY_LINE_PIN_RESCUE_PX = 20.0
# A wire tip this close to a ground/supply symbol bbox is treated as connected
# to it, even if the wire stopped just short of the box (the symbol's short
# connection stub often goes undetected, leaving the pin a floating label net).
SUPPLY_SYMBOL_PROXIMITY_PX = 6.0
# A label-net pin whose stub reaches this close to a junction marker adopts the
# real net merged at that junction (the node-stage junction force-merge misses
# it when the stub stops just short of the junction bbox).
JUNCTION_PROXIMITY_PX = 8.0
# Score penalty added to an unanchored (node_id < 0) nearby-line candidate so a
# real anchored-node line always wins over a closer dangling fragment.
NEARBY_LINE_UNANCHORED_PENALTY = 1000.0

# Bbox-adaptive grayscale threshold bounds for "black enough" line pixels.
MIN_BLACK_THRESHOLD = 35
MAX_BLACK_THRESHOLD = 90
BLACK_THRESHOLD_AVG_RATIO = 0.35

# (base_type, orient) -> (g_pin_name, [low_axis_pin, high_axis_pin])
# axis is y (asc) for orient l/r; x (asc) for orient u/d.
# Derived directly from note.txt pin rules.
MOS_PIN_LAYOUT = {
    ("nmos", "l"): ("G", ["D", "S"]),
    ("nmos", "r"): ("G", ["D", "S"]),
    ("nmos", "u"): ("G", ["S", "D"]),
    ("nmos", "d"): ("G", ["D", "S"]),
    ("pmos", "l"): ("G", ["S", "D"]),
    ("pmos", "r"): ("G", ["S", "D"]),
    ("pmos", "u"): ("G", ["D", "S"]),
    ("pmos", "d"): ("G", ["S", "D"]),
    ("npn", "l"): ("B", ["C", "E"]),
    ("npn", "r"): ("B", ["C", "E"]),
    ("npn", "u"): ("B", ["E", "C"]),
    ("npn", "d"): ("B", ["C", "E"]),
    ("pnp", "l"): ("B", ["E", "C"]),
    ("pnp", "r"): ("B", ["E", "C"]),
    ("pnp", "u"): ("B", ["C", "E"]),
    ("pnp", "d"): ("B", ["E", "C"]),
}

# orient -> the edge name where G/B/K should sit
ORIENT_EDGE = {"l": "left", "r": "right", "u": "top", "d": "bottom"}

OPPOSITE_EDGE = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}

# 2-pin device classes — subject to the 3-touch pre-pass cleanup.
TWO_PIN_CLASSES = {8, 9, 10, 11, 12, 13, 14, 15, 17}


def load_bboxes(path):
    bboxes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("x1"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            x1, y1, x2, y2 = map(float, parts[:4])
            cls = int(parts[4])
            conf = float(parts[5]) if len(parts) > 5 else None
            bboxes.append(
                {
                    "idx": len(bboxes),
                    "bbox": [x1, y1, x2, y2],
                    "cls": cls,
                    "conf": conf,
                }
            )
    return bboxes


def load_orientation(path):
    with open(path) as f:
        data = json.load(f)
    orient = {}
    for c in data.get("components", []):
        idx = c["component_id"] - 1
        orient[idx] = c.get("orientation")
    return orient


def load_touches(path):
    with open(path) as f:
        return json.load(f)


def load_text_bboxes(path):
    bboxes = []
    if not path.exists():
        return bboxes
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=5)
            if len(parts) < 4:
                continue
            try:
                x1, y1, x2, y2 = map(float, parts[:4])
                conf = float(parts[4]) if len(parts) >= 5 else None
            except ValueError:
                continue
            text = parts[5] if len(parts) >= 6 else ""
            bboxes.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf,
                    "text": text,
                }
            )
    return bboxes


def load_combined_lines(path):
    lines = []
    if not path.exists():
        return lines
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                x1, y1, x2, y2 = map(float, parts[:4])
            except ValueError:
                continue
            lines.append([[x1, y1], [x2, y2]])
    return lines


def load_node_data(path):
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_image_path(image_id):
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        path = IMAGE_DIR / f"{image_id}{ext}"
        if path.exists():
            return path
    return None


def load_image(path):
    return Image.open(path).convert("RGB")


def grayscale(rgb):
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


def bbox_black_threshold(image, bbox):
    width, height = image.size
    x1, y1, x2, y2 = bbox
    left = max(0, min(width - 1, int(round(x1))))
    top = max(0, min(height - 1, int(round(y1))))
    right = max(left + 1, min(width, int(round(x2))))
    bottom = max(top + 1, min(height, int(round(y2))))

    pixels = image.crop((left, top, right, bottom)).getdata()
    vals = [grayscale(p) for p in pixels]
    if not vals:
        return MIN_BLACK_THRESHOLD
    avg = sum(vals) / len(vals)
    return max(
        MIN_BLACK_THRESHOLD, min(MAX_BLACK_THRESHOLD, avg * BLACK_THRESHOLD_AVG_RATIO)
    )


def bresenham_line(p0, p1):
    x0, y0 = map(lambda v: int(round(v)), p0)
    x1, y1 = map(lambda v: int(round(v)), p1)
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    points = []

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy

    return points


def is_black_line_between(image, p0, p1, bbox):
    points = bresenham_line(p0, p1)
    margin = SHORT_ENDPOINT_MARGIN_PX
    if len(points) <= 2 * margin:
        return False

    width, height = image.size
    threshold = bbox_black_threshold(image, bbox)
    gap = 0
    for x, y in points[margin:-margin]:
        if x < 0 or y < 0 or x >= width or y >= height:
            return False
        if grayscale(image.getpixel((x, y))) > threshold:
            gap += 1
            if gap > SHORT_LINE_MAX_GAP_PX:
                return False
        else:
            gap = 0
    return True


def black_line_to_touch(image, p0, touch, bbox):
    """is_black_line_between from p0 to a touch, retrying each contributor's own
    contact point. A merged multi-contributor touch stores the AVERAGE of its
    contributors' contacts, which can land in white space between two distinct
    right-/opposite-edge wires; the straight probe to that midpoint then leaves
    the real conductor and misses. Testing each contributor's true contact
    recovers the actual wire (e.g. a gate that passes through the body and exits
    aligned with the gate on one contributor but averaged off it)."""
    if is_black_line_between(image, p0, touch["contact_xy"], bbox):
        return True
    for c in touch.get("contributors", []):
        cxy = c.get("contact_xy")
        if cxy is not None and is_black_line_between(image, p0, cxy, bbox):
            return True
    return False


def is_on_g_side(touch, bbox, orientation):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx, cy = touch["contact_xy"]
    if orientation == "l":
        return cx <= x1 + G_SPLIT_RATIO * w
    if orientation == "r":
        return cx >= x2 - G_SPLIT_RATIO * w
    if orientation == "u":
        return cy <= y1 + G_SPLIT_RATIO * h
    if orientation == "d":
        return cy >= y2 - G_SPLIT_RATIO * h
    return False


def perp_axis_idx(orientation):
    # returns the contact_xy index (0=x, 1=y) used to sort non-G touches
    return 1 if orientation in ("l", "r") else 0


def touch_distance(a, b):
    ax, ay = a["contact_xy"]
    bx, by = b["contact_xy"]
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def point_rect_distance(x, y, rect):
    x1, y1, x2, y2 = rect
    dx = max(x1 - x, 0, x - x2)
    dy = max(y1 - y, 0, y - y2)
    return (dx * dx + dy * dy) ** 0.5


def point_inside_rect(x, y, rect):
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def text_overlap_score(touch, text_bboxes, margin=TEXT_TOUCH_MARGIN_PX):
    if not text_bboxes:
        return 0.0
    x, y = touch["contact_xy"]
    score = 0.0
    for item in text_bboxes:
        rect = item["bbox"]
        dist = point_rect_distance(x, y, rect)
        if point_inside_rect(x, y, rect):
            score += 100.0
        elif dist <= margin:
            score += max(0.0, margin - dist)
    return score


def prune_text_overlap_excess(
    touches,
    expected_count,
    text_bboxes,
    flags,
    note_prefix,
):
    if len(touches) <= expected_count or not text_bboxes:
        return touches

    scored = [
        (text_overlap_score(touch, text_bboxes), idx, touch)
        for idx, touch in enumerate(touches)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    remove_count = len(touches) - expected_count
    remove_indices = {idx for score, idx, _ in scored[:remove_count] if score > 0}
    if len(remove_indices) < remove_count:
        return touches

    removed = [touches[idx] for idx in sorted(remove_indices)]
    kept = [touch for idx, touch in enumerate(touches) if idx not in remove_indices]
    details = []
    for touch in removed:
        score = text_overlap_score(touch, text_bboxes)
        details.append(
            f"node{touch.get('node_id')}:{touch.get('edge')}:score={score:.2f}"
        )
    flags.append(f"{note_prefix}_dropped_text_overlap_touch=" + ",".join(details))
    return kept


def median(values):
    vals = sorted(float(v) for v in values)
    n = len(vals)
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def merge_touch_cluster(cluster):
    cx = median(t["contact_xy"][0] for t in cluster)
    cy = median(t["contact_xy"][1] for t in cluster)
    rep = min(
        cluster,
        key=lambda t: ((t["contact_xy"][0] - cx) ** 2 + (t["contact_xy"][1] - cy) ** 2),
    )
    merged = dict(rep)
    merged["contact_xy"] = [cx, cy]
    merged["merged_edges"] = sorted({t["edge"] for t in cluster})
    merged["merged_touch_count"] = len(cluster)
    return merged


def merge_close_same_node_touches(touches, max_dist=None):
    """Merge close duplicate contacts for the same node on the same component.

    The caller passes touches already restricted to one component and one side
    class (G-side or non-G-side), so this does not merge across pin categories.
    """
    if max_dist is None:
        max_dist = CLOSE_SAME_NODE_TOUCH_MERGE_PX

    indexed = list(enumerate(touches))
    by_node = defaultdict(list)
    for idx, touch in indexed:
        by_node[touch["node_id"]].append((idx, touch))

    clusters_with_order = []
    for node_items in by_node.values():
        clusters = []
        for idx, touch in node_items:
            placed = False
            for cluster in clusters:
                if any(
                    touch_distance(touch, other_touch) <= max_dist
                    for _, other_touch in cluster
                ):
                    cluster.append((idx, touch))
                    placed = True
                    break
            if not placed:
                clusters.append([(idx, touch)])
        clusters_with_order.extend(clusters)

    clusters_with_order.sort(key=lambda cluster: min(idx for idx, _ in cluster))
    merged = []
    merge_count = 0
    for cluster in clusters_with_order:
        cluster_touches = [touch for _, touch in cluster]
        if len(cluster_touches) == 1:
            merged.append(cluster_touches[0])
        else:
            merged.append(merge_touch_cluster(cluster_touches))
            merge_count += len(cluster_touches) - 1
    return merged, merge_count


def clamp(value, low, high):
    return max(low, min(high, value))


def candidate_contact_for_line_edge(line, bbox, edge):
    x1, y1, x2, y2 = bbox
    best_endpoint = None
    for endpoint in line:
        px, py = endpoint
        if edge == "left":
            axis_gap = max(0.0, y1 - py, py - y2)
            edge_gap = abs(px - x1)
            contact = [x1, clamp(py, y1, y2)]
        elif edge == "right":
            axis_gap = max(0.0, y1 - py, py - y2)
            edge_gap = abs(px - x2)
            contact = [x2, clamp(py, y1, y2)]
        elif edge == "top":
            axis_gap = max(0.0, x1 - px, px - x2)
            edge_gap = abs(py - y1)
            contact = [clamp(px, x1, x2), y1]
        else:
            axis_gap = max(0.0, x1 - px, px - x2)
            edge_gap = abs(py - y2)
            contact = [clamp(px, x1, x2), y2]

        gap = (edge_gap * edge_gap + axis_gap * axis_gap) ** 0.5
        if best_endpoint is None or gap < best_endpoint[0]:
            best_endpoint = (gap, contact, "endpoint")

    if best_endpoint is not None and best_endpoint[0] <= NEARBY_LINE_PIN_RESCUE_PX:
        return best_endpoint

    fallback = candidate_segment_contact_for_line_edge(line, bbox, edge)
    if fallback is None:
        return best_endpoint
    if best_endpoint is None or fallback[0] < best_endpoint[0]:
        return fallback
    return best_endpoint


def closest_point_on_segment_to_point(point, seg_a, seg_b):
    px, py = point
    ax, ay = seg_a
    bx, by = seg_b
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    denom = vx * vx + vy * vy
    if denom <= 0:
        return [ax, ay]
    t = clamp((wx * vx + wy * vy) / denom, 0.0, 1.0)
    return [ax + t * vx, ay + t * vy]


def segment_segment_distance(line_a, line_b):
    (a1, a2), (b1, b2) = line_a, line_b
    dists = [
        ((a1, closest_point_on_segment_to_point(a1, b1, b2))),
        ((a2, closest_point_on_segment_to_point(a2, b1, b2))),
        ((b1, closest_point_on_segment_to_point(b1, a1, a2))),
        ((b2, closest_point_on_segment_to_point(b2, a1, a2))),
    ]
    best = None
    for p, q in dists:
        dist = ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5
        if best is None or dist < best[0]:
            best = (dist, p, q)
    return best


def segment_intersection_vertical(line, x, y1, y2):
    (ax, ay), (bx, by) = line
    if (ax - x) * (bx - x) > 0:
        return None
    if ax == bx:
        if ax != x:
            return None
        lo, hi = sorted((ay, by))
        y = clamp(0.5 * (max(lo, y1) + min(hi, y2)), y1, y2)
        return [x, y] if lo <= y2 and hi >= y1 else None
    t = (x - ax) / (bx - ax)
    if t < 0.0 or t > 1.0:
        return None
    y = ay + t * (by - ay)
    if y1 <= y <= y2:
        return [x, y]
    return None


def segment_intersection_horizontal(line, y, x1, x2):
    (ax, ay), (bx, by) = line
    if (ay - y) * (by - y) > 0:
        return None
    if ay == by:
        if ay != y:
            return None
        lo, hi = sorted((ax, bx))
        x = clamp(0.5 * (max(lo, x1) + min(hi, x2)), x1, x2)
        return [x, y] if lo <= x2 and hi >= x1 else None
    t = (y - ay) / (by - ay)
    if t < 0.0 or t > 1.0:
        return None
    x = ax + t * (bx - ax)
    if x1 <= x <= x2:
        return [x, y]
    return None


def candidate_segment_contact_for_line_edge(line, bbox, edge):
    x1, y1, x2, y2 = bbox
    if edge in ("left", "right"):
        edge_x = x1 if edge == "left" else x2
        projected = segment_intersection_vertical(line, edge_x, y1, y2)
        if projected is not None:
            return (0.0, projected, "segment")
        target = [edge_x, 0.5 * (y1 + y2)]
        closest = closest_point_on_segment_to_point(target, line[0], line[1])
        px, py = closest
        contact = [edge_x, clamp(py, y1, y2)]
        axis_gap = max(0.0, y1 - py, py - y2)
        edge_gap = abs(px - edge_x)
    else:
        edge_y = y1 if edge == "top" else y2
        projected = segment_intersection_horizontal(line, edge_y, x1, x2)
        if projected is not None:
            return (0.0, projected, "segment")
        target = [0.5 * (x1 + x2), edge_y]
        closest = closest_point_on_segment_to_point(target, line[0], line[1])
        px, py = closest
        contact = [clamp(px, x1, x2), edge_y]
        axis_gap = max(0.0, x1 - px, px - x2)
        edge_gap = abs(py - edge_y)

    gap = (edge_gap * edge_gap + axis_gap * axis_gap) ** 0.5
    return (gap, contact, "segment")


def edge_center_distance(contact, bbox, edge):
    x1, y1, x2, y2 = bbox
    if edge in ("left", "right"):
        return abs(contact[1] - 0.5 * (y1 + y2))
    return abs(contact[0] - 0.5 * (x1 + x2))


def point_distance(p0, p1):
    return ((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2) ** 0.5


def mos_bjt_ideal_geometry(bbox, orient):
    x1, y1, x2, y2 = bbox
    mid_x = 0.5 * (x1 + x2)
    mid_y = 0.5 * (y1 + y2)
    if orient == "l":
        return {
            "g": [x1, mid_y],
            "non_g_edges": ["top", "bottom"],
            "bulk": [x2, mid_y],
        }
    if orient == "r":
        return {
            "g": [x2, mid_y],
            "non_g_edges": ["top", "bottom"],
            "bulk": [x1, mid_y],
        }
    if orient == "u":
        return {
            "g": [mid_x, y1],
            "non_g_edges": ["bottom", "bottom"],
            "bulk": [mid_x, y2],
        }
    if orient == "d":
        return {
            "g": [mid_x, y2],
            "non_g_edges": ["top", "top"],
            "bulk": [mid_x, y1],
        }
    return None


def edge_ideal_point(bbox, edge):
    x1, y1, x2, y2 = bbox
    if edge == "left":
        return [x1, 0.5 * (y1 + y2)]
    if edge == "right":
        return [x2, 0.5 * (y1 + y2)]
    if edge == "top":
        return [0.5 * (x1 + x2), y1]
    if edge == "bottom":
        return [0.5 * (x1 + x2), y2]
    return [0.5 * (x1 + x2), 0.5 * (y1 + y2)]


def edge_match_cost(touch, edge, bbox):
    edge_penalty = 0.0 if touch.get("edge") == edge else 1000.0
    return edge_penalty + point_distance(touch["contact_xy"], edge_ideal_point(bbox, edge))


def best_touch_edge_match(touches, edges, bbox):
    if len(touches) < len(edges):
        return None
    best = None
    for chosen in itertools.permutations(touches, len(edges)):
        score = sum(
            edge_match_cost(t, edge, bbox)
            for t, edge in zip(chosen, edges)
        )
        if best is None or score < best[0]:
            best = (score, list(chosen))
    return best[1] if best else None


def bulk_touch_max_ideal_distance(bbox):
    x1, y1, x2, y2 = bbox
    return max(20.0, 0.5 * max(x2 - x1, y2 - y1))


def mos_bjt_count_status(cname, g_touches, ng_touches):
    base_type = cname.replace("-bulk", "")
    is_bulk_class = cname.endswith("-bulk")
    kept_non_bulk_3 = False
    reclassified_to_bulk = False

    if (not is_bulk_class) and base_type in ("nmos", "pmos") and len(ng_touches) == 3:
        n_distinct = len({t["node_id"] for t in ng_touches})
        if n_distinct == 3:
            is_bulk_class = True
            reclassified_to_bulk = True
        else:
            kept_non_bulk_3 = True

    expected_ng = 3 if is_bulk_class else 2
    g_wrong = len(g_touches) != 1
    ng_wrong = len(ng_touches) != expected_ng and not kept_non_bulk_3
    return g_wrong, ng_wrong, reclassified_to_bulk, kept_non_bulk_3, expected_ng


def cascade_single_bbox(nid, touches_by_bbox, single_by_bbox):
    """If node `nid` now touches at most 1 distinct bbox in touches_by_bbox,
    move all its remaining regular touches into single_by_bbox (port pool).
    Returns True if the node was cascaded."""
    nid_touches = []
    for b_idx in list(touches_by_bbox.keys()):
        for t in list(touches_by_bbox[b_idx]):
            if t["node_id"] == nid:
                nid_touches.append((b_idx, t))
    distinct_bboxes = {bi for bi, _ in nid_touches}
    if len(distinct_bboxes) <= 1 and nid_touches:
        for b_idx, t in nid_touches:
            single_by_bbox[b_idx].append(t)
            touches_by_bbox[b_idx].remove(t)
        return True
    return False


def prepass_two_pin(bboxes, touches_by_bbox, single_by_bbox):
    """
    For every 2-pin device that starts with exactly 3 touches, keep one
    (top, bottom) pair or one (left, right) pair and discard the third
    'wire-wrap' touch. Cascade the discarded touch's node to single_by_bbox
    if it now only touches one distinct bbox.

    Returns dict[bbox_idx -> list[flag_string]] describing what was dropped.
    """
    prepass_flags = {}
    for b in bboxes:
        if b["cls"] not in TWO_PIN_CLASSES:
            continue
        b_idx = b["idx"]
        touches = list(touches_by_bbox.get(b_idx, []))
        if len(touches) != 3:
            continue

        edges_present = {t["edge"] for t in touches}
        if {"top", "bottom"}.issubset(edges_present):
            keep_edges = ("top", "bottom")
        elif {"left", "right"}.issubset(edges_present):
            keep_edges = ("left", "right")
        else:
            continue  # no clean pair; let downstream red_flag handle it

        kept, discarded, used = [], [], set()
        for t in touches:
            if t["edge"] in keep_edges and t["edge"] not in used:
                kept.append(t)
                used.add(t["edge"])
            else:
                discarded.append(t)

        touches_by_bbox[b_idx] = kept

        flag_list = []
        for d in discarded:
            cascaded = cascade_single_bbox(
                d["node_id"], touches_by_bbox, single_by_bbox
            )
            tag = "+cascade" if cascaded else ""
            flag_list.append(
                f"prepass_dropped_{d['edge']}_touch_node{d['node_id']}{tag}"
            )
        prepass_flags[b_idx] = flag_list

    return prepass_flags


def build_image(image_id):
    bbox_path = BBOX_DIR / f"{image_id}.txt"
    orient_path = ORIENT_DIR / f"{image_id}.json"
    touch_path = TOUCH_DIR / f"{image_id}.json"
    image_path = find_image_path(image_id)
    missing = [str(p) for p in (bbox_path, orient_path, touch_path) if not p.exists()]
    if image_path is None:
        missing.append(str(IMAGE_DIR / f"{image_id}.[png|jpg|...]"))
    if missing:
        raise FileNotFoundError(f"Missing inputs for {image_id}: {missing}")

    bboxes = load_bboxes(bbox_path)
    orientations = load_orientation(orient_path)
    touch_data = load_touches(touch_path)
    image = load_image(image_path)
    text_bboxes = load_text_bboxes(TEXT_BBOX_DIR / f"{image_id}.txt")
    combined_lines = load_combined_lines(COMBINED_LINES_DIR / f"{image_id}.txt")
    node_data = load_node_data(NODE_DATA_DIR / f"{image_id}.json")
    line_node_ids = node_data.get("line_node_ids", [])
    junction_bboxes = node_data.get("junction_bboxes", []) or []

    # bbox_idx -> list of {node_id, edge, contact_xy}
    touches_by_bbox = defaultdict(list)
    for n in touch_data["nodes"]:
        nid = n["node_id"]
        for t in n["touches"]:
            touch = dict(t)
            touch["node_id"] = nid
            touches_by_bbox[t["component_bbox_idx"]].append(touch)

    # Single-bbox (port-candidate) nodes
    single_by_bbox = defaultdict(list)
    for n in touch_data.get("removed_single_bbox_nodes", []):
        nid = n["node_id"]
        for t in n["touches"]:
            touch = dict(t)
            touch["node_id"] = nid
            single_by_bbox[t["component_bbox_idx"]].append(touch)

    # Pre-pass: clean up 3-touch 2-pin devices BEFORE complex-device handling.
    # May cascade nodes into single_by_bbox.
    prepass_flags = prepass_two_pin(bboxes, touches_by_bbox, single_by_bbox)

    # Active nodes = those still having ≥1 touch in touches_by_bbox after prepass.
    # Nodes that cascaded out are not given a regular n* name (they live in the
    # port pool and may surface later as label_net_*).
    active_nodes = set()
    for ts in touches_by_bbox.values():
        for t in ts:
            active_nodes.add(t["node_id"])

    # GND / VDD net merging — any node touching a gnd/vdd bbox folds in.
    # Include single-bbox (removed) nodes too: a wire that only reaches a ground
    # symbol is still ground, and a component pin may later rescue onto its line.
    gnd_nodes, vdd_nodes = set(), set()
    for b in bboxes:
        if b["cls"] == 0:
            for t in touches_by_bbox.get(b["idx"], []):
                gnd_nodes.add(t["node_id"])
            for t in single_by_bbox.get(b["idx"], []):
                gnd_nodes.add(t["node_id"])
        elif b["cls"] == 1:
            for t in touches_by_bbox.get(b["idx"], []):
                vdd_nodes.add(t["node_id"])
            for t in single_by_bbox.get(b["idx"], []):
                vdd_nodes.add(t["node_id"])

    # Proximity fold: a wire tip that lands within SUPPLY_SYMBOL_PROXIMITY_PX of a
    # ground/supply symbol bbox is connected to it even when its endpoint stopped
    # just outside the box (so it was never recorded as a strict touch and the
    # pin stayed a floating label net). Endpoint-based and tightly bounded so a
    # wire merely routed past the symbol is not folded in.
    def _endpoint_near_bbox(pt, bx, pad):
        px, py = pt
        x1, y1, x2, y2 = bx
        dx = max(x1 - px, 0.0, px - x2)
        dy = max(y1 - py, 0.0, py - y2)
        return dx * dx + dy * dy <= pad * pad

    supply_syms = [b for b in bboxes if b["cls"] in (0, 1)]
    if supply_syms and len(combined_lines):
        for li in range(len(combined_lines)):
            try:
                lnid = int(line_node_ids[li])
            except (IndexError, TypeError, ValueError):
                continue
            if lnid < 0:
                continue
            (ax, ay), (bx_, by_) = combined_lines[li]
            for b in supply_syms:
                bx = b["bbox"]
                if _endpoint_near_bbox((ax, ay), bx, SUPPLY_SYMBOL_PROXIMITY_PX) or \
                   _endpoint_near_bbox((bx_, by_), bx, SUPPLY_SYMBOL_PROXIMITY_PX):
                    (gnd_nodes if b["cls"] == 0 else vdd_nodes).add(lnid)
                    break

    # Assign final node names: GND, VDD, n0, n1, ...
    node_name = {}
    node_origins = {}
    if gnd_nodes:
        node_origins["GND"] = sorted(gnd_nodes)
    if vdd_nodes:
        node_origins["VDD"] = sorted(vdd_nodes)

    seq = 0
    for n in touch_data["nodes"]:
        nid = n["node_id"]
        if nid not in active_nodes:
            continue  # cascaded out — no regular name
        if nid in gnd_nodes:
            node_name[nid] = "GND"
        elif nid in vdd_nodes:
            node_name[nid] = "VDD"
        else:
            name = f"n{seq}"
            seq += 1
            node_name[nid] = name
            node_origins[name] = [nid]

    # Ground/supply nodes that only appear as single-bbox (removed) touches are
    # not in active_nodes, so name them here so net_for_line resolves their line
    # to GND/VDD instead of inventing a floating label net.
    for nid in gnd_nodes:
        node_name.setdefault(nid, "GND")
    for nid in vdd_nodes:
        node_name.setdefault(nid, "VDD")

    auto_port_counter = [0]

    def new_label_net(origin_node_ids):
        nm = f"label_net_{auto_port_counter[0]}"
        auto_port_counter[0] += 1
        node_origins[nm] = list(origin_node_ids)
        return nm

    def label_net_for_node(nid):
        """Return ONE shared label net per single-bbox / unnamed node id, so any
        two components that resolve to the SAME node (e.g. a single-bbox node
        feeding both a MOS gate and an amplifier pin) get the SAME net instead of
        each minting a separate floating label net. (labelnet_for_nid is also
        used by net_for_line, so both paths agree on the same name.)"""
        if nid in node_name:
            return node_name[nid]
        if nid not in labelnet_for_nid:
            labelnet_for_nid[nid] = new_label_net([nid])
        return labelnet_for_nid[nid]

    def lookup_port_candidates(bbox_idx, edge):
        """Return single-bbox node touches for `bbox_idx` on `edge`."""
        cands = [s for s in single_by_bbox.get(bbox_idx, []) if s["edge"] == edge]
        if not cands:
            return []
        return cands

    def lookup_port(bbox_idx, edge):
        """Return label_net name if a single-bbox node touches `bbox_idx` on `edge`."""
        cands = lookup_port_candidates(bbox_idx, edge)
        if not cands:
            return None
        nids = [c["node_id"] for c in cands]
        if len(set(nids)) == 1:
            return label_net_for_node(nids[0])
        return new_label_net(nids)

    def net_for_touch(touch):
        """Return a regular node name or a label net created from a port candidate."""
        if "_net" in touch:
            return touch["_net"]
        if "_label_net" in touch:
            return touch["_label_net"]
        return node_name[touch["node_id"]]

    # Cache label nets so that one physical wire maps to ONE net even when it
    # is rescued by several components (e.g. a gate line bridging two MOS gates
    # must give both gates the SAME net, not two floating label nets).
    labelnet_for_nid = {}  # node id (>=0) without a real node name -> label net
    labelnet_for_line = {}  # node-less line index -> label net

    def net_for_line(line_idx):
        if line_idx < 0 or line_idx >= len(line_node_ids):
            return new_label_net([])
        try:
            nid = int(line_node_ids[line_idx])
        except (TypeError, ValueError):
            nid = -1
        if nid in node_name:
            return node_name[nid]
        if nid >= 0:
            # Same node id (not a named node) -> one shared label net.
            if nid not in labelnet_for_nid:
                labelnet_for_nid[nid] = new_label_net([nid])
            return labelnet_for_nid[nid]
        # Node-less line: the same line index shares one label net across the
        # components that rescue from it; distinct lines stay distinct.
        if line_idx not in labelnet_for_line:
            labelnet_for_line[line_idx] = new_label_net([])
        return labelnet_for_line[line_idx]

    def net_via_nearby_junction(node_id, own_net, touch_line_idx=None):
        """For a pin left as a floating label net: if any line of its node
        reaches within JUNCTION_PROXIMITY_PX of a junction marker, return the
        real net that other lines at that same junction resolve to. A junction
        is an explicit "these wires connect" marker, so a stub that stops just
        short of one still belongs to the net merged there. The node-stage
        junction force-merge misses it only because the stub did not quite touch
        the junction bbox. Returns None if no clean single real net is found.

        When the pin's wire never coalesced into a node (node_id < 0, a bare
        node-less stub), fall back to probing the pin's own touch line
        (touch_line_idx) so a stub fragment that reaches a junction is still
        rescued."""
        if not junction_bboxes:
            return None
        pad = JUNCTION_PROXIMITY_PX

        def near(line, jb):
            for px, py in line:
                if jb[0] - pad <= px <= jb[2] + pad and jb[1] - pad <= py <= jb[3] + pad:
                    return True
            return False

        if node_id is not None and node_id >= 0:
            my_lines = [
                j
                for j in range(len(combined_lines))
                if j < len(line_node_ids) and int(line_node_ids[j]) == node_id
            ]
        elif touch_line_idx is not None and 0 <= int(touch_line_idx) < len(combined_lines):
            my_lines = [int(touch_line_idx)]
        else:
            return None
        if not my_lines:
            return None
        for jb in junction_bboxes:
            if not any(near(combined_lines[j], jb) for j in my_lines):
                continue
            counts = {}
            my_line_set = set(my_lines)
            for j in range(len(combined_lines)):
                if j in my_line_set:
                    continue
                if node_id is not None and node_id >= 0 and (
                    j < len(line_node_ids) and int(line_node_ids[j]) == node_id
                ):
                    continue
                if not near(combined_lines[j], jb):
                    continue
                net = net_for_line(j)
                if not isinstance(net, str) or net == own_net:
                    continue
                if net.startswith("label_net"):
                    continue
                counts[net] = counts.get(net, 0) + 1
            if counts:
                return max(counts, key=counts.get)
        return None

    def line_is_stub(j):
        """A combined line that does not belong to a real (named) net -- i.e. a
        node-less segment or a cascaded/removed node. These are the fragments of
        a broken wire that surface as label nets."""
        nidj = line_node_ids[j] if j < len(line_node_ids) else -1
        return nidj < 0 or nidj not in node_name

    def real_node_net_via_endpoint_extension(touch, current_net):
        """For a dangling (label_net) non-gate pin: walk the connected chain of
        node-less stub segments starting from the pin's touch line, then extend
        each DANGLING tip of that chain outward along its own direction and
        return the net of the nearest distinct REAL node line it reaches (or
        None). The pin's wire is often broken into several stub segments that
        turn a corner (e.g. a short jog off the device edge, then a long run
        toward the real node), so probing only the single touch line misses the
        real continuation. Each bridge must be inked. Scoped to broken pins."""
        li = touch.get("_nearby_line_idx")
        if li is None:
            li = touch.get("line_idx")
        if li is None:
            return None
        li = int(li)
        if li < 0 or li >= len(combined_lines):
            return None
        contact = touch.get("contact_xy")
        if contact is None:
            return None

        JOIN_TOL = 3.0
        # Connected component of stub segments reachable from the touch line via
        # shared endpoints. These are the fragments of this broken wire.
        comp = []
        seen = set()
        stack = [li] if line_is_stub(li) else []
        while stack and len(comp) < 40:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            comp.append(cur)
            ca, cb = combined_lines[cur]
            for j in range(len(combined_lines)):
                if j in seen or not line_is_stub(j):
                    continue
                oa, ob = combined_lines[j]
                if (
                    point_distance(ca, oa) <= JOIN_TOL
                    or point_distance(ca, ob) <= JOIN_TOL
                    or point_distance(cb, oa) <= JOIN_TOL
                    or point_distance(cb, ob) <= JOIN_TOL
                ):
                    stack.append(j)
        if not comp:
            comp = [li]

        # The endpoint nearest the device contact is where the chain attaches to
        # the device; never probe from it.
        all_ends = []
        for j in comp:
            a, b = combined_lines[j]
            all_ends.append(tuple(a))
            all_ends.append(tuple(b))
        attach = min(all_ends, key=lambda p: point_distance(p, contact))

        # For each segment, the dangling tip = the end FARTHER from the attach
        # point; probe outward along base->tip. Duplicate/parallel detections of
        # the same stub produce coincident tips, so dedupe within JOIN_TOL. This
        # handles both duplicates and corners (a short jog off the device edge
        # followed by a long run toward the real node).
        probes = []
        for j in comp:
            a, b = combined_lines[j]
            if point_distance(a, attach) >= point_distance(b, attach):
                tip, base = tuple(a), tuple(b)
            else:
                tip, base = tuple(b), tuple(a)
            if point_distance(tip, attach) <= JOIN_TOL:
                continue
            probes.append((tip, base, j))
        deduped = []
        for tip, base, seg in probes:
            if any(point_distance(tip, t2) <= JOIN_TOL for t2, _, _ in deduped):
                continue
            deduped.append((tip, base, seg))

        comp_set = set(comp)
        best = None
        for tip, base, seg in deduped:
            dx, dy = tip[0] - base[0], tip[1] - base[1]
            norm = (dx * dx + dy * dy) ** 0.5
            if norm < 1e-6:
                continue
            ux, uy = dx / norm, dy / norm
            tipl = list(tip)
            probe = [
                tipl,
                [tipl[0] + ux * EXTEND_PIN_PROBE_PX, tipl[1] + uy * EXTEND_PIN_PROBE_PX],
            ]
            for j, other in enumerate(combined_lines):
                if j in comp_set:
                    continue
                nidj = line_node_ids[j] if j < len(line_node_ids) else -1
                if nidj < 0 or nidj not in node_name:
                    continue
                netj = node_name[nidj]
                if netj == current_net:
                    continue
                dist, _p, _q = segment_segment_distance(probe, other)
                if dist <= EXTEND_PIN_PROBE_TOL_PX:
                    cp = closest_point_on_segment_to_point(tipl, other[0], other[1])
                    along = point_distance(tipl, cp)
                    # Accept the bridge when EITHER the gap is inked (HAWP kept a
                    # faint segment) OR the two stubs are collinear and face each
                    # other across a short gap (HAWP dropped the segment whole).
                    bridged = gap_is_inked(tipl, cp)
                    if not bridged:
                        # near/far ends of the target; near is the dangling tip
                        # the broken wire would reconnect to.
                        oa, ob = other
                        if point_distance(tipl, oa) <= point_distance(tipl, ob):
                            near, far = oa, ob
                        else:
                            near, far = ob, oa
                        gap = point_distance(tipl, near)
                        if gap <= ENDPOINT_BRIDGE_MAX_GAP_PX:
                            nd_ = point_distance(tipl, near)
                            tn = [
                                (near[0] - tipl[0]) / nd_,
                                (near[1] - tipl[1]) / nd_,
                            ] if nd_ > 1e-6 else None
                            od_ = point_distance(near, far)
                            on = [
                                (far[0] - near[0]) / od_,
                                (far[1] - near[1]) / od_,
                            ] if od_ > 1e-6 else None
                            if (
                                tn is not None
                                and on is not None
                                and (ux * tn[0] + uy * tn[1])
                                >= ENDPOINT_BRIDGE_COLLINEAR_COS
                                and (ux * on[0] + uy * on[1])
                                >= ENDPOINT_BRIDGE_COLLINEAR_COS
                            ):
                                bridged = True
                                along = gap
                    if not bridged:
                        continue
                    if best is None or along < best[0]:
                        best = (along, netj)
        return best[1] if best else None

    def gap_is_inked(p0, p1, min_frac=0.7, thr=128.0, band=2):
        # Sample a small perpendicular band around the straight path: the
        # detected stub can be 1-2 px off the real wire, so a single-pixel ray
        # would miss it. A point counts as inked if any pixel within `band` is
        # dark; a genuine white gap stays white even with the band.
        pts = bresenham_line(p0, p1)
        if len(pts) <= 2:
            return True
        width, height = image.size
        black = 0
        tot = 0
        for x, y in pts:
            tot += 1
            hit = False
            for dx in range(-band, band + 1):
                for dy in range(-band, band + 1):
                    xx, yy = x + dx, y + dy
                    if 0 <= xx < width and 0 <= yy < height and grayscale(
                        image.getpixel((xx, yy))
                    ) < thr:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                black += 1
        return tot > 0 and (black / tot) >= min_frac

    def nearby_line_touch_for_edge(
        bbox_idx,
        bbox,
        edge,
        existing_touches=None,
        max_gap=None,
        allow_existing_node=False,
        edge_center_weight=0.0,
    ):
        if not combined_lines:
            return None
        if max_gap is None:
            max_gap = NEARBY_LINE_PIN_RESCUE_PX

        existing_touches = existing_touches or []
        existing_node_ids = {
            t.get("node_id")
            for t in existing_touches
            if t.get("node_id") is not None and t.get("node_id") >= 0
        }
        candidates = []
        for line_idx, line in enumerate(combined_lines):
            try:
                nid = int(line_node_ids[line_idx])
            except (IndexError, TypeError, ValueError):
                nid = -1
            if not allow_existing_node and nid in existing_node_ids:
                continue

            hit = candidate_contact_for_line_edge(line, bbox, edge)
            if not hit:
                continue
            gap, contact, method = hit
            if gap > max_gap:
                continue
            # Prefer a line that belongs to a real anchored node (node_id >= 0)
            # over an unanchored fragment (node_id < 0): connecting a pin to an
            # unanchored line only mints an isolated label-net, whereas a nearby
            # anchored node is almost always the pin's true net. A slightly
            # farther rail line should therefore win over a closer dangling stub.
            unanchored_penalty = 0.0 if nid >= 0 else NEARBY_LINE_UNANCHORED_PENALTY
            score = (
                unanchored_penalty
                + gap
                + edge_center_weight * edge_center_distance(contact, bbox, edge)
            )
            candidates.append((score, gap, line_idx, nid, contact, method))

        if not candidates:
            return None
        _, gap, line_idx, nid, contact, method = sorted(candidates)[0]
        touch = {
            "node_id": nid,
            "edge": edge,
            "contact_xy": contact,
            "_net": net_for_line(line_idx),
            "_nearby_line_idx": line_idx,
            "_nearby_line_gap": gap,
            "_nearby_line_method": method,
        }
        return touch

    def ideal_gb_black_pixel_touch_for_edge(bbox, edge):
        x1, y1, x2, y2 = bbox
        radius = IDEAL_GB_BLACK_PIXEL_RESCUE_RADIUS_PX
        if edge in ("left", "right"):
            cx = x1 if edge == "left" else x2
            cy = 0.5 * (y1 + y2)
            y_start = int(round(cy)) - radius
            y_stop = int(round(cy)) + radius
            if edge == "left":
                x_start = int(round(x1)) - radius
                x_stop = int(round(x1))
            else:
                x_start = int(round(x2))
                x_stop = int(round(x2)) + radius
        else:
            cx = 0.5 * (x1 + x2)
            cy = y1 if edge == "top" else y2
            x_start = int(round(cx)) - radius
            x_stop = int(round(cx)) + radius
            if edge == "top":
                y_start = int(round(y1)) - radius
                y_stop = int(round(y1))
            else:
                y_start = int(round(y2))
                y_stop = int(round(y2)) + radius

        width, height = image.size
        threshold = bbox_black_threshold(image, bbox)
        candidates = []
        for yy in range(y_start, y_stop + 1):
            if yy < 0 or yy >= height:
                continue
            for xx in range(x_start, x_stop + 1):
                if xx < 0 or xx >= width:
                    continue
                g = grayscale(image.getpixel((xx, yy)))
                if g > threshold:
                    continue
                dist = ((xx - cx) ** 2 + (yy - cy) ** 2) ** 0.5
                candidates.append((dist, g, xx, yy))

        if not candidates:
            return None
        dist, g, xx, yy = sorted(candidates)[0]
        return {
            "node_id": -1,
            "edge": edge,
            "contact_xy": [float(xx), float(yy)],
            "_ideal_black_pixel_dist": dist,
            "_ideal_black_pixel_gray": g,
            "_ideal_black_pixel_threshold": threshold,
        }

    def add_nearby_line_touches(
        bbox_idx,
        bbox,
        current_touches,
        target_count,
        candidate_edges,
        flags,
        note_prefix,
        allow_reused_edges=False,
        reject_predicate=None,
    ):
        added = []
        touches = list(current_touches)
        used_edges = {t.get("edge") for t in touches}
        for edge in candidate_edges:
            if len(touches) >= target_count:
                break
            if not allow_reused_edges and edge in used_edges:
                continue
            rescue = nearby_line_touch_for_edge(bbox_idx, bbox, edge, touches)
            if not rescue:
                continue
            # Reject a rescued contact at a non-ideal position (e.g. a non-gate
            # pin landing on the gate side of a MOS): such a line is not the
            # real pin wire.
            if reject_predicate is not None and reject_predicate(rescue):
                continue
            touches.append(rescue)
            added.append(rescue)
            if not allow_reused_edges:
                used_edges.add(edge)
        if added:
            notes = [
                f"{t['edge']}:line{t['_nearby_line_idx']}:"
                f"{t.get('_nearby_line_method', 'line')}:"
                f"gap={t['_nearby_line_gap']:.1f}"
                for t in added
            ]
            flags.append(f"{note_prefix}_via_nearby_line=" + ",".join(notes))
        return touches

    def real_line_touch_near_label_touch(label_touch, edge, max_gap=None):
        if not combined_lines:
            return None
        if max_gap is None:
            max_gap = NEARBY_LINE_PIN_RESCUE_PX

        source_line_indices = []
        if label_touch.get("line_idx") is not None:
            source_line_indices.append(label_touch.get("line_idx"))
        for contributor in label_touch.get("contributors", []):
            if contributor.get("line_idx") is not None:
                source_line_indices.append(contributor.get("line_idx"))

        source_line_indices = [
            int(idx)
            for idx in dict.fromkeys(source_line_indices)
            if isinstance(idx, int) or str(idx).lstrip("-").isdigit()
        ]
        source_line_indices = [
            idx for idx in source_line_indices if 0 <= idx < len(combined_lines)
        ]
        if not source_line_indices:
            return None

        candidates = []
        for source_idx in source_line_indices:
            source_line = combined_lines[source_idx]
            for line_idx, line in enumerate(combined_lines):
                if line_idx == source_idx:
                    continue
                try:
                    nid = int(line_node_ids[line_idx])
                except (IndexError, TypeError, ValueError):
                    continue
                if nid not in node_name:
                    continue
                gap, _source_point, target_point = segment_segment_distance(
                    source_line, line
                )
                if gap > max_gap:
                    continue
                candidates.append((gap, line_idx, nid, target_point))

        if not candidates:
            return None
        gap, line_idx, nid, contact = sorted(candidates)[0]
        return {
            "node_id": nid,
            "edge": edge,
            "contact_xy": contact,
            "_net": node_name[nid],
            "_nearby_line_idx": line_idx,
            "_nearby_line_gap": gap,
            "_nearby_line_method": "label_stub",
        }

    def edge_contact_xy(bbox, edge, neighbor_bbox):
        x1, y1, x2, y2 = bbox
        nx1, ny1, nx2, ny2 = neighbor_bbox
        if edge in ("left", "right"):
            lo = max(y1, ny1)
            hi = min(y2, ny2)
            y = 0.5 * (lo + hi) if lo <= hi else 0.5 * (y1 + y2)
            x = x1 if edge == "left" else x2
        else:
            lo = max(x1, nx1)
            hi = min(x2, nx2)
            x = 0.5 * (lo + hi) if lo <= hi else 0.5 * (x1 + x2)
            y = y1 if edge == "top" else y2
        return [x, y]

    def bbox_intersects_edge_band(
        bbox, edge, neighbor_bbox, pad=CLOSE_BBOX_PIN_RESCUE_PX
    ):
        x1, y1, x2, y2 = bbox
        nx1, ny1, nx2, ny2 = neighbor_bbox
        # The neighbor must lie on the CORRECT side of the edge (its facing edge
        # near this edge), not merely overlap the edge level. Otherwise a
        # same-level side neighbor gets accepted for a top/bottom rescue (and
        # vice versa), e.g. grabbing the transistor next to it instead of the
        # supply symbol above it.
        if edge in ("left", "right"):
            edge_x = x1 if edge == "left" else x2
            # left  -> neighbor's right edge near this left edge (neighbor left)
            # right -> neighbor's left edge near this right edge (neighbor right)
            facing = nx2 if edge == "left" else nx1
            crosses_edge = (edge_x - pad) <= facing <= (edge_x + pad)
            overlaps_axis = ny1 <= y2 + pad and ny2 >= y1 - pad
            return crosses_edge and overlaps_axis
        edge_y = y1 if edge == "top" else y2
        # top    -> neighbor's bottom edge near this top edge (neighbor above)
        # bottom -> neighbor's top edge near this bottom edge (neighbor below)
        facing = ny2 if edge == "top" else ny1
        crosses_edge = (edge_y - pad) <= facing <= (edge_y + pad)
        overlaps_axis = nx1 <= x2 + pad and nx2 >= x1 - pad
        return crosses_edge and overlaps_axis

    def edge_neighbor_score(bbox, edge, neighbor_bbox):
        x1, y1, x2, y2 = bbox
        nx1, ny1, nx2, ny2 = neighbor_bbox
        if edge == "left":
            edge_gap = min(abs(nx2 - x1), abs(nx1 - x1))
            axis_gap = max(0, max(y1, ny1) - min(y2, ny2))
        elif edge == "right":
            edge_gap = min(abs(nx1 - x2), abs(nx2 - x2))
            axis_gap = max(0, max(y1, ny1) - min(y2, ny2))
        elif edge == "top":
            edge_gap = min(abs(ny2 - y1), abs(ny1 - y1))
            axis_gap = max(0, max(x1, nx1) - min(x2, nx2))
        else:
            edge_gap = min(abs(ny1 - y2), abs(ny2 - y2))
            axis_gap = max(0, max(x1, nx1) - min(x2, nx2))
        return (edge_gap, axis_gap)

    def processed_component_net_for_edge(other, other_edge, contact_xy):
        component = built_components_by_bbox.get(other["idx"])
        if not component:
            return None

        pins = component.get("pins", {})
        orient = component.get("orientation")
        cname = component.get("class", CLASS_NAMES.get(other["cls"], ""))
        cls = other["cls"]

        if cls in (2, 3, 4, 5, 6, 7) and orient in ("l", "r", "u", "d"):
            base_type = cname.replace("-bulk", "")
            layout = MOS_PIN_LAYOUT.get((base_type, orient))
            if not layout:
                return None
            input_pin, low_high = layout
            if other_edge == ORIENT_EDGE[orient]:
                return pins.get(input_pin)
            axis = perp_axis_idx(orient)
            x1, y1, x2, y2 = other["bbox"]
            mid = 0.5 * ((y1 + y2) if axis == 1 else (x1 + x2))
            return pins.get(low_high[0] if contact_xy[axis] <= mid else low_high[1])

        if cls == 11 and orient in ("l", "r", "u", "d"):
            k_edge = ORIENT_EDGE[orient]
            if other_edge == k_edge:
                return pins.get("K")
            if other_edge == OPPOSITE_EDGE[k_edge]:
                return pins.get("A")

        if cls in (12, 13, 14, 15):
            if other_edge in ("top", "left"):
                return pins.get("+")
            if other_edge in ("bottom", "right"):
                return pins.get("-")

        return None

    def close_bbox_net_for_edge(bbox_idx, bbox, edge):
        """Find a net supplied by a bbox touching `edge` within the rescue band."""
        candidates = []
        for other in bboxes:
            if other["idx"] == bbox_idx:
                continue
            if not bbox_intersects_edge_band(bbox, edge, other["bbox"]):
                continue

            net = None
            contact_xy = edge_contact_xy(bbox, edge, other["bbox"])
            if other["cls"] == 0:
                net = "GND"
            elif other["cls"] == 1:
                net = "VDD"
            else:
                other_edge = OPPOSITE_EDGE[edge]
                other_touches = [
                    t
                    for t in touches_by_bbox.get(other["idx"], [])
                    if t["edge"] == other_edge and t.get("node_id") in node_name
                ]
                if other_touches:
                    other_touches.sort(
                        key=lambda t: (
                            (
                                t["contact_xy"][0]
                                - edge_contact_xy(bbox, edge, other["bbox"])[0]
                            )
                            ** 2
                            + (
                                t["contact_xy"][1]
                                - edge_contact_xy(bbox, edge, other["bbox"])[1]
                            )
                            ** 2
                        )
                    )
                    net = net_for_touch(other_touches[0])
                else:
                    net = processed_component_net_for_edge(
                        other, other_edge, contact_xy
                    )

            if net is not None:
                candidates.append(
                    (
                        edge_neighbor_score(bbox, edge, other["bbox"]),
                        {
                            "net": net,
                            "neighbor_bbox_idx": other["idx"],
                            "contact_xy": contact_xy,
                        },
                    )
                )

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def rescue_touch_from_close_bbox(bbox_idx, bbox, edge):
        bridge = close_bbox_net_for_edge(bbox_idx, bbox, edge)
        if not bridge:
            return None
        return {
            "node_id": -1,
            "edge": edge,
            "contact_xy": bridge["contact_xy"],
            "_net": bridge["net"],
            "_neighbor_bbox_idx": bridge["neighbor_bbox_idx"],
        }

    def rescue_note_for_touch(note_prefix, edge, rescue):
        if "_neighbor_bbox_idx" in rescue:
            return (
                f"{note_prefix}_via_close_bbox_edge="
                f"{edge}:bbox{rescue['_neighbor_bbox_idx']}"
            )
        if "_nearby_line_idx" in rescue:
            return (
                f"{note_prefix}_via_nearby_line={edge}:"
                f"line{rescue['_nearby_line_idx']}:"
                f"{rescue.get('_nearby_line_method', 'line')}:"
                f"gap={rescue['_nearby_line_gap']:.1f}"
            )
        return f"{note_prefix}_rescued={edge}"

    def is_usable_rescue_touch(rescue, pins, pin_name=None):
        if not rescue:
            return False
        rescued_net = net_for_touch(rescue)
        if not isinstance(rescued_net, str):
            return False
        if rescued_net.startswith("label_net_") and "_neighbor_bbox_idx" not in rescue:
            return False
        for other_pin, other_net in pins.items():
            # The bulk pin (B) is an implicit supply default; a source/drain
            # legitimately shares that supply (e.g. pmos S=VDD with B=VDD), so it
            # must not block a real S/D rescue onto the same supply net.
            if other_pin == "B":
                continue
            if other_pin != pin_name and other_net == rescued_net:
                return False
        return True

    def rescue_touch_for_missing_edge(
        bbox_idx,
        bbox,
        edge,
        existing_touches,
        pins,
        pin_name,
        flags,
        note_prefix,
        label_touch=None,
    ):
        """Rescue a confirmed missing pin: label-stub merge, bbox, then line."""

        existing_touches = list(existing_touches or [])

        label_touches = []
        if label_touch:
            label_touches.append(label_touch)
        else:
            existing_nodes = {
                t.get("node_id")
                for t in existing_touches
                if t.get("node_id") is not None and t.get("node_id") >= 0
            }
            for cand in lookup_port_candidates(bbox_idx, edge):
                if cand.get("node_id") in existing_nodes:
                    continue
                port_touch = dict(cand)
                label_touches.append(port_touch)

        for lt in label_touches:
            rescue = real_line_touch_near_label_touch(lt, edge)
            if is_usable_rescue_touch(rescue, pins, pin_name):
                flags.append(rescue_note_for_touch(note_prefix, edge, rescue))
                return rescue

        rescue = rescue_touch_from_close_bbox(bbox_idx, bbox, edge)
        if is_usable_rescue_touch(rescue, pins, pin_name):
            flags.append(rescue_note_for_touch(note_prefix, edge, rescue))
            return rescue

        rescue = nearby_line_touch_for_edge(
            bbox_idx,
            bbox,
            edge,
            existing_touches,
            allow_existing_node=True,
            edge_center_weight=0.5,
        )
        if is_usable_rescue_touch(rescue, pins, pin_name):
            flags.append(rescue_note_for_touch(note_prefix, edge, rescue))
            return rescue

        if label_touches and label_touch is None:
            fallback = dict(label_touches[0])
            fallback["_label_net"] = label_net_for_node(fallback["node_id"])
            flags.append(f"{note_prefix}_via_single_bbox_port={edge}")
            return fallback

        return None

    def maybe_commit_rescue(original_red_flags, trial_pins, trial_red_flags):
        if not original_red_flags or trial_red_flags:
            return False
        return len(trial_pins) > 0

    def try_close_bbox_pin_rescue(
        cls, cname, b_idx, b_bbox, b_touches, orient, pins, flags, red_flags
    ):
        if not red_flags:
            return

        original_pins = dict(pins)
        original_flags = list(flags)
        original_red_flags = list(red_flags)

        def commit(note):
            for net in pins.values():
                if net in ("GND", "VDD"):
                    node_origins.setdefault(net, [])
            flags.append(note)
            red_flags.clear()

        # MOS/BJT G/B-side rescue.
        if cls in (2, 3, 4, 5, 6, 7) and orient in ("l", "r", "u", "d"):
            base_type = cname.replace("-bulk", "")
            g_pin_name, low_high = MOS_PIN_LAYOUT[(base_type, orient)]
            g_edge = ORIENT_EDGE[orient]
            if original_red_flags == ["g_touches=0_no_single_bbox"]:
                rescue = rescue_touch_from_close_bbox(b_idx, b_bbox, g_edge)
                if rescue:
                    pins[g_pin_name] = rescue["_net"]
                    trial_red_flags = []
                    if maybe_commit_rescue(original_red_flags, pins, trial_red_flags):
                        commit(
                            f"{g_pin_name}_pin_via_close_bbox_edge="
                            f"{g_edge}:bbox{rescue['_neighbor_bbox_idx']}"
                        )
                        return

            non_g_flags = [
                rf
                for rf in original_red_flags
                if rf.startswith("non_g_touches=") and "_expected=" in rf
            ]
            if len(original_red_flags) == 1 and non_g_flags:
                is_bulk_class = cname.endswith("-bulk")
                expected_ng = 3 if is_bulk_class else 2
                ng_touches = [
                    t for t in b_touches if not is_on_g_side(t, b_bbox, orient)
                ]
                if len(ng_touches) > 1:
                    ng_touches, _ = merge_close_same_node_touches(ng_touches)
                if len(ng_touches) + 1 == expected_ng:
                    for edge in ("left", "right", "top", "bottom"):
                        rescue = rescue_touch_from_close_bbox(b_idx, b_bbox, edge)
                        if not rescue:
                            continue
                        rescue_touch = dict(rescue)
                        if is_on_g_side(rescue_touch, b_bbox, orient):
                            continue
                        trial_ng = ng_touches + [rescue_touch]
                        axis = perp_axis_idx(orient)
                        ng_sorted = sorted(
                            trial_ng, key=lambda t: t["contact_xy"][axis]
                        )
                        pin_seq = (
                            [low_high[0], "B", low_high[1]]
                            if is_bulk_class
                            else low_high
                        )

                        pins.clear()
                        pins.update(original_pins)
                        for pin, t in zip(pin_seq, ng_sorted):
                            pins[pin] = net_for_touch(t)

                        trial_red_flags = []
                        if maybe_commit_rescue(
                            original_red_flags, pins, trial_red_flags
                        ):
                            commit(
                                "non_g_pin_via_close_bbox_edge="
                                f"{edge}:bbox{rescue['_neighbor_bbox_idx']}"
                            )
                            return

                pins.clear()
                pins.update(original_pins)

        # Diode missing opposite pin.
        elif (
            cls == 11
            and original_red_flags == ["diode_only_one_touch_no_port"]
            and len(b_touches) == 1
        ):
            k_edge = ORIENT_EDGE.get(orient)
            a_edge = OPPOSITE_EDGE.get(k_edge) if k_edge else None
            only = b_touches[0]
            missing_edge = OPPOSITE_EDGE[only["edge"]]
            rescue = rescue_touch_from_close_bbox(b_idx, b_bbox, missing_edge)
            if rescue:
                if missing_edge == k_edge:
                    pins["K"] = rescue["_net"]
                elif missing_edge == a_edge:
                    pins["A"] = rescue["_net"]
                else:
                    pins.setdefault("A" if "A" not in pins else "K", rescue["_net"])
                if "A" in pins and "K" in pins:
                    commit(
                        "diode_pin_via_close_bbox_edge="
                        f"{missing_edge}:bbox{rescue['_neighbor_bbox_idx']}"
                    )
                    return

        # Source missing opposite pin.
        elif (
            cls in (12, 13, 14, 15)
            and original_red_flags == ["src_only_one_touch_no_port"]
            and len(b_touches) == 1
        ):
            only = b_touches[0]
            if only["edge"] in ("top", "left"):
                missing_pin = "-"
            else:
                missing_pin = "+"
            missing_edge = OPPOSITE_EDGE[only["edge"]]
            rescue = rescue_touch_from_close_bbox(b_idx, b_bbox, missing_edge)
            if rescue:
                pins[missing_pin] = rescue["_net"]
                if "+" in pins and "-" in pins:
                    commit(
                        "src_pin_via_close_bbox_edge="
                        f"{missing_edge}:bbox{rescue['_neighbor_bbox_idx']}"
                    )
                    return

        # R / C / L / switch missing opposite pin.
        elif (
            cls in (8, 9, 10, 17)
            and original_red_flags == ["only_one_touch_no_port"]
            and len(b_touches) == 1
        ):
            only = b_touches[0]
            missing_edge = OPPOSITE_EDGE[only["edge"]]
            rescue = rescue_touch_from_close_bbox(b_idx, b_bbox, missing_edge)
            if rescue:
                pins["2"] = rescue["_net"]
                if "1" in pins and "2" in pins:
                    commit(
                        "pin2_via_close_bbox_edge="
                        f"{missing_edge}:bbox{rescue['_neighbor_bbox_idx']}"
                    )
                    return

        pins.clear()
        pins.update(original_pins)
        flags.clear()
        flags.extend(original_flags)
        red_flags.clear()
        red_flags.extend(original_red_flags)

    def label_net_pin_rescue(cls, cname, b_idx, b_bbox, pins, pin_touches, flags):
        """Replace non-input label nets with nearby real line nets when possible."""
        if not pins or not pin_touches:
            return

        def pins_to_check():
            if cls in (2, 3, 4, 5, 6, 7):
                base_type = cname.replace("-bulk", "")
                if base_type in ("nmos", "pmos"):
                    return ("D", "S")
                if base_type in ("npn", "pnp"):
                    return ("C", "E")
                return ()
            if cls == 11:
                return ("A", "K")
            if cls in (12, 13, 14, 15):
                return ("+", "-")
            if cls in (8, 9, 10, 17):
                return tuple(k for k in pins if k.startswith("pin") or k in ("1", "2"))
            return ()

        for pin in pins_to_check():
            net = pins.get(pin)
            touch = pin_touches.get(pin)
            if not isinstance(net, str) or not net.startswith("label_net_"):
                continue
            if not touch or "edge" not in touch:
                continue

            rescue = rescue_touch_for_missing_edge(
                b_idx,
                b_bbox,
                touch["edge"],
                list(pin_touches.values()),
                pins,
                pin,
                flags,
                f"{pin}_label_net",
                label_touch=touch,
            )
            if not rescue:
                continue

            pins[pin] = net_for_touch(rescue)
            pin_touches[pin] = rescue

    components = []
    built_components_by_bbox = {}
    image_flags = []
    net_alias_pairs = []
    type_counters = defaultdict(int)

    for b in bboxes:
        cls = b["cls"]
        if cls in (0, 1):
            continue  # gnd / vdd are net labels, not components
        cname = CLASS_NAMES.get(cls, f"class_{cls}")
        b_idx = b["idx"]
        b_bbox = b["bbox"]
        b_touches = touches_by_bbox.get(b_idx, [])
        orient = orientations.get(b_idx)
        flags = list(prepass_flags.get(b_idx, []))  # benign auto-adjustments

        # Orientation sanity check for MOS. A u/d device (gate on the top/bottom
        # edge, D and S brought out the other horizontal edge) is laid out wider
        # than tall. If the classifier called it u/d but the bbox is clearly
        # taller than wide, the gate is really on a vertical edge -> reclassify
        # to l/r. Choose the side whose half of the bbox holds FEWER touches as
        # the gate side (the gate is a lone contact; D/S share the opposite
        # side). Skip when ambiguous (equal split). Fixes D/S-order errors from
        # a mirrored u-orientation (MOS_PIN_LAYOUT flips D/S only for "u").
        if cls in (2, 3, 4, 5) and orient in ("u", "d") and len(b_touches) >= 2:
            x1, y1, x2, y2 = b_bbox
            w, h = x2 - x1, y2 - y1
            # Require clearly taller-than-wide (not merely near-square) so a
            # legitimately u/d but roughly square device is left alone.
            if h > w * 1.2:
                cx = 0.5 * (x1 + x2)
                left = sum(1 for t in b_touches if t["contact_xy"][0] < cx)
                right = len(b_touches) - left
                if left != right:
                    orient = "l" if left < right else "r"
                    flags.append(
                        f"orientation_ud_to_lr_by_shape={orient}"
                        f"(w={w:.0f},h={h:.0f},L={left},R={right})"
                    )

        # Two-terminal elements with too many touches: if some of the extra
        # touches land on the same node, they are the same wire reaching the
        # device at multiple points, not distinct pins. Collapse all touches
        # that share a node into a single touch (distance-independent) before
        # the per-class pin assignment so they are treated as one contact.
        if cls in TWO_PIN_CLASSES and len(b_touches) > 2:
            collapsed, n_merged = merge_close_same_node_touches(
                b_touches, max_dist=float("inf")
            )
            if n_merged:
                b_touches = collapsed
                flags.append(f"two_terminal_same_node_collapsed={n_merged}")
        red_flags = []  # image recognition errors needing human review
        pins = {}
        pin_touches = {}

        # ------- MOS / BJT (class 2..7) -------
        if cls in (2, 3, 4, 5, 6, 7):
            if orient not in ("l", "r", "u", "d"):
                red_flags.append("missing_orientation")
                for i, t in enumerate(b_touches):
                    pins[f"pin{i+1}"] = node_name[t["node_id"]]
            else:
                base_type = cname.replace("-bulk", "")
                is_bulk_class = cname.endswith("-bulk")

                g_touches = [t for t in b_touches if is_on_g_side(t, b_bbox, orient)]
                ng_touches = [
                    t for t in b_touches if not is_on_g_side(t, b_bbox, orient)
                ]
                merge_notes = []
                if len(g_touches) > 1:
                    g_touches, n_merged = merge_close_same_node_touches(g_touches)
                    if n_merged:
                        merge_notes.append(f"g_side_close_same_node_merged={n_merged}")
                ng_same_node_merge_count = 0
                if len(ng_touches) > 1:
                    ng_touches, n_merged = merge_close_same_node_touches(ng_touches)
                    if n_merged:
                        ng_same_node_merge_count = n_merged
                        merge_notes.append(f"non_g_close_same_node_merged={n_merged}")
                # A MOSFET's D/S (and a BJT's C/E) are distinct nets, so two non-G
                # touches sharing a node are usually one wire wrapping the symbol.
                # Collapse them ONLY when a distinct rescue candidate (a single-bbox
                # port on a non-G edge, on another node) is available to supply the
                # real second pin -- otherwise the terminals may be genuinely
                # shorted (e.g. a diode-connected device) and must be kept as-is.
                if (
                    len(ng_touches) >= 2
                    and len({t["node_id"] for t in ng_touches}) == 1
                ):
                    ng_node = ng_touches[0]["node_id"]
                    has_distinct_port = any(
                        c["node_id"] != ng_node
                        and not is_on_g_side(c, b_bbox, orient)
                        for c in single_by_bbox.get(b_idx, [])
                    )
                    if has_distinct_port:
                        ng_touches = [ng_touches[0]]
                        merge_notes.append("non_g_wire_wrap_collapsed_for_rescue")
                if merge_notes:
                    flags.extend(merge_notes)
                g_wrong, ng_wrong, _, _, _ = mos_bjt_count_status(
                    cname, g_touches, ng_touches
                )

                # Gate/base contact position used by the short checks below. Use a
                # real gate touch if present, otherwise fall back to a single-bbox
                # port on the gate edge, then a last-resort ideal-edge black pixel,
                # so a gate/base that was only recovered as a rescued label-net
                # still participates in short detection.
                ideal_gb_rescue_touch = None
                g_anchor_xy = None
                if g_touches:
                    g_anchor_xy = g_touches[0]["contact_xy"]
                elif orient in ORIENT_EDGE:
                    g_ports = lookup_port_candidates(b_idx, ORIENT_EDGE[orient])
                    if g_ports:
                        g_anchor_xy = g_ports[0]["contact_xy"]
                    else:
                        ideal_gb_rescue_touch = ideal_gb_black_pixel_touch_for_edge(
                            b_bbox, ORIENT_EDGE[orient]
                        )
                        if ideal_gb_rescue_touch is not None:
                            g_anchor_xy = ideal_gb_rescue_touch["contact_xy"]

                expected_ng_before_reclass = 3 if is_bulk_class else 2
                ideal_geom = mos_bjt_ideal_geometry(b_bbox, orient)
                mos_non_g_short_from_three_touches = False
                mos_extra_short_touch = None
                mos_middle_non_g_line_touch = None
                bjt_non_g_short_from_three_touches = False
                bjt_extra_short_touch = None

                if (
                    (not is_bulk_class)
                    and base_type in ("nmos", "pmos")
                    and g_anchor_xy is not None
                    and len(ng_touches) == 3
                ):
                    axis = perp_axis_idx(orient)
                    trial_sorted = sorted(
                        ng_touches, key=lambda t: t["contact_xy"][axis]
                    )
                    if black_line_to_touch(
                        image,
                        g_anchor_xy,
                        trial_sorted[1],
                        b_bbox,
                    ):
                        mos_non_g_short_from_three_touches = True
                        mos_extra_short_touch = trial_sorted[1]
                        flags.append("mos_gate_to_extra_touch_black_line_short")

                if (
                    (not is_bulk_class)
                    and base_type in ("nmos", "pmos")
                    and len(ng_touches) == 3
                    and not mos_non_g_short_from_three_touches
                ):
                    axis = perp_axis_idx(orient)
                    trial_sorted = sorted(
                        ng_touches, key=lambda t: t["contact_xy"][axis]
                    )
                    middle_touch = trial_sorted[1]
                    non_g_edge = OPPOSITE_EDGE[ORIENT_EDGE[orient]]
                    has_source_line = (
                        middle_touch.get("line_idx") is not None
                        or any(
                            c.get("line_idx") is not None
                            for c in middle_touch.get("contributors", [])
                        )
                    )
                    if middle_touch.get("edge") == non_g_edge and has_source_line:
                        mos_middle_non_g_line_touch = middle_touch
                        flags.append("mos_middle_non_g_line_touch_dropped")

                if len(g_touches) > 1:
                    g_touches = prune_text_overlap_excess(
                        g_touches,
                        1,
                        text_bboxes,
                        flags,
                        "g_side",
                    )
                if (
                    len(ng_touches) > expected_ng_before_reclass
                    and not mos_non_g_short_from_three_touches
                    and mos_middle_non_g_line_touch is None
                ):
                    ng_touches = prune_text_overlap_excess(
                        ng_touches,
                        expected_ng_before_reclass,
                        text_bboxes,
                        flags,
                        "non_g",
                    )

                bulk_bs_short_from_two_touches = (
                    is_bulk_class
                    and base_type in ("nmos", "pmos")
                    and len(ng_touches) == 2
                )
                if (
                    len(ng_touches) < expected_ng_before_reclass
                    and not bulk_bs_short_from_two_touches
                ):
                    # A u/d MOS brings BOTH D and S out the same edge. When both
                    # legitimately land on one node (e.g. a MOS-cap with D=S tied
                    # to a rail), the same-node merge above collapses the two pins
                    # into one touch and leaves the device short. Restore the
                    # missing pin(s) as the SAME real node (a D=S short) rather
                    # than rescuing a spurious net -- but only when a non-G merge
                    # actually collapsed pins and every surviving touch is on a
                    # real (named) node.
                    if (
                        base_type in ("nmos", "pmos")
                        and orient in ("u", "d")
                        and ng_same_node_merge_count > 0
                        and len(ng_touches) >= 1
                        and all(
                            isinstance(t.get("node_id"), int) and t["node_id"] >= 0
                            for t in ng_touches
                        )
                    ):
                        src = ng_touches[-1]
                        while len(ng_touches) < expected_ng_before_reclass:
                            ng_touches.append(dict(src))
                        flags.append("mos_ud_non_g_same_node_pins_restored")
                    missing = expected_ng_before_reclass - len(ng_touches)
                    axis = perp_axis_idx(orient)
                    port_cands = [
                        c
                        for c in single_by_bbox.get(b_idx, [])
                        if not is_on_g_side(c, b_bbox, orient)
                    ]
                    existing_port_nodes = {
                        t["node_id"] for t in ng_touches if "_label_net" in t
                    }
                    port_cands = [
                        c for c in port_cands if c["node_id"] not in existing_port_nodes
                    ]
                    ideal_edges = (
                        list(ideal_geom["non_g_edges"])
                        if ideal_geom is not None
                        else []
                    )
                    added_ports = []
                    skipped_ports = 0
                    if ideal_edges:
                        missing_edges = list(ideal_edges[:missing])
                        used_port_ids = set()
                        chosen_ports = []
                        for edge in missing_edges:
                            edge_cands = [
                                c
                                for c in port_cands
                                if id(c) not in used_port_ids and c.get("edge") == edge
                            ]
                            if not edge_cands:
                                skipped_ports += 1
                                continue
                            cand = min(
                                edge_cands,
                                key=lambda c: edge_match_cost(c, edge, b_bbox),
                            )
                            used_port_ids.add(id(cand))
                            chosen_ports.append(cand)
                        port_cands = chosen_ports
                    else:
                        port_cands.sort(key=lambda c: c["contact_xy"][axis])
                        port_cands = port_cands[:missing]

                    for cand in port_cands[:missing]:
                        port_touch = dict(cand)
                        port_touch["_label_net"] = label_net_for_node(cand["node_id"])
                        ng_touches.append(port_touch)
                        added_ports.append(str(cand["edge"]))
                    if added_ports:
                        flags.append(
                            "non_g_pin_via_single_bbox_port=" + ",".join(added_ports)
                        )
                    if skipped_ports:
                        flags.append(
                            "non_g_single_bbox_port_skipped_by_ideal_position="
                            f"{skipped_ports}"
                        )
                    if len(ng_touches) < expected_ng_before_reclass:
                        g_edge = ORIENT_EDGE[orient]
                        present_edges = {t["edge"] for t in ng_touches}
                        preferred_edges = [
                            e
                            for e in ("left", "right", "top", "bottom")
                            if e != g_edge and e not in present_edges
                        ]
                        fallback_edges = [
                            e for e in ("left", "right", "top", "bottom") if e != g_edge
                        ]
                        rescue_edges = ideal_edges + preferred_edges + fallback_edges
                        allow_reused_rescue_edges = len(set(ideal_edges)) < len(
                            ideal_edges
                        )
                        # A non-gate pin should not be rescued onto the gate's
                        # own net (that would invent a gate-source/drain short).
                        g_nets = {net_for_touch(t) for t in g_touches}
                        close_bbox_added = []
                        used_edges = {t.get("edge") for t in ng_touches}
                        for edge in rescue_edges:
                            if len(ng_touches) >= expected_ng_before_reclass:
                                break
                            if (
                                not allow_reused_rescue_edges
                                and edge in used_edges
                            ):
                                continue
                            rescue = rescue_touch_from_close_bbox(b_idx, b_bbox, edge)
                            if (
                                not rescue
                                or is_on_g_side(rescue, b_bbox, orient)
                                or net_for_touch(rescue) in g_nets
                            ):
                                continue
                            ng_touches.append(rescue)
                            close_bbox_added.append(rescue)
                            if not allow_reused_rescue_edges:
                                used_edges.add(edge)
                        if close_bbox_added:
                            flags.append(
                                "non_g_pin_via_close_bbox_edge="
                                + ",".join(
                                    f"{t['edge']}:bbox{t['_neighbor_bbox_idx']}"
                                    for t in close_bbox_added
                                )
                            )
                        ng_touches = add_nearby_line_touches(
                            b_idx,
                            b_bbox,
                            ng_touches,
                            expected_ng_before_reclass,
                            rescue_edges,
                            flags,
                            "non_g_pin",
                            allow_reused_edges=allow_reused_rescue_edges,
                            reject_predicate=lambda t: is_on_g_side(
                                t, b_bbox, orient
                            )
                            or net_for_touch(t) in g_nets,
                        )

                # Auto-reclassify nmos/pmos -> *-bulk ONLY if 3 non-G touches are on 3
                # distinct nodes. If they're on 2 distinct nodes, the duplicate one is
                # a wire-wrap touch and the device stays as a normal 2-pin non-G MOS.
                # If all 3 are the same node, it's a full wire wrap; same handling.
                kept_non_bulk_3 = False
                if (
                    (not is_bulk_class)
                    and base_type in ("nmos", "pmos")
                    and len(ng_touches) == 3
                    and not mos_non_g_short_from_three_touches
                    and mos_middle_non_g_line_touch is None
                ):
                    n_distinct = len({t["node_id"] for t in ng_touches})
                    if n_distinct == 3:
                        cname = base_type + "-bulk"
                        is_bulk_class = True
                        flags.append("reclassified_to_bulk")
                    else:
                        kept_non_bulk_3 = True
                        flags.append(
                            f"non_g_touches=3_distinct_nodes={n_distinct}_treated_as_wire_wrap"
                        )
                elif base_type in ("npn", "pnp") and len(ng_touches) == 3:
                    axis = perp_axis_idx(orient)
                    trial_sorted = sorted(
                        ng_touches, key=lambda t: t["contact_xy"][axis]
                    )
                    if g_anchor_xy is not None and black_line_to_touch(
                        image,
                        g_anchor_xy,
                        trial_sorted[1],
                        b_bbox,
                    ):
                        bjt_non_g_short_from_three_touches = True
                        bjt_extra_short_touch = trial_sorted[1]
                        flags.append("bjt_base_to_extra_touch_black_line_short")

                if ideal_geom is not None and len(g_touches) > 1:
                    chosen_g = min(
                        g_touches,
                        key=lambda t: point_distance(
                            t["contact_xy"], ideal_geom["g"]
                        ),
                    )
                    dropped = len(g_touches) - 1
                    g_touches = [chosen_g]
                    flags.append(
                        "g_side_extra_touches_dropped_by_ideal_position="
                        f"{dropped}"
                    )

                # Gate wire passing THROUGH the body when there are >2 non-G
                # touches: the exit touch (opposite edge, aligned with the gate,
                # joined by a black line) is the gate wire continuing, not a real
                # pin. Capture its net BEFORE pruning drops it, then merge it into
                # the gate net after pin assignment. (The 2-touch case is handled
                # separately by mos_passthrough_touch below.)
                mos_through_extra_net = None
                if (
                    base_type in ("nmos", "pmos")
                    and g_anchor_xy is not None
                    and len(ng_touches) > 2
                    and not mos_non_g_short_from_three_touches
                    and not kept_non_bulk_3
                    and mos_middle_non_g_line_touch is None
                ):
                    opp_edge = OPPOSITE_EDGE[ORIENT_EDGE[orient]]
                    pax = perp_axis_idx(orient)
                    for t in ng_touches:
                        if t["edge"] != opp_edge:
                            continue
                        # A merged touch averages its contributors' contacts, so
                        # test each real contact for alignment AND a black line
                        # rather than the (possibly off-conductor) midpoint.
                        cand_contacts = [t["contact_xy"]] + [
                            c["contact_xy"]
                            for c in t.get("contributors", [])
                            if c.get("contact_xy") is not None
                        ]
                        if any(
                            abs(cxy[pax] - g_anchor_xy[pax]) <= MOS_PASSTHROUGH_ALIGN_PX
                            and is_black_line_between(image, g_anchor_xy, cxy, b_bbox)
                            for cxy in cand_contacts
                        ):
                            mos_through_extra_net = net_for_touch(t)
                            break

                if (
                    ideal_geom is not None
                    and len(ng_touches) > 2
                    and not mos_non_g_short_from_three_touches
                    and mos_middle_non_g_line_touch is None
                    and not bjt_non_g_short_from_three_touches
                ):
                    ds_chosen = best_touch_edge_match(
                        ng_touches,
                        ideal_geom["non_g_edges"],
                        b_bbox,
                    )
                    if ds_chosen:
                        chosen = list(ds_chosen)
                        remaining = [t for t in ng_touches if t not in chosen]
                        if is_bulk_class and remaining:
                            bulk_touch = min(
                                remaining,
                                key=lambda t: point_distance(
                                    t["contact_xy"], ideal_geom["bulk"]
                                ),
                            )
                            bulk_dist = point_distance(
                                bulk_touch["contact_xy"], ideal_geom["bulk"]
                            )
                            if bulk_dist <= bulk_touch_max_ideal_distance(b_bbox):
                                chosen.append(bulk_touch)
                            else:
                                flags.append(
                                    "bulk_touch_dropped_by_bad_ideal_position="
                                    f"dist={bulk_dist:.1f}"
                                )
                        dropped = len(ng_touches) - len(chosen)
                        if dropped > 0:
                            ng_touches = chosen
                            flags.append(
                                "non_g_extra_touches_dropped_by_ideal_position="
                                f"{dropped}"
                            )
                            bulk_bs_short_from_two_touches = (
                                is_bulk_class
                                and base_type in ("nmos", "pmos")
                                and len(ng_touches) == 2
                            )

                expected_ng = 3 if is_bulk_class else 2
                if (
                    len(ng_touches) != expected_ng
                    and not kept_non_bulk_3
                    and not mos_non_g_short_from_three_touches
                    and mos_middle_non_g_line_touch is None
                    and not bjt_non_g_short_from_three_touches
                    and not bulk_bs_short_from_two_touches
                ):
                    red_flags.append(
                        f"non_g_touches={len(ng_touches)}_expected={expected_ng}"
                    )

                g_pin_name, low_high = MOS_PIN_LAYOUT[(base_type, orient)]

                # G-side assignment
                if len(g_touches) == 1:
                    pins[g_pin_name] = net_for_touch(g_touches[0])
                    pin_touches[g_pin_name] = g_touches[0]
                elif len(g_touches) == 0:
                    port_cands = lookup_port_candidates(b_idx, ORIENT_EDGE[orient])
                    if port_cands:
                        nids = [c["node_id"] for c in port_cands]
                        if len(set(nids)) == 1:
                            port = label_net_for_node(nids[0])
                        else:
                            port = new_label_net(nids)
                        pins[g_pin_name] = port
                        pin_touches[g_pin_name] = port_cands[0]
                        flags.append("g_pin_via_single_bbox_port")
                    else:
                        rescue = nearby_line_touch_for_edge(
                            b_idx,
                            b_bbox,
                            ORIENT_EDGE[orient],
                            g_touches + ng_touches,
                            allow_existing_node=True,
                            edge_center_weight=0.5,
                        )
                        if rescue:
                            pins[g_pin_name] = net_for_touch(rescue)
                            pin_touches[g_pin_name] = rescue
                            g_touches.append(rescue)
                            flags.append(
                                f"g_pin_via_nearby_line={ORIENT_EDGE[orient]}:"
                                f"line{rescue['_nearby_line_idx']}:"
                                f"{rescue.get('_nearby_line_method', 'line')}:"
                                f"gap={rescue['_nearby_line_gap']:.1f}"
                            )
                        else:
                            if ideal_gb_rescue_touch is not None:
                                rescue = dict(ideal_gb_rescue_touch)
                                rescue["_label_net"] = new_label_net([])
                                pins[g_pin_name] = net_for_touch(rescue)
                                pin_touches[g_pin_name] = rescue
                                g_touches.append(rescue)
                                flags.append(
                                    f"g_pin_via_ideal_edge_black_pixel="
                                    f"{ORIENT_EDGE[orient]}:"
                                    f"xy={rescue['contact_xy'][0]:.0f},"
                                    f"{rescue['contact_xy'][1]:.0f}:"
                                    f"dist={rescue['_ideal_black_pixel_dist']:.1f}:"
                                    f"gray={rescue['_ideal_black_pixel_gray']:.1f}:"
                                    f"threshold={rescue['_ideal_black_pixel_threshold']:.1f}"
                                )
                            else:
                                red_flags.append("g_touches=0_no_single_bbox")
                else:
                    red_flags.append(f"g_touches={len(g_touches)}_expected=1")
                    pins[g_pin_name] = net_for_touch(g_touches[0])
                    pin_touches[g_pin_name] = g_touches[0]

                # A wire drawn straight through the MOS body shows up as a non-G
                # touch on the edge opposite the gate, aligned with the gate
                # contact on the perpendicular axis and joined to it by a
                # continuous black line. That touch is the gate wire passing
                # through the symbol, not a real source/drain pin: the side it
                # exits has no real contact, so it must not be read as a pin.
                mos_passthrough_touch = None
                if (
                    base_type in ("nmos", "pmos")
                    and not is_bulk_class
                    and not kept_non_bulk_3
                    and not mos_non_g_short_from_three_touches
                    and len(ng_touches) == 2
                    and g_pin_name in pin_touches
                ):
                    gate_touch = pin_touches[g_pin_name]
                    opp_edge = OPPOSITE_EDGE[ORIENT_EDGE[orient]]
                    align_axis = perp_axis_idx(orient)
                    for t in ng_touches:
                        if t["edge"] != opp_edge:
                            continue
                        t_node = t.get("node_id")
                        gate_node = gate_touch.get("node_id")
                        if (
                            t_node is not None
                            and gate_node is not None
                            and t_node >= 0
                            and gate_node >= 0
                            and t_node == gate_node
                        ):
                            continue
                        g_xy = gate_touch["contact_xy"]
                        cand_contacts = [t["contact_xy"]] + [
                            c["contact_xy"]
                            for c in t.get("contributors", [])
                            if c.get("contact_xy") is not None
                        ]
                        if any(
                            abs(cxy[align_axis] - g_xy[align_axis])
                            <= MOS_PASSTHROUGH_ALIGN_PX
                            and is_black_line_between(image, g_xy, cxy, b_bbox)
                            for cxy in cand_contacts
                        ):
                            mos_passthrough_touch = t
                            break

                # Non-G side assignment (sorted by perpendicular axis ascending)
                axis = perp_axis_idx(orient)
                ng_sorted = sorted(ng_touches, key=lambda t: t["contact_xy"][axis])

                if is_bulk_class:
                    if bulk_bs_short_from_two_touches:
                        pin_seq = [low_high[0], low_high[1]]
                        chosen = ng_sorted
                    else:
                        pin_seq = [low_high[0], "B", low_high[1]]
                        chosen = ng_sorted
                elif kept_non_bulk_3:
                    # Drop the wire-wrap middle touch; keep extremes.
                    pin_seq = [low_high[0], low_high[1]]
                    chosen = [ng_sorted[0], ng_sorted[-1]]
                elif mos_non_g_short_from_three_touches:
                    # A drawn wire crosses the MOS symbol from gate to the
                    # middle non-G touch; keep the outer contacts as pins and
                    # merge the gate and middle-touch nets below.
                    pin_seq = [low_high[0], low_high[1]]
                    chosen = [ng_sorted[0], ng_sorted[-1]]
                elif mos_middle_non_g_line_touch is not None:
                    # A real line touches the middle of the non-G side. Treat it
                    # as a side/body contact and keep the outer contacts as D/S,
                    # independent of whether that middle net is named GND/VDD.
                    pin_seq = [low_high[0], low_high[1]]
                    chosen = [ng_sorted[0], ng_sorted[-1]]
                elif bjt_non_g_short_from_three_touches:
                    # A drawn wire crosses the BJT symbol between C/E; keep the
                    # outer contacts as pins and merge their nets below.
                    pin_seq = [low_high[0], low_high[1]]
                    chosen = [ng_sorted[0], ng_sorted[-1]]
                else:
                    pin_seq = [low_high[0], low_high[1]]
                    chosen = ng_sorted

                for pin, t in zip(pin_seq, chosen):
                    pins[pin] = net_for_touch(t)
                    pin_touches[pin] = t

                if mos_passthrough_touch is not None:
                    # The pass-through touch's side carries the gate wire, not a
                    # real S/D contact. The pin assigned that touch therefore has
                    # no real contact -- but the device's actual D/S pin on the
                    # proper edge may simply have gone undetected (e.g. a short
                    # source stub absorbed into a thick rail). Before falling back
                    # to a dangling net, try to rescue this pin from a nearby line
                    # on the ideal D/S edge that the other (real) pin did not use.
                    # Only then merge the gate wire's net into the gate net.
                    g_net_here = pins.get(g_pin_name)
                    for pin, t in zip(pin_seq, chosen):
                        if t is mos_passthrough_touch:
                            through_net = net_for_touch(mos_passthrough_touch)
                            other_pin = (
                                pin_seq[0] if pin is pin_seq[1] else pin_seq[1]
                            )
                            used_edge = (pin_touches.get(other_pin) or {}).get("edge")
                            rescued = None
                            if ideal_geom is not None:
                                kept_touches = g_touches + [
                                    x for x in chosen if x is not mos_passthrough_touch
                                ]
                                for cand_edge in ideal_geom["non_g_edges"]:
                                    if cand_edge == used_edge:
                                        continue
                                    rc = nearby_line_touch_for_edge(
                                        b_idx,
                                        b_bbox,
                                        cand_edge,
                                        kept_touches,
                                        allow_existing_node=True,
                                    )
                                    if (
                                        rc is not None
                                        and not is_on_g_side(rc, b_bbox, orient)
                                        and net_for_touch(rc) != g_net_here
                                        and net_for_touch(rc) != through_net
                                    ):
                                        rescued = rc
                                        break
                            if rescued is not None:
                                pins[pin] = net_for_touch(rescued)
                                pin_touches[pin] = rescued
                                flags.append(
                                    "mos_passthrough_pin_rescued_via_nearby_line="
                                    f"{rescued['edge']}:line{rescued['_nearby_line_idx']}:"
                                    f"gap={rescued['_nearby_line_gap']:.1f}"
                                )
                            else:
                                pins[pin] = new_label_net([])
                                pin_touches.pop(pin, None)
                            net_alias_pairs.append((pins[g_pin_name], through_net))
                            flags.append("mos_gate_wire_through_body_merged")
                            break

                # >2-touch pass-through: the exit net was pruned from D/S, so just
                # merge it into the gate net (skip if it is one of this device's
                # own pins, to avoid inventing a gate-source/drain short).
                if (
                    mos_through_extra_net is not None
                    and g_pin_name in pins
                    and mos_through_extra_net != pins[g_pin_name]
                    and mos_through_extra_net
                    not in {
                        pins.get(low_high[0]),
                        pins.get(low_high[1]),
                        pins.get("B"),
                    }
                ):
                    net_alias_pairs.append((pins[g_pin_name], mos_through_extra_net))
                    flags.append("mos_gate_wire_through_body_merged")

                if bulk_bs_short_from_two_touches and "S" in pins:
                    pins["B"] = pins["S"]
                    pin_touches["B"] = pin_touches["S"]
                    flags.append("bulk_non_g_touches=2_B_short_to_S")

                if mos_non_g_short_from_three_touches:
                    if mos_extra_short_touch is not None and g_pin_name in pins:
                        net_alias_pairs.append(
                            (pins[g_pin_name], net_for_touch(mos_extra_short_touch))
                        )
                        flags.append(f"{g_pin_name}_extra_touch_short_merged")

                if bjt_non_g_short_from_three_touches:
                    if bjt_extra_short_touch is not None and g_pin_name in pins:
                        net_alias_pairs.append(
                            (pins[g_pin_name], net_for_touch(bjt_extra_short_touch))
                        )
                        flags.append(f"{g_pin_name}_extra_touch_short_merged")

                # An adjacent MOS's drain/source wire can run straight into the
                # SIDE of this MOS and join one of its D/S pins at a T-junction
                # hidden under this bbox's mask. That wire is seen here only as a
                # dropped "middle non-G line touch", which splits one real net
                # (the shared output node) into two. If the dropped middle line
                # is joined to exactly ONE kept D/S pin by a straight black line
                # -- directly, or by dropping from the pin along the channel axis
                # onto the middle line -- merge their nets. Requiring a single
                # connected pin keeps it from shorting D to S.
                if (
                    mos_middle_non_g_line_touch is not None
                    and base_type in ("nmos", "pmos")
                ):
                    mt = mos_middle_non_g_line_touch
                    mid_net = net_for_touch(mt)
                    mx, my = mt["contact_xy"]
                    connected = []
                    for pin in (pin_seq[0], pin_seq[-1]):
                        pt = pin_touches.get(pin)
                        pnet = pins.get(pin)
                        if pt is None or pnet is None or pnet == mid_net:
                            continue
                        px, py = pt["contact_xy"]
                        aligned = [px, my] if orient in ("l", "r") else [mx, py]
                        if black_line_to_touch(
                            image, pt["contact_xy"], mt, b_bbox
                        ) or is_black_line_between(
                            image, pt["contact_xy"], aligned, b_bbox
                        ):
                            connected.append((pin, pnet))
                    if len(connected) == 1:
                        pin, pnet = connected[0]
                        net_alias_pairs.append((pnet, mid_net))
                        flags.append(f"mos_middle_non_g_line_merged_to_{pin}")
                    elif len(connected) >= 2:
                        flags.append("mos_middle_non_g_line_merge_ambiguous_skipped")

                # Scoped rescue for a broken non-gate connection: if a D/S
                # (or C/E) pin is still a dangling label_net, extend its stub
                # tip outward and adopt the real node it reaches. Only touches
                # already-broken pins, so it cannot disturb correct cases.
                for pin in list(pins.keys()):
                    if pin == g_pin_name or pin == "B":
                        continue
                    net = pins.get(pin)
                    if not (isinstance(net, str) and net.startswith("label_net")):
                        continue
                    t = pin_touches.get(pin)
                    if t is None:
                        continue
                    found = real_node_net_via_endpoint_extension(t, net)
                    if found:
                        pins[pin] = found
                        flags.append(
                            f"{pin}_label_net_extended_to_node={found}"
                        )

                # Junction-proximity rescue: a pin still a floating label net
                # whose stub reaches a junction marker adopts the real net merged
                # at that junction. Covers the GATE too (e.g. a diode-connected
                # MOS whose gate ties to its drain through a junction); the
                # node-stage force-merge missed it only because the stub stopped
                # just short of the junction bbox.
                for pin in list(pins.keys()):
                    # "B" is the MOS *bulk* pin (defaults to GND/VDD, not a
                    # junction-routed signal) -- skip it for MOS. But for a BJT
                    # "B" is the *base*, a real signal pin that can tie to a net
                    # through a junction, so it stays eligible.
                    if pin == "B" and base_type in ("nmos", "pmos"):
                        continue
                    net = pins.get(pin)
                    if not (isinstance(net, str) and net.startswith("label_net")):
                        continue
                    t = pin_touches.get(pin)
                    if t is None:
                        continue
                    found = net_via_nearby_junction(
                        t.get("node_id"),
                        net,
                        t.get("line_idx", t.get("_nearby_line_idx")),
                    )
                    if found:
                        pins[pin] = found
                        flags.append(f"{pin}_label_net_via_junction={found}")

                # Default bulk for plain MOS
                if not is_bulk_class:
                    if base_type == "nmos":
                        pins.setdefault("B", "GND")
                    elif base_type == "pmos":
                        pins.setdefault("B", "VDD")
                elif (
                    base_type in ("nmos", "pmos")
                    and "G" in pins
                    and "B" in pins
                    and "G" in pin_touches
                    and "B" in pin_touches
                    and is_black_line_between(
                        image,
                        pin_touches["G"]["contact_xy"],
                        pin_touches["B"]["contact_xy"],
                        b_bbox,
                    )
                ):
                    net_alias_pairs.append((pins["G"], pins["B"]))
                    flags.append("gate_bulk_short_merged")

        # ------- Diode (class 11) -------
        elif cls == 11:
            if orient not in ("l", "r", "u", "d"):
                red_flags.append("missing_orientation")
            k_edge = ORIENT_EDGE.get(orient)
            a_edge = OPPOSITE_EDGE.get(k_edge) if k_edge else None
            if len(b_touches) > 2:
                b_touches = prune_text_overlap_excess(
                    b_touches,
                    2,
                    text_bboxes,
                    flags,
                    "diode",
                )

            k_t = next((t for t in b_touches if t["edge"] == k_edge), None)
            a_t = next((t for t in b_touches if t["edge"] == a_edge), None)
            if k_t:
                pins["K"] = net_for_touch(k_t)
                pin_touches["K"] = k_t
            if a_t:
                pins["A"] = net_for_touch(a_t)
                pin_touches["A"] = a_t

            if len(b_touches) == 1 and ("K" not in pins or "A" not in pins):
                only = b_touches[0]
                opp = OPPOSITE_EDGE[only["edge"]]
                if opp == k_edge:
                    missing_pin = "K"
                elif opp == a_edge:
                    missing_pin = "A"
                else:
                    missing_pin = "A" if "A" not in pins else "K"
                rescue = rescue_touch_for_missing_edge(
                    b_idx,
                    b_bbox,
                    opp,
                    b_touches,
                    pins,
                    missing_pin,
                    flags,
                    "diode_pin",
                )
                if rescue:
                    b_touches = b_touches + [rescue]
                    pins[missing_pin] = net_for_touch(rescue)
                    pin_touches[missing_pin] = rescue
                else:
                    red_flags.append("diode_only_one_touch_no_port")
            elif len(b_touches) != 2:
                red_flags.append(f"diode_touches={len(b_touches)}_expected=2")

            # Touches landed on edges not matching K/A
            for t in b_touches:
                if k_edge and t["edge"] not in (k_edge, a_edge):
                    red_flags.append(f"diode_unexpected_edge={t['edge']}")

        # ------- Sources: voltage / current / ac / battery -------
        elif cls in (12, 13, 14, 15):
            if len(b_touches) > 2:
                b_touches = prune_text_overlap_excess(
                    b_touches,
                    2,
                    text_bboxes,
                    flags,
                    "source",
                )
            if len(b_touches) == 2:
                edges = {t["edge"] for t in b_touches}
                if edges == {"top", "bottom"}:
                    top_t = next(t for t in b_touches if t["edge"] == "top")
                    bot_t = next(t for t in b_touches if t["edge"] == "bottom")
                    pins["+"] = net_for_touch(top_t)
                    pins["-"] = net_for_touch(bot_t)
                    pin_touches["+"] = top_t
                    pin_touches["-"] = bot_t
                elif edges == {"left", "right"}:
                    left_t = next(t for t in b_touches if t["edge"] == "left")
                    right_t = next(t for t in b_touches if t["edge"] == "right")
                    pins["+"] = net_for_touch(left_t)
                    pins["-"] = net_for_touch(right_t)
                    pin_touches["+"] = left_t
                    pin_touches["-"] = right_t
                else:
                    red_flags.append(f"src_ambiguous_edges_{sorted(edges)}")
                    pins["+"] = net_for_touch(b_touches[0])
                    pins["-"] = net_for_touch(b_touches[1])
                    pin_touches["+"] = b_touches[0]
                    pin_touches["-"] = b_touches[1]
            elif len(b_touches) == 1:
                only = b_touches[0]
                # Which pin (+/-) does this touch hold?
                if only["edge"] in ("top", "left"):
                    pins["+"] = net_for_touch(only)
                    pin_touches["+"] = only
                    missing_pin = "-"
                else:
                    pins["-"] = net_for_touch(only)
                    pin_touches["-"] = only
                    missing_pin = "+"
                opp = OPPOSITE_EDGE[only["edge"]]
                rescue = rescue_touch_for_missing_edge(
                    b_idx,
                    b_bbox,
                    opp,
                    b_touches,
                    pins,
                    missing_pin,
                    flags,
                    "src_pin",
                )
                if rescue:
                    b_touches = b_touches + [rescue]
                    pins[missing_pin] = net_for_touch(rescue)
                    pin_touches[missing_pin] = rescue
                else:
                    red_flags.append("src_only_one_touch_no_port")
            else:
                red_flags.append(f"src_touches={len(b_touches)}_expected=2")
                for i, t in enumerate(b_touches):
                    pins[f"pin{i+1}"] = net_for_touch(t)

        # ------- R / C / L / Switch (2-pin, no polarity) -------
        elif cls in (8, 9, 10, 17):
            two_pin_touches = b_touches
            if len(two_pin_touches) > 2:
                two_pin_touches = prune_text_overlap_excess(
                    two_pin_touches,
                    2,
                    text_bboxes,
                    flags,
                    "two_pin",
                )
                b_touches = two_pin_touches
            if len(two_pin_touches) != 2:
                two_pin_touches, n_merged = merge_close_same_node_touches(
                    two_pin_touches
                )
                if n_merged:
                    b_touches = two_pin_touches
                    flags.append(f"close_same_node_merged={n_merged}")

            if len(two_pin_touches) == 2:
                pins["1"] = net_for_touch(two_pin_touches[0])
                pins["2"] = net_for_touch(two_pin_touches[1])
                pin_touches["1"] = two_pin_touches[0]
                pin_touches["2"] = two_pin_touches[1]
            elif len(two_pin_touches) == 1:
                only = two_pin_touches[0]
                pins["1"] = net_for_touch(only)
                pin_touches["1"] = only
                missing_edge = OPPOSITE_EDGE[only["edge"]]
                rescue = rescue_touch_for_missing_edge(
                    b_idx,
                    b_bbox,
                    missing_edge,
                    two_pin_touches,
                    pins,
                    "2",
                    flags,
                    "pin2",
                )
                if rescue:
                    two_pin_touches = two_pin_touches + [rescue]
                    b_touches = two_pin_touches
                    pins["2"] = net_for_touch(rescue)
                    pin_touches["2"] = rescue
                else:
                    red_flags.append("only_one_touch_no_port")
            else:
                red_flags.append(
                    f"unexpected_touches={len(two_pin_touches)}_expected=2"
                )
                for i, t in enumerate(two_pin_touches):
                    pins[f"pin{i+1}"] = net_for_touch(t)

        # ------- Amplifier (class 16) -------
        elif cls == 16:
            # orient points toward the apex/output edge; the input edge is
            # opposite. A differential input doubles up on the input edge.
            amp_out_in_edges = {
                "u": ("top", "bottom"),
                "d": ("bottom", "top"),
                "r": ("right", "left"),
                "l": ("left", "right"),
            }
            amp_touches = list(b_touches)
            # Amplifier pins are distinct nets, so two touches on the same node
            # are one wire wrapping the symbol, not separate pins. Collapse them
            # (distance-independent) before pin assignment.
            if len(amp_touches) > 1:
                amp_touches, n_amp_merged = merge_close_same_node_touches(
                    amp_touches, max_dist=float("inf")
                )
                if n_amp_merged:
                    flags.append(f"amplifier_same_node_collapsed={n_amp_merged}")
            if orient in amp_out_in_edges:
                out_edge, in_edge = amp_out_in_edges[orient]
                align_axis = 0 if orient in ("u", "d") else 1

                def ensure_edge_pins(edge, target):
                    edge_touches = [t for t in amp_touches if t["edge"] == edge]
                    while len(edge_touches) < target:
                        rescue = rescue_touch_for_missing_edge(
                            b_idx,
                            b_bbox,
                            edge,
                            amp_touches,
                            pins,
                            f"pin{len(amp_touches) + 1}",
                            flags,
                            f"amplifier_{edge}_pin",
                        )
                        if not rescue:
                            return edge_touches, False
                        amp_touches.append(rescue)
                        edge_touches.append(rescue)
                    return edge_touches, True

                # Require at least one pin on each side (rescue pulls a
                # single-bbox label-net port or a nearby line if missing).
                out_touches, out_ok = ensure_edge_pins(out_edge, 1)
                if not out_ok:
                    red_flags.append(
                        f"amplifier_{out_edge}_touches={len(out_touches)}"
                        f"_expected_at_least=1"
                    )
                in_touches, in_ok = ensure_edge_pins(in_edge, 1)
                if not in_ok:
                    red_flags.append(
                        f"amplifier_{in_edge}_touches={len(in_touches)}"
                        f"_expected_at_least=1"
                    )

                # Differential input: a single pin on each side that is
                # misaligned along the symmetry axis means the input side is
                # really two pins. Use whatever coordinate we have (a real
                # touch or a single-bbox label-net port both carry contact_xy).
                if (
                    len(out_touches) == 1
                    and len(in_touches) == 1
                    and out_touches[0].get("contact_xy")
                    and in_touches[0].get("contact_xy")
                ):
                    sep = (
                        b_bbox[2] - b_bbox[0]
                        if align_axis == 0
                        else b_bbox[3] - b_bbox[1]
                    )
                    delta = abs(
                        out_touches[0]["contact_xy"][align_axis]
                        - in_touches[0]["contact_xy"][align_axis]
                    )
                    if sep > 0 and delta > AMP_DIFF_ALIGN_RATIO * sep:
                        # Resolve the second differential input. It must be a
                        # distinct net; prefer a real connection, but never grab
                        # an adjacent component (close-bbox): the missing input
                        # of a differential pair is typically a reference net
                        # (e.g. vref) that is a separate port or simply dangling.
                        existing_nodes = {
                            t.get("node_id")
                            for t in amp_touches
                            if t.get("node_id") is not None and t.get("node_id") >= 0
                        }
                        existing_nets = {net_for_touch(t) for t in amp_touches}
                        second = None
                        method = None
                        # (1) a single-bbox label-net port on the input edge
                        for cand in lookup_port_candidates(b_idx, in_edge):
                            if cand.get("node_id") in existing_nodes:
                                continue
                            second = dict(cand)
                            second["_label_net"] = label_net_for_node(cand["node_id"])
                            method = "single_bbox_port"
                            break
                        # (2) a real nearby line on the input edge, distinct net
                        if second is None:
                            line_touch = nearby_line_touch_for_edge(
                                b_idx,
                                b_bbox,
                                in_edge,
                                amp_touches,
                                allow_existing_node=False,
                                edge_center_weight=0.5,
                            )
                            if (
                                line_touch is not None
                                and net_for_touch(line_touch) not in existing_nets
                            ):
                                second = line_touch
                                method = "nearby_line"
                        # (3) otherwise a dangling reference input (e.g. vref)
                        if second is None:
                            second = {
                                "edge": in_edge,
                                "node_id": -1,
                                "_label_net": new_label_net([]),
                                "contact_xy": list(in_touches[0]["contact_xy"]),
                            }
                            method = "dangling_ref"
                        amp_touches.append(second)
                        flags.append(
                            f"amplifier_diff_input_2nd_pin={in_edge}:{method}"
                        )

                allowed_edges = {out_edge, in_edge}
                disallowed = [
                    t for t in amp_touches if t.get("edge") not in allowed_edges
                ]
                disallowed_edges = {t.get("edge") for t in disallowed}
                # Touches off the orientation axis are usually spurious -- BUT a
                # Dido amp carries its differential outputs on BOTH perpendicular
                # edges (e.g. left AND right for a u/d amp). Touches spanning both
                # perpendicular edges are real Dido pins and are kept as-is. When
                # they sit on only ONE perpendicular edge, use symmetry first: try
                # to rescue the differential partner on the opposite (mirror) edge
                # from a nearby line. Keep both if found; otherwise drop the lone
                # stray touch (original behavior).
                if disallowed and len(disallowed_edges) == 1:
                    have_edge = next(iter(disallowed_edges))
                    mirror_edge = OPPOSITE_EDGE.get(have_edge)
                    # A Diso amp carries its single output on the apex
                    # (out_edge); a Dido amp has NO apex output and instead
                    # carries differential outputs on the perpendicular edges.
                    # So if a real output pin already sits on the apex, this is
                    # a Diso: the lone perpendicular touch is spurious and must
                    # be dropped, never mirrored into a fake differential pair.
                    has_apex_output = any(
                        t.get("edge") == out_edge
                        and t.get("node_id") is not None
                        and t.get("node_id", -1) >= 0
                        for t in amp_touches
                    )
                    rescued = None
                    if (
                        not has_apex_output
                        and mirror_edge
                        and mirror_edge not in allowed_edges
                    ):
                        existing_nets = {net_for_touch(t) for t in amp_touches}
                        cand = nearby_line_touch_for_edge(
                            b_idx,
                            b_bbox,
                            mirror_edge,
                            amp_touches,
                            allow_existing_node=False,
                            edge_center_weight=0.5,
                        )
                        if cand is not None and net_for_touch(cand) not in existing_nets:
                            rescued = cand
                    if rescued is not None:
                        amp_touches.append(rescued)
                        flags.append(f"amplifier_dido_mirror_pin={mirror_edge}")
                    else:
                        kept = [
                            t for t in amp_touches if t.get("edge") in allowed_edges
                        ]
                        if kept:
                            amp_touches = kept
                            flags.append(
                                "amplifier_bad_edge_touches_dropped="
                                f"{len(disallowed)}"
                            )
            elif orient is None:
                red_flags.append("missing_orientation")

            edge_order = {"top": 0, "right": 1, "bottom": 2, "left": 3}
            sorted_t = sorted(
                amp_touches,
                key=lambda t: (
                    edge_order.get(t["edge"], 99),
                    t["contact_xy"][0],
                    t["contact_xy"][1],
                ),
            )
            # Amplifier pins are distinct nets; a touch whose net is already used
            # by another pin is a wire wrap or a duplicate rescue (e.g. a forced
            # edge pin that landed back on an existing net) -> drop it.
            seen_amp_nets = set()
            pin_idx = 0
            for t in sorted_t:
                net = net_for_touch(t)
                if net in seen_amp_nets:
                    flags.append("amplifier_duplicate_net_pin_dropped")
                    continue
                seen_amp_nets.add(net)
                pin_idx += 1
                pin_name = f"pin{pin_idx}"
                pins[pin_name] = net
                pin_touches[pin_name] = t

            # Junction-proximity rescue for amplifier pins left as floating
            # label nets: a stub that stops just short of a junction marker
            # adopts the real net merged there (same rule as the transistor
            # pins). Works for a node-less stub (node_id < 0) by probing the
            # pin's own touch line. Skip if the resolved net is already used by
            # another amp pin (amplifier pins are distinct nets).
            for _pn in list(pins.keys()):
                _net = pins.get(_pn)
                if not (isinstance(_net, str) and _net.startswith("label_net")):
                    continue
                _t = pin_touches.get(_pn)
                if _t is None:
                    continue
                _found = net_via_nearby_junction(
                    _t.get("node_id"),
                    _net,
                    _t.get("line_idx", _t.get("_nearby_line_idx")),
                )
                if _found and _found not in pins.values():
                    pins[_pn] = _found
                    flags.append(f"{_pn}_label_net_via_junction={_found}")

        label_net_pin_rescue(cls, cname, b_idx, b_bbox, pins, pin_touches, flags)

        try_close_bbox_pin_rescue(
            cls, cname, b_idx, b_bbox, b_touches, orient, pins, flags, red_flags
        )

        prefix = PREFIX.get(cname, "U")
        type_counters[prefix] += 1
        name = f"{prefix}{type_counters[prefix]}"

        component_record = {
            "name": name,
            "class": cname,
            "bbox_idx": b_idx,
            "bbox": [round(v, 2) for v in b_bbox],
            "orientation": orient,
            "pins": pins,
            "flags": flags,
            "red_flags": red_flags,
        }
        components.append(component_record)
        built_components_by_bbox[b_idx] = component_record

    # Final node list — stable order: GND, VDD, n*, label_net_*
    def node_sort_key(n):
        if n == "GND":
            return (0, 0)
        if n == "VDD":
            return (0, 1)
        m = re.match(r"n(\d+)$", n)
        if m:
            return (1, int(m.group(1)))
        m = re.match(r"label_net_(\d+)$", n)
        if m:
            return (2, int(m.group(1)))
        return (3, n)

    # Merge any nets proven equivalent by image-level short checks.
    parent = {}

    def find(n):
        parent.setdefault(n, n)
        if parent[n] != n:
            parent[n] = find(parent[n])
        return parent[n]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        keep, drop = sorted((ra, rb), key=node_sort_key)
        parent[drop] = keep

    for name in node_origins:
        parent.setdefault(name, name)
    for a, b in net_alias_pairs:
        union(a, b)

    if net_alias_pairs:
        merged_origins = defaultdict(list)
        for name, origins in node_origins.items():
            merged_origins[find(name)].extend(origins)

        node_origins = {
            name: sorted(set(origins)) for name, origins in merged_origins.items()
        }

        for c in components:
            c["pins"] = {pin: find(net) for pin, net in c["pins"].items()}
            if c["class"] in ("nmos", "nmos-bulk"):
                node_origins.setdefault("GND", [])
                c["pins"]["B"] = "GND"
            elif c["class"] in ("pmos", "pmos-bulk"):
                node_origins.setdefault("VDD", [])
                c["pins"]["B"] = "VDD"

    all_nodes = sorted(node_origins.keys(), key=node_sort_key)

    return {
        "image": image_id,
        "nodes": all_nodes,
        "node_origins": node_origins,
        "components": components,
        "flags": image_flags,
    }


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
        image_ids = sorted(
            p.stem for p in BBOX_DIR.glob("*.txt") if p.stem != "class_id"
        )

    for img in image_ids:
        result = build_image(img)
        out_path = OUT_DIR / f"{img}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        n_comp = len(result["components"])
        n_node = len(result["nodes"])
        n_flagged = sum(1 for c in result["components"] if c["flags"])
        n_red = sum(1 for c in result["components"] if c["red_flags"])
        print(
            f"[{img}] {n_comp} components, {n_node} nodes, "
            f"{n_flagged} with benign flags, {n_red} with RED flags -> {out_path}"
        )


if __name__ == "__main__":
    main()
