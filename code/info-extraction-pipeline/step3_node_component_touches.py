#!/usr/bin/env python3

import json
from collections import defaultdict

import numpy as np


def load_component_bboxes(bbox_path):
    if not bbox_path.exists():
        return []

    arr = np.loadtxt(str(bbox_path), skiprows=1, ndmin=2)
    if arr.size == 0:
        return []

    out = []
    for row in arr:
        if len(row) < 6:
            continue
        out.append(
            {
                "xyxy": [float(v) for v in row[:4]],
                "class": int(row[4]),
                "conf": float(row[5]),
            }
        )
    return out


def nearest_endpoint(line, contact_xy):
    p0 = np.asarray(line[0], dtype=np.float32)
    p1 = np.asarray(line[1], dtype=np.float32)
    contact = np.asarray(contact_xy, dtype=np.float32)

    d0 = float(np.linalg.norm(p0 - contact))
    d1 = float(np.linalg.norm(p1 - contact))

    if d0 <= d1:
        return 0, [float(p0[0]), float(p0[1])], d0
    return 1, [float(p1[0]), float(p1[1])], d1


def outward_direction(line, endpoint_idx):
    p0 = np.asarray(line[0], dtype=np.float32)
    p1 = np.asarray(line[1], dtype=np.float32)
    direction = p0 - p1 if int(endpoint_idx) == 0 else p1 - p0
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return None
    return direction / norm


def segment_bbox_hits(start_xy, direction, max_dist, bbox_xyxy):
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    px, py = [float(v) for v in start_xy]
    dx, dy = [float(v) for v in direction]
    eps = 1e-6
    hits = []

    def add_hit(t, x, y, edge):
        if -eps <= t <= float(max_dist) + eps:
            hits.append(
                {
                    "edge": edge,
                    "dist": max(0.0, float(t)),
                    "contact_xy": [float(x), float(y)],
                }
            )

    if abs(dx) > eps:
        t = (x1 - px) / dx
        y = py + t * dy
        if y1 - eps <= y <= y2 + eps:
            add_hit(t, x1, min(max(y, y1), y2), "left")

        t = (x2 - px) / dx
        y = py + t * dy
        if y1 - eps <= y <= y2 + eps:
            add_hit(t, x2, min(max(y, y1), y2), "right")

    if abs(dy) > eps:
        t = (y1 - py) / dy
        x = px + t * dx
        if x1 - eps <= x <= x2 + eps:
            add_hit(t, min(max(x, x1), x2), y1, "top")

        t = (y2 - py) / dy
        x = px + t * dx
        if x1 - eps <= x <= x2 + eps:
            add_hit(t, min(max(x, x1), x2), y2, "bottom")

    # At a rectangle corner the same geometric hit may appear twice. Keep the
    # first edge after sorting by distance; edge choice at an exact corner is
    # inherently ambiguous, but the contact point remains correct.
    hits.sort(key=lambda h: (h["dist"], h["edge"]))
    unique = []
    seen = set()
    for hit in hits:
        key = (round(hit["contact_xy"][0], 4), round(hit["contact_xy"][1], 4))
        if key in seen:
            continue
        seen.add(key)
        unique.append(hit)
    return unique


def find_extended_endpoint_touch(
    line_idx,
    endpoint_idx,
    lines,
    line_node_ids,
    component_bboxes,
    max_extend_px,
):
    line = lines[int(line_idx)]
    endpoint = np.asarray(line[int(endpoint_idx)], dtype=np.float32)
    direction = outward_direction(line, endpoint_idx)
    if direction is None:
        return None

    best = None
    for bbox_idx, bbox_info in enumerate(component_bboxes):
        hits = segment_bbox_hits(
            start_xy=endpoint,
            direction=direction,
            max_dist=max_extend_px,
            bbox_xyxy=bbox_info["xyxy"],
        )
        if not hits:
            continue

        hit = hits[0]
        if best is None or hit["dist"] < best["extension_dist"]:
            best = {
                "node_id": int(line_node_ids[int(line_idx)]),
                "line_idx": int(line_idx),
                "endpoint_idx": int(endpoint_idx),
                "endpoint_xy": [float(endpoint[0]), float(endpoint[1])],
                "endpoint_to_contact_dist": float(hit["dist"]),
                "contact_xy": hit["contact_xy"],
                "component_bbox_idx": int(bbox_idx),
                "component_bbox_xyxy": bbox_info["xyxy"],
                "component_class": bbox_info["class"],
                "component_conf": bbox_info["conf"],
                "edge": hit["edge"],
                "anchor_dist": None,
                "extension_dist": float(hit["dist"]),
                "source": "extended_endpoint",
            }
    return best


def median_xy(points):
    arr = np.asarray(points, dtype=np.float32)
    med = np.median(arr, axis=0)
    return [float(med[0]), float(med[1])]


def min_optional(values):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(min(vals))


def aggregate_touches(touches):
    groups = defaultdict(list)
    for touch in touches:
        key = (
            int(touch["node_id"]),
            int(touch["component_bbox_idx"]),
            str(touch["edge"]),
        )
        groups[key].append(touch)

    aggregated = []
    for (node_id, bbox_idx, edge), items in groups.items():
        first = items[0]
        sources = sorted(set(str(t.get("source", "unknown")) for t in items))
        if len(sources) == 1:
            source = sources[0]
        else:
            source = "mixed"

        contributors = []
        for t in items:
            contributors.append(
                {
                    "line_idx": int(t["line_idx"]),
                    "endpoint_idx": int(t["endpoint_idx"]),
                    "endpoint_xy": t["endpoint_xy"],
                    "contact_xy": t["contact_xy"],
                    "endpoint_to_contact_dist": float(
                        t.get("endpoint_to_contact_dist", 0.0)
                    ),
                    "anchor_dist": t.get("anchor_dist"),
                    "extension_dist": t.get("extension_dist"),
                    "source": t.get("source", "unknown"),
                }
            )

        endpoint_xy = median_xy([t["endpoint_xy"] for t in items])
        contact_xy = median_xy([t["contact_xy"] for t in items])
        endpoint_to_contact_dist = float(
            np.median([float(t.get("endpoint_to_contact_dist", 0.0)) for t in items])
        )

        aggregated.append(
            {
                "node_id": int(node_id),
                "line_idx": int(first["line_idx"]),
                "endpoint_idx": int(first["endpoint_idx"]),
                "endpoint_xy": endpoint_xy,
                "endpoint_to_contact_dist": endpoint_to_contact_dist,
                "contact_xy": contact_xy,
                "component_bbox_idx": int(bbox_idx),
                "component_bbox_xyxy": first.get("component_bbox_xyxy"),
                "component_class": first.get("component_class"),
                "component_conf": first.get("component_conf"),
                "edge": edge,
                "anchor_dist": min_optional(t.get("anchor_dist") for t in items),
                "extension_dist": min_optional(t.get("extension_dist") for t in items),
                "source": source,
                "sources": sources,
                "num_contributors": int(len(items)),
                "contributors": contributors,
            }
        )
    return aggregated


def build_touch_record(anchor, lines, line_node_ids, component_bboxes):
    line_idx = int(anchor["line_idx"])
    if line_idx < 0 or line_idx >= len(lines):
        raise IndexError(
            f"anchor line_idx {line_idx} out of bounds for {len(lines)} lines"
        )

    line = lines[line_idx]
    contact_xy = [float(v) for v in anchor["contact"]]
    endpoint_idx, endpoint_xy, endpoint_to_contact_dist = nearest_endpoint(
        line, contact_xy
    )

    bbox_idx = int(anchor["bbox_idx"])
    bbox_info = (
        component_bboxes[bbox_idx] if 0 <= bbox_idx < len(component_bboxes) else None
    )

    node_id = int(line_node_ids[line_idx]) if line_idx < len(line_node_ids) else -1

    return {
        "node_id": node_id,
        "line_idx": line_idx,
        "endpoint_idx": int(endpoint_idx),
        "endpoint_xy": endpoint_xy,
        "endpoint_to_contact_dist": float(endpoint_to_contact_dist),
        "contact_xy": contact_xy,
        "component_bbox_idx": bbox_idx,
        "component_bbox_xyxy": bbox_info["xyxy"] if bbox_info is not None else None,
        "component_class": bbox_info["class"] if bbox_info is not None else None,
        "component_conf": bbox_info["conf"] if bbox_info is not None else None,
        "edge": str(anchor["edge_name"]),
        "anchor_dist": float(anchor.get("dist", endpoint_to_contact_dist)),
        "extension_dist": None,
        "source": "anchor",
    }


def add_extended_endpoint_touches(
    touches,
    lines,
    line_node_ids,
    component_bboxes,
    max_extend_px,
):
    node_to_line_indices = defaultdict(list)
    for line_idx, node_id in enumerate(line_node_ids):
        if int(node_id) >= 0:
            node_to_line_indices[int(node_id)].append(int(line_idx))

    existing_endpoint_keys = {
        (
            int(t["line_idx"]),
            int(t["endpoint_idx"]),
            int(t["component_bbox_idx"]),
            str(t["edge"]),
        )
        for t in touches
    }

    added = []
    for node_id, node_line_indices in node_to_line_indices.items():
        for line_idx in node_line_indices:
            for endpoint_idx in (0, 1):
                touch = find_extended_endpoint_touch(
                    line_idx=line_idx,
                    endpoint_idx=endpoint_idx,
                    lines=lines,
                    line_node_ids=line_node_ids,
                    component_bboxes=component_bboxes,
                    max_extend_px=max_extend_px,
                )
                if touch is None:
                    continue

                key = (
                    int(touch["line_idx"]),
                    int(touch["endpoint_idx"]),
                    int(touch["component_bbox_idx"]),
                    str(touch["edge"]),
                )
                if key in existing_endpoint_keys:
                    continue

                existing_endpoint_keys.add(key)
                added.append(touch)

    touches.extend(added)
    return added


def filter_single_bbox_nodes(touches):
    bbox_ids_by_node = defaultdict(set)
    for touch in touches:
        bbox_idx = touch.get("component_bbox_idx")
        if bbox_idx is not None and int(bbox_idx) >= 0:
            bbox_ids_by_node[int(touch["node_id"])].add(int(bbox_idx))

    keep_node_ids = {
        node_id for node_id, bbox_ids in bbox_ids_by_node.items() if len(bbox_ids) > 1
    }
    removed_node_ids = sorted(
        node_id for node_id in bbox_ids_by_node.keys() if node_id not in keep_node_ids
    )
    kept = [t for t in touches if int(t["node_id"]) in keep_node_ids]
    return kept, removed_node_ids


def export_one(
    stem,
    npz_path,
    json_path,
    bbox_dir,
    output_dir,
    dedupe=False,
    extra_endpoint_extend_px=10.0,
    keep_single_bbox_nodes=False,
):
    data = np.load(npz_path)
    lines = data["lines"].astype(np.float32)
    line_node_ids = data["line_node_ids"].astype(np.int32)

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    component_bboxes = load_component_bboxes(bbox_dir / f"{stem}.txt")
    touches = []
    warnings = []

    for anchor in meta.get("anchors", []):
        try:
            touches.append(
                build_touch_record(anchor, lines, line_node_ids, component_bboxes)
            )
        except Exception as exc:
            warnings.append(str(exc))

    added_touches = add_extended_endpoint_touches(
        touches=touches,
        lines=lines,
        line_node_ids=line_node_ids,
        component_bboxes=component_bboxes,
        max_extend_px=extra_endpoint_extend_px,
    )

    # raw_num_touches_before_aggregation = len(touches)
    # touches = aggregate_touches(touches)
    # raw_num_touches = len(touches)
    # removed_single_bbox_node_ids = []
    # if not keep_single_bbox_nodes:
    #     touches, removed_single_bbox_node_ids = filter_single_bbox_nodes(touches)

    # touches.sort(
    #     key=lambda t: (
    #         int(t["node_id"]),
    #         int(t["component_bbox_idx"]),
    #         str(t["edge"]),
    #         int(t["line_idx"]),
    #     )
    # )

    # removed_single_bbox_node_ids.sort(
    #     key=lambda t: (
    #         int(t["node_id"]),
    #         int(t["component_bbox_idx"]),
    #         str(t["edge"]),
    #         int(t["line_idx"]),
    #     )
    # )
    raw_num_touches_before_aggregation = len(touches)
    touches = aggregate_touches(touches)
    raw_num_touches = len(touches)

    # Keep a snapshot of aggregated touches before the single-bbox filter.
    touches_before_single_bbox_filter = list(touches)

    removed_single_bbox_node_ids = []
    removed_single_bbox_touches = []

    if not keep_single_bbox_nodes:
        touches, removed_single_bbox_node_ids = filter_single_bbox_nodes(touches)

        removed_node_id_set = set(int(x) for x in removed_single_bbox_node_ids)

        # Recover full touch records for nodes removed by the filter.
        removed_single_bbox_touches = [
            t
            for t in touches_before_single_bbox_filter
            if int(t["node_id"]) in removed_node_id_set
        ]

    # Sort retained touches.
    touches.sort(
        key=lambda t: (
            int(t["node_id"]),
            int(t["component_bbox_idx"]),
            str(t["edge"]),
            int(t["line_idx"]),
        )
    )

    # Sort single-bbox touches removed by the filter.
    removed_single_bbox_touches.sort(
        key=lambda t: (
            int(t["node_id"]),
            int(t["component_bbox_idx"]),
            str(t["edge"]),
            int(t["line_idx"]),
        )
    )

    # Removed node ids are plain ints, so regular sorting is enough.
    removed_single_bbox_node_ids = sorted(int(x) for x in removed_single_bbox_node_ids)

    grouped = defaultdict(list)
    grouped_single = defaultdict(list)
    for touch in touches:
        grouped[int(touch["node_id"])].append(touch)

    for removed_touch in removed_single_bbox_touches:
        grouped_single[int(removed_touch["node_id"])].append(removed_touch)
    nodes = [
        {
            "node_id": node_id,
            "num_touches": len(node_touches),
            "touches": node_touches,
        }
        for node_id, node_touches in sorted(grouped.items())
    ]

    single_bbox_nodes = [
        {
            "node_id": node_id,
            "num_touches": len(node_touches),
            "touches": node_touches,
        }
        for node_id, node_touches in sorted(grouped_single.items())
    ]

    out = {
        "image": stem,
        "source_npz": str(npz_path),
        "source_json": str(json_path),
        "source_bbox_txt": str(bbox_dir / f"{stem}.txt"),
        "num_lines": int(len(lines)),
        "num_source_nodes": int(meta.get("num_nodes", len(nodes))),
        "num_nodes": int(len(nodes)),
        "num_touches": int(len(touches)),
        "num_touches_before_single_bbox_filter": int(raw_num_touches),
        "num_raw_touches_before_aggregation": int(raw_num_touches_before_aggregation),
        "num_anchor_touches": int(
            sum(1 for t in touches if "anchor" in t.get("sources", []))
        ),
        "num_extended_endpoint_touches": int(
            sum(1 for t in touches if "extended_endpoint" in t.get("sources", []))
        ),
        "num_added_extended_endpoint_touches_before_filter": int(len(added_touches)),
        "removed_single_bbox_node_ids": removed_single_bbox_node_ids,
        "aggregated_by": ["node_id", "component_bbox_idx", "edge"],
        "deduped": bool(dedupe),
        "params": {
            "extra_endpoint_extend_px": float(extra_endpoint_extend_px),
            "keep_single_bbox_nodes": bool(keep_single_bbox_nodes),
        },
        "nodes": nodes,
        "removed_single_bbox_nodes": single_bbox_nodes,
    }
    if warnings:
        out["warnings"] = warnings

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{stem}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return out_path, len(touches), len(nodes), len(added_touches), warnings

