#!/usr/bin/env python3

import json

import matplotlib.pyplot as plt
import numpy as np
import skimage.io

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def node_color(node_id):
    if node_id < 0:
        return (0.1, 0.1, 0.1)

    # Deterministic hue per node: step 37 around the wheel so adjacent node ids
    # land far apart on the hsv colormap.
    hue = ((node_id * 37) % 360) / 360.0
    return tuple(float(v) for v in plt.cm.hsv(hue)[:3])


def find_image(image_dir, stem):
    for ext in IMG_EXTS:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def read_image(path):
    im = skimage.io.imread(str(path))
    if im.ndim == 2:
        im = np.repeat(im[:, :, None], 3, axis=2)
    return im[:, :, :3]


def all_touches(doc):
    for node in doc.get("nodes", []):
        for touch in node.get("touches", []):
            yield touch


def draw_bbox_once(ax, bbox, seen):
    if bbox is None:
        return
    key = tuple(round(float(v), 2) for v in bbox)
    if key in seen:
        return
    seen.add(key)
    x1, y1, x2, y2 = [float(v) for v in bbox]
    ax.add_patch(
        plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor=(0.55, 0.55, 0.55),
            linewidth=0.8,
            linestyle=":",
            zorder=2,
        )
    )


def visualize_one(
    touch_path, image_dir, output_dir, draw_labels=True, show_endpoints=False
):
    with open(touch_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    stem = doc.get("image", touch_path.stem)
    image_path = find_image(image_dir, stem)
    if image_path is None:
        raise FileNotFoundError(f"image for stem {stem!r} not found in {image_dir}")

    im = read_image(image_path)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im)
    ax.set_axis_off()
    ax.set_title(
        f"{stem} node-component touches "
        f"(nodes={doc.get('num_nodes', '?')}, touches={doc.get('num_touches', '?')})"
    )

    seen_bboxes = set()
    touches = list(all_touches(doc))

    for touch in touches:
        draw_bbox_once(ax, touch.get("component_bbox_xyxy"), seen_bboxes)

    for touch in touches:
        node_id = int(touch["node_id"])
        color = node_color(node_id)
        contact = touch["contact_xy"]
        cx, cy = float(contact[0]), float(contact[1])
        source = touch.get("source", "anchor")
        if source == "anchor":
            marker = "o"
            size = 44
        elif source == "extended_endpoint":
            marker = "^"
            size = 54
        else:
            marker = "D"
            size = 50

        ax.scatter(
            [cx],
            [cy],
            s=size,
            marker=marker,
            color=[color],
            edgecolors="white",
            linewidths=0.7,
            zorder=6,
        )

        if draw_labels:
            label = f"N{node_id}" if source == "anchor" else f"N{node_id}*"
            if touch.get("num_contributors", 1) > 1:
                label = f"{label}({touch['num_contributors']})"
            ax.text(
                cx + 3,
                cy - 3,
                label,
                color=color,
                fontsize=7,
                weight="bold",
                zorder=7,
                bbox={
                    "facecolor": "white",
                    "alpha": 0.72,
                    "edgecolor": "none",
                    "pad": 0.8,
                },
            )

        if show_endpoints:
            contributors = touch.get("contributors") or [touch]
            for contributor in contributors:
                endpoint = contributor["endpoint_xy"]
                ex, ey = float(endpoint[0]), float(endpoint[1])
                ax.plot(
                    [ex, cx],
                    [ey, cy],
                    color=color,
                    linewidth=0.6,
                    alpha=0.45,
                    zorder=5,
                )
                ax.scatter(
                    [ex],
                    [ey],
                    s=14,
                    marker="x",
                    color=[color],
                    linewidths=0.8,
                    alpha=0.8,
                    zorder=6,
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{stem}.png"
    fig.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight", dpi=140)
    plt.close(fig)
    return out_path, len(touches)

