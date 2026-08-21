import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
IMAGE_DIR = ROOT / "images"
INCIDENCE_DIR = ROOT / "incidence_matrix"
TOUCH_DIR = ROOT / "node_touches"

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

NET_COLORS = [
    (220, 20, 60),
    (30, 144, 255),
    (34, 139, 34),
    (255, 140, 0),
    (148, 0, 211),
    (0, 139, 139),
    (255, 20, 147),
    (128, 128, 0),
    (70, 130, 180),
    (178, 34, 34),
    (46, 139, 87),
    (255, 99, 71),
    (75, 0, 130),
    (0, 128, 255),
]

SUPPLY_COLORS = {
    "GND": (35, 35, 35),
    "VDD": (210, 0, 0),
}


def find_image_path(image_id):
    for ext in IMAGE_EXTS:
        path = IMAGE_DIR / f"{image_id}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No source image found for {image_id} in {IMAGE_DIR}")


def load_font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf", "calibri.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def text_size(draw, text, font):
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def draw_label(draw, xy, text, font, fill=(0, 0, 0), bg=(255, 255, 255, 225)):
    x, y = xy
    w, h = text_size(draw, text, font)
    pad = 3
    draw.rounded_rectangle(
        (x - pad, y - pad, x + w + pad, y + h + pad),
        radius=3,
        fill=bg,
    )
    draw.text((x, y), text, font=font, fill=fill)


def node_sort_key(name):
    if name == "GND":
        return (0, 0)
    if name == "VDD":
        return (0, 1)
    if name.startswith("n") and name[1:].isdigit():
        return (1, int(name[1:]))
    if name.startswith("label_net_") and name[10:].isdigit():
        return (2, int(name[10:]))
    return (3, name)


def net_color(net_name, index):
    if net_name in SUPPLY_COLORS:
        return SUPPLY_COLORS[net_name]
    return NET_COLORS[index % len(NET_COLORS)]


def collect_touch_points(touch_data):
    points_by_node_id = defaultdict(list)

    def add_touch(node_id, touch, source):
        points_by_node_id[node_id].append({
            "xy": tuple(touch["contact_xy"]),
            "bbox_idx": touch["component_bbox_idx"],
            "edge": touch["edge"],
            "source": source,
        })

    for node in touch_data.get("nodes", []):
        node_id = node["node_id"]
        for touch in node.get("touches", []):
            add_touch(node_id, touch, "regular")

    for node in touch_data.get("removed_single_bbox_nodes", []):
        node_id = node["node_id"]
        for touch in node.get("touches", []):
            add_touch(node_id, touch, "single_bbox")

    return points_by_node_id


def points_for_net(node_origins, points_by_node_id):
    net_points = {}
    for net, origin_ids in node_origins.items():
        pts = []
        for origin_id in origin_ids:
            pts.extend(points_by_node_id.get(origin_id, []))
        net_points[net] = pts
    return net_points


def centroid(points):
    xs = [p["xy"][0] for p in points]
    ys = [p["xy"][1] for p in points]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def draw_net_overlay(draw, data, net_points, font):
    for idx, net in enumerate(sorted(data["nodes"], key=node_sort_key)):
        pts = net_points.get(net, [])
        if not pts:
            continue

        color = net_color(net, idx)
        rgba = (*color, 185)
        center = centroid(pts)

        if len(pts) > 1:
            for p in pts:
                draw.line((center, p["xy"]), fill=rgba, width=3)

        cx, cy = center
        draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=(*color, 230))
        draw_label(draw, (cx + 7, cy + 7), net, font, fill=color)

        for p in pts:
            x, y = p["xy"]
            r = 5 if p["source"] == "regular" else 4
            outline = (255, 255, 255, 230)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(*color, 235), outline=outline, width=2)


def draw_components(draw, data, font, small_font):
    for comp in data["components"]:
        x1, y1, x2, y2 = comp["bbox"]
        has_red = bool(comp.get("red_flags"))
        outline = (255, 0, 0, 240) if has_red else (30, 30, 30, 220)
        draw.rectangle((x1, y1, x2, y2), outline=outline, width=2)

        title = f"{comp['name']} {comp['class']}"
        draw_label(draw, (x1, max(0, y1 - 18)), title, font, fill=outline[:3])

        pin_lines = [f"{pin}:{net}" for pin, net in comp.get("pins", {}).items()]
        if pin_lines:
            text = "\n".join(pin_lines)
            draw_label(draw, (x2 + 5, y1), text, small_font, fill=(0, 0, 0), bg=(255, 255, 255, 210))


def draw_legend(draw, x0, image_height, data, font, small_font):
    draw.rectangle((x0, 0, x0 + 360, image_height), fill=(248, 248, 248))
    draw.text((x0 + 16, 14), f"{data['image']} incidence overlay", font=font, fill=(0, 0, 0))
    draw.text((x0 + 16, 40), "Net colors", font=small_font, fill=(60, 60, 60))

    y = 64
    for idx, net in enumerate(sorted(data["nodes"], key=node_sort_key)):
        color = net_color(net, idx)
        draw.rectangle((x0 + 16, y + 3, x0 + 30, y + 17), fill=color)
        origins = ",".join(str(n) for n in data["node_origins"].get(net, []))
        line = f"{net}  [{origins}]"
        draw.text((x0 + 38, y), line, font=small_font, fill=(0, 0, 0))
        y += 22
        if y > image_height - 24:
            draw.text((x0 + 16, y), "...", font=small_font, fill=(0, 0, 0))
            break


def visualize_image(image_id, out_dir):
    incidence_path = INCIDENCE_DIR / f"{image_id}.json"
    touch_path = TOUCH_DIR / f"{image_id}.json"
    if not incidence_path.exists():
        raise FileNotFoundError(f"Missing incidence matrix: {incidence_path}")
    if not touch_path.exists():
        raise FileNotFoundError(f"Missing node touches: {touch_path}")

    with open(incidence_path) as f:
        data = json.load(f)
    with open(touch_path) as f:
        touch_data = json.load(f)

    image = Image.open(find_image_path(image_id)).convert("RGBA")
    width, height = image.size
    legend_width = 360
    canvas = Image.new("RGBA", (width + legend_width, height), (255, 255, 255, 255))
    canvas.paste(image, (0, 0))

    overlay = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    font = load_font(13)
    small_font = load_font(11)

    points_by_node_id = collect_touch_points(touch_data)
    net_points = points_for_net(data["node_origins"], points_by_node_id)

    draw_net_overlay(draw, data, net_points, small_font)
    draw_components(draw, data, font, small_font)
    draw_legend(draw, width, height, data, font, small_font)

    result = Image.alpha_composite(canvas, overlay).convert("RGB")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_id}_incidence_overlay.png"
    result.save(out_path)
    return out_path

