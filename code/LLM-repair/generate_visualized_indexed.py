#!/usr/bin/env python3
"""Draw bounding-box annotations on schematic images using pre-computed bbox txt files.

Usage:
  python3 generate_visualized_indexed.py --base-dir /path/to/resultXxx
  python3 generate_visualized_indexed.py --base-dir /path/to/resultXxx \
      --bbox-dir /custom/bbox --img-dir /custom/images --out-dir /custom/out
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CLASS_NAMES = [
    "gnd", "vdd", "nmos", "nmos-bulk", "pmos", "pmos-bulk",
    "npn", "pnp", "resistor", "capacitor", "inductor", "diode",
    "voltage_src", "ac_src", "current_src", "battery", "amplifier", "switch_ideal",
]

INDEXED_TYPES = {
    "nmos": "M", "nmos-bulk": "M", "pmos": "M", "pmos-bulk": "M",
    "npn": "Q", "pnp": "Q",
    "resistor": "R", "capacitor": "C", "inductor": "L",
    "voltage_src": "V", "current_src": "I", "ac_src": "V",
    "diode": "D", "battery": "V", "amplifier": "A", "switch_ideal": "S",
}

COLORS = {
    "nmos": "#FF4444", "nmos-bulk": "#FF6666",
    "pmos": "#4444FF", "pmos-bulk": "#6666FF",
    "npn": "#FF8800", "pnp": "#FF8800",
    "resistor": "#00AA00", "capacitor": "#AA00AA",
    "inductor": "#008888", "diode": "#888800",
    "voltage_src": "#AA4400", "current_src": "#AA4400",
    "ac_src": "#AA4400", "battery": "#AA4400",
    "amplifier": "#004488", "switch_ideal": "#448800",
    "gnd": "#999999", "vdd": "#999999",
}


def get_label_size(draw: ImageDraw.ImageDraw, label: str, font) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), label, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def process(bbox_dir: Path, img_dir: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", 10)
    except Exception:
        font = ImageFont.load_default()

    count = 0
    for bbox_file in sorted(bbox_dir.glob("*.txt")):
        cid = bbox_file.stem
        img_path = next(
            (img_dir / f"{cid}{s}" for s in (".jpg", ".jpeg", ".png")
             if (img_dir / f"{cid}{s}").exists()),
            None,
        )
        if img_path is None:
            continue

        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        img_w, img_h = img.size

        lines = bbox_file.read_text().splitlines()
        boxes = []
        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 6:
                continue
            x1, y1, x2, y2 = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
            cls_id = int(parts[4])
            conf = float(parts[5])
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls{cls_id}"
            boxes.append((conf, x1, y1, x2, y2, cls_name))

        boxes.sort(reverse=True)
        type_counter: dict[str, int] = {}
        for conf, x1, y1, x2, y2, cls_name in boxes:
            color = COLORS.get(cls_name, "#AAAAAA")
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            if cls_name in INDEXED_TYPES:
                prefix = INDEXED_TYPES[cls_name]
                type_counter[prefix] = type_counter.get(prefix, 0) + 1
                label = f"{prefix}{type_counter[prefix]} ({cls_name})"
            else:
                label = cls_name

            lw, lh = get_label_size(draw, label, font)
            pad = 2
            lx = max(0, min(x1, img_w - lw - pad * 2))
            ly = y1 - lh - pad * 2 if y1 >= lh + pad * 2 else y1 + pad

            draw.rectangle([lx, ly, lx + lw + pad * 2, ly + lh + pad * 2], fill=color)
            draw.text((lx + pad, ly + pad), label, fill="white", font=font)

        img.save(out_dir / f"{cid}.jpg")
        count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate bbox-annotated visualized images.")
    parser.add_argument("--base-dir", type=Path, required=True,
                        help="Dataset root (expects subdirs: component_bbox/, images/)")
    parser.add_argument("--bbox-dir", type=Path, default=None,
                        help="Override bbox directory (default: <base-dir>/component_bbox)")
    parser.add_argument("--img-dir", type=Path, default=None,
                        help="Override image directory (default: <base-dir>/images)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Override output directory (default: <base-dir>/visualized_indexed)")
    args = parser.parse_args()

    base = args.base_dir.resolve()
    bbox_dir = (args.bbox_dir or base / "component_bbox").resolve()
    img_dir  = (args.img_dir  or base / "images").resolve()
    out_dir  = (args.out_dir  or base / "visualized_indexed").resolve()

    print(f"bbox dir : {bbox_dir}")
    print(f"image dir: {img_dir}")
    print(f"out dir  : {out_dir}")

    count = process(bbox_dir, img_dir, out_dir)
    print(f"Done: {count} images -> {out_dir}")


if __name__ == "__main__":
    main()
