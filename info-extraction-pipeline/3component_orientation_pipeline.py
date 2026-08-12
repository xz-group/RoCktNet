"""Detect schematic components with YOLO and classify each crop orientation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms
from ultralytics import YOLO


ORIENTATION_CLASS_TO_IDX = {"u": 0, "r": 1, "d": 2, "l": 3}
ORIENTATION_IDX_TO_CLASS = {idx: label for label, idx in ORIENTATION_CLASS_TO_IDX.items()}
ORIENTATION_REQUIRED_CLASSES = {
    "nmos",
    "nmos-bulk",
    "pmos",
    "pmos-bulk",
    "npn",
    "pnp",
    "diode",
    "amplifier",
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def build_orientation_model(weights_path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(weights_path, map_location=device)
    class_to_idx = checkpoint.get("class_to_idx", ORIENTATION_CLASS_TO_IDX)
    if class_to_idx != ORIENTATION_CLASS_TO_IDX:
        raise ValueError(f"Unexpected orientation class mapping in checkpoint: {class_to_idx}")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(ORIENTATION_CLASS_TO_IDX))
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def orientation_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(path for path in input_path.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def clamp_bbox(xyxy: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    left = max(0, min(width, int(round(x1))))
    top = max(0, min(height, int(round(y1))))
    right = max(0, min(width, int(round(x2))))
    bottom = max(0, min(height, int(round(y2))))
    return left, top, right, bottom


def classify_orientation(
    crop: Image.Image,
    model: nn.Module,
    transform: transforms.Compose,
    device: torch.device,
) -> dict[str, Any]:
    with torch.no_grad():
        tensor = transform(crop.convert("RGB")).unsqueeze(0).to(device)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu()
        orientation_id = int(torch.argmax(probs).item())
        orientation = ORIENTATION_IDX_TO_CLASS[orientation_id]
        return {
            "orientation": orientation,
            "orientation_id": orientation_id,
            "orientation_confidence": float(probs[orientation_id].item()),
            "orientation_probabilities": {
                ORIENTATION_IDX_TO_CLASS[idx]: float(prob.item()) for idx, prob in enumerate(probs)
            },
        }


def safe_component_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def predict_image(
    image_path: Path,
    yolo_model: YOLO,
    orientation_model: nn.Module,
    transform: transforms.Compose,
    output_dir: Path,
    device: torch.device,
    conf: float,
    iou: float,
    save_annotated: bool,
) -> dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    image_stem = image_path.stem
    image_output_dir = output_dir / image_stem
    crops_dir = image_output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    results = yolo_model.predict(source=str(image_path), conf=conf, iou=iou, verbose=False)
    result = results[0]
    names = result.names
    components: list[dict[str, Any]] = []

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    for component_id, box in enumerate(result.boxes, start=1):
        class_id = int(box.cls.item())
        component_class = str(names.get(class_id, class_id))
        yolo_confidence = float(box.conf.item())
        x1, y1, x2, y2 = clamp_bbox(box.xyxy[0].detach().cpu().tolist(), width, height)

        record: dict[str, Any] = {
            "component_id": component_id,
            "component_class": component_class,
            "component_class_id": class_id,
            "yolo_confidence": yolo_confidence,
            "bbox_xyxy": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "crop_path": None,
            "orientation": None,
            "orientation_id": None,
            "orientation_confidence": None,
            "orientation_probabilities": None,
            "status": "ok",
        }

        if component_class not in ORIENTATION_REQUIRED_CLASSES:
            record["status"] = "orientation_not_required"
            components.append(record)
            label = f"{component_class} {yolo_confidence:.2f}"
            draw.rectangle((x1, y1, x2, y2), outline="gray", width=2)
            draw_label_y = max(0, y1 - 12)
            draw.text((x1, draw_label_y), label, fill="gray", font=font)
            continue

        if x2 <= x1 or y2 <= y1:
            record["status"] = "skipped_empty_bbox"
            components.append(record)
            continue

        crop = image.crop((x1, y1, x2, y2))
        crop_name = f"{image_stem}_{component_id:04d}_{safe_component_name(component_class)}.jpg"
        crop_path = crops_dir / crop_name
        crop.save(crop_path, quality=95)

        orientation_record = classify_orientation(crop, orientation_model, transform, device)
        record.update(orientation_record)
        record["crop_path"] = str(Path("crops") / crop_name)
        components.append(record)

        label = f"{component_class} {yolo_confidence:.2f} {record['orientation']} {record['orientation_confidence']:.2f}"
        draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
        text_y = max(0, y1 - 12)
        draw.text((x1, text_y), label, fill="red", font=font)

    payload = {
        "image_path": str(image_path),
        "image_width": width,
        "image_height": height,
        "yolo_weights": str(yolo_model.ckpt_path) if hasattr(yolo_model, "ckpt_path") else None,
        "resnet_orientation_classes": ORIENTATION_CLASS_TO_IDX,
        "orientation_required_classes": sorted(ORIENTATION_REQUIRED_CLASSES),
        "num_components": len(components),
        "num_orientation_predictions": sum(1 for component in components if component["orientation"] is not None),
        "components": components,
    }

    json_path = image_output_dir / f"{image_stem}.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if save_annotated:
        annotated.save(image_output_dir / f"{image_stem}_annotated.jpg", quality=95)

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Single schematic image to process.")
    parser.add_argument("--input-dir", type=Path, help="Directory of schematic images to process.")
    parser.add_argument("--yolo-weights", type=Path, default=Path("componentDetection") / "best.pt")
    parser.add_argument("--resnet-weights", type=Path, default=Path("runs") / "orientation_resnet" / "best_model.pt")
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / "component_orientation_predict")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-annotated", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.image) == bool(args.input_dir):
        raise ValueError("Provide exactly one of --image or --input-dir.")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    device = torch.device(args.device)
    input_path = args.image if args.image else args.input_dir
    assert input_path is not None
    image_paths = collect_images(input_path)
    if not image_paths:
        raise FileNotFoundError(f"No images found under: {input_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    yolo_model = YOLO(str(args.yolo_weights))
    orientation_model = build_orientation_model(args.resnet_weights, device)
    transform = orientation_transform(args.image_size)

    print(f"device={device}")
    print(f"images={len(image_paths)}")
    print(f"output_dir={args.output_dir}")

    for image_path in image_paths:
        payload = predict_image(
            image_path=image_path,
            yolo_model=yolo_model,
            orientation_model=orientation_model,
            transform=transform,
            output_dir=args.output_dir,
            device=device,
            conf=args.conf,
            iou=args.iou,
            save_annotated=not args.no_annotated,
        )
        print(f"{image_path}: {payload['num_components']} components")


if __name__ == "__main__":
    main()
