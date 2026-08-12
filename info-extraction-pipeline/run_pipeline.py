#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image, ImageDraw, ImageFont

from pipeline_common import (
    CLASS_NAMES,
    IMAGE_EXTS,
    collect_images,
    ensure_output_dirs,
    import_module_from_path,
    load_pipeline_config,
    prepare_input_images,
    selected_stems,
    sync_component_bboxes,
)


STEPS = (
    "detect_components",
    "detect_orientation",
    "extract_lines",
    "generate_nodes",
    "export_touches",
    "build_incidence",
    "build_netlist",
)


def resolve_device(value: str | None) -> torch.device:
    if not value or value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if value == "mps" and (
        not getattr(torch.backends, "mps", None)
        or not torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available.")
    return torch.device(value)


def make_stage1_args(config, stems: list[str] | None, device: torch.device, force_ocr: bool):
    return SimpleNamespace(
        device=str(device),
        conf=config.getfloat("params", "component_conf", 0.25),
        iou=config.getfloat("params", "component_iou", 0.45),
        pad=config.getint("params", "mask_pad", 1),
        mask_color=config.get("params", "mask_color", "white"),
        ocr_pad=config.getint("params", "ocr_pad", 2),
        ocr_conf=config.getfloat("params", "ocr_conf", 0.2),
        force_ocr=force_ocr,
        combined_hawp_threshold=config.getfloat("params", "combined_hawp_threshold", 0.05),
        text_distance=config.getfloat("params", "text_distance", 5.0),
        text_keep_line_len=config.getfloat("params", "text_keep_line_len", 12.0),
        black_bridge_len=config.getint("params", "black_bridge_len", 0),
        min_line_length=config.getfloat("params", "min_line_length", 4.0),
        whitespace_threshold=config.getint("params", "whitespace_threshold", 240),
        whitespace_fraction=config.getfloat("params", "whitespace_fraction", 0.8),
        extra_endpoint_extend_px=config.getfloat("params", "extra_endpoint_extend_px", 10.0),
        node_union_dist=config.getfloat("params", "node_union_dist", 8.0),
        node_pixel_bridge_dist=config.getfloat("params", "node_pixel_bridge_dist", 0.0),
        node_pixel_bridge_fill=config.getfloat("params", "node_pixel_bridge_fill", 0.6),
        node_pixel_bridge_radius=config.getint("params", "node_pixel_bridge_radius", 1),
        jj_conf=config.getfloat("params", "jj_conf", 0.25),
        jj_bbox_pad=config.getfloat("params", "jj_bbox_pad", 2.0),
        perp_tol=config.getfloat("params", "perp_tol_deg", 10.0),
        crossover_probe_dist=config.getfloat("params", "crossover_probe_dist", 5.0),
        crossover_probe_tol=config.getfloat("params", "crossover_probe_tol", 3.0),
        crossover_min_line_len=config.getfloat("params", "crossover_min_line_len", 3.0),
        implicit_jump_min_extend_px=config.getfloat("params", "implicit_jump_min_extend_px", 3.0),
        explicit_jump_min_extend_px=config.getfloat("params", "explicit_jump_min_extend_px", 1.0),
        jump_edge_component_adjacent_px=config.getfloat("params", "jump_edge_component_adjacent_px", 6.0),
        jj_trust_yolo=config.getbool("params", "jj_trust_yolo", False),
        jump_parallel_tol=config.getfloat("params", "jump_parallel_tol", 20.0),
        max_anchor_dist=None,
        max_extend_px=12.0,
        extra_into_bbox_px=2.0,
        black_threshold=128,
        inside_depth=4,
        patch_radius=1,
        stem=stems,
    )


def load_stage1(config):
    module = import_module_from_path("stage1_pipeline", config.paths.repo_root / "1pipeline.py")
    module.YOLO_WEIGHTS = config.paths.component_weights
    module.JJ_YOLO_WEIGHTS = config.paths.junction_jump_weights
    module.JJ_YOLO_WEIGHTS_2CLS = config.paths.junction_jump_weights_2cls
    module.TEST_IMAGES = config.paths.images_dir
    module.MASKED_DIR = config.paths.masked_dir
    module.MASKED_NOTEXT_DIR = config.paths.masked_no_text_dir
    module.HAWP_DIR = config.paths.hawp_dir
    module.HAWP_WEIGHTS = config.paths.hawp_dir / "bestv3.pth"
    module.HAWP_CONFIG = config.paths.hawp_dir / "hawp" / "ssl" / "config" / "hawpv3.yaml"
    module.COMBINED_RESULTS_DIR = config.paths.output_dir
    module.COMBINED_LINES_DIR = config.paths.combined_lines_dir
    module.COMBINED_VIS_DIR = config.paths.combined_vis_dir
    module.NODE_DIR = config.paths.nodes_dir
    module.NODE_VIS_DIR = config.paths.node_vis_dir
    module.NODE_DATA_DIR = config.paths.node_data_dir
    # Optional GUI hook: if result/manual_jj_overrides/{stem}.json exists,
    # its added jumps/junctions are unioned with the YOLO output during
    # generate_nodes. The dir doesn't need to exist; missing means no-op.
    module.MANUAL_JJ_OVERRIDES_DIR = config.paths.output_dir / "manual_jj_overrides"
    return module


def read_component_bboxes(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing component bbox file: {path}")
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("x1"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        x1, y1, x2, y2 = map(float, parts[:4])
        cls = int(float(parts[4]))
        conf = float(parts[5]) if len(parts) > 5 else None
        out.append(
            {
                "idx": len(out),
                "xyxy": [x1, y1, x2, y2],
                "class_id": cls,
                "class_name": CLASS_NAMES.get(cls, f"class_{cls}"),
                "confidence": conf,
            }
        )
    return out


def clamp_bbox(xyxy: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    return (
        max(0, min(width, int(round(x1)))),
        max(0, min(height, int(round(y1)))),
        max(0, min(width, int(round(x2)))),
        max(0, min(height, int(round(y2)))),
    )


def find_image(images_dir: Path, stem: str) -> Path:
    for ext in IMAGE_EXTS:
        path = images_dir / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing image for stem {stem} in {images_dir}")


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font) -> None:
    draw.text(xy, text, fill="red", font=font)


def step_detect_components(config, stems: list[str] | None, device: torch.device, force_ocr: bool) -> None:
    stage1 = load_stage1(config)
    args = make_stage1_args(config, stems, device, force_ocr)
    stage1.run_yolo_and_mask(args, device)
    sync_component_bboxes(config.paths, direction="from_masked")


def step_detect_orientation(config, stems: list[str] | None, device: torch.device) -> None:
    orient_mod = import_module_from_path(
        "component_orientation", config.paths.repo_root / "3component_orientation_pipeline.py"
    )
    model = orient_mod.build_orientation_model(config.paths.orientation_weights, device)
    transform = orient_mod.orientation_transform(config.getint("params", "orientation_image_size", 224))
    config.paths.orientation_dir.mkdir(parents=True, exist_ok=True)
    config.paths.orientation_crop_dir.mkdir(parents=True, exist_ok=True)
    config.paths.orientation_vis_dir.mkdir(parents=True, exist_ok=True)

    for stem in selected_stems(config.paths, stems):
        image_path = find_image(config.paths.images_dir, stem)
        bbox_path = config.paths.component_bbox_dir / f"{stem}.txt"
        bboxes = read_component_bboxes(bbox_path)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        crops_dir = config.paths.orientation_crop_dir / stem
        crops_dir.mkdir(parents=True, exist_ok=True)

        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        font = ImageFont.load_default()
        components = []

        for bbox in bboxes:
            component_id = bbox["idx"] + 1
            x1, y1, x2, y2 = clamp_bbox(bbox["xyxy"], width, height)
            component_class = bbox["class_name"]
            record = {
                "component_id": component_id,
                "component_class": component_class,
                "component_class_id": bbox["class_id"],
                "yolo_confidence": bbox["confidence"],
                "bbox_xyxy": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "crop_path": None,
                "orientation": None,
                "orientation_id": None,
                "orientation_confidence": None,
                "orientation_probabilities": None,
                "status": "ok",
            }

            if component_class not in orient_mod.ORIENTATION_REQUIRED_CLASSES:
                record["status"] = "orientation_not_required"
                components.append(record)
                draw.rectangle((x1, y1, x2, y2), outline="gray", width=2)
                draw_label(draw, (x1, max(0, y1 - 12)), component_class, font)
                continue

            if x2 <= x1 or y2 <= y1:
                record["status"] = "skipped_empty_bbox"
                components.append(record)
                continue

            crop = image.crop((x1, y1, x2, y2))
            crop_name = f"{stem}_{component_id:04d}_{orient_mod.safe_component_name(component_class)}.jpg"
            crop_path = crops_dir / crop_name
            crop.save(crop_path, quality=95)
            record.update(orient_mod.classify_orientation(crop, model, transform, device))
            record["crop_path"] = str(crop_path.relative_to(config.paths.output_dir))
            components.append(record)

            label = (
                f"{component_class} {record['orientation']} "
                f"{record['orientation_confidence']:.2f}"
            )
            draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
            draw_label(draw, (x1, max(0, y1 - 12)), label, font)

        payload = {
            "image_path": str(image_path),
            "image_width": width,
            "image_height": height,
            "source_bbox_txt": str(bbox_path),
            "orientation_weights": str(config.paths.orientation_weights),
            "resnet_orientation_classes": orient_mod.ORIENTATION_CLASS_TO_IDX,
            "orientation_required_classes": sorted(orient_mod.ORIENTATION_REQUIRED_CLASSES),
            "num_components": len(components),
            "num_orientation_predictions": sum(
                1 for component in components if component["orientation"] is not None
            ),
            "components": components,
        }
        out_json = config.paths.orientation_dir / f"{stem}.json"
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        annotated.save(config.paths.orientation_vis_dir / f"{stem}_annotated.jpg", quality=95)
        print(f"[orientation] {stem}: {payload['num_orientation_predictions']} predictions -> {out_json}")


def step_extract_lines(config, stems: list[str] | None, device: torch.device, force_ocr: bool) -> None:
    sync_component_bboxes(config.paths, direction="to_masked")
    stage1 = load_stage1(config)
    args = make_stage1_args(config, stems, device, force_ocr)
    stage1.run_combined_stage(device, args)


def step_generate_nodes(config, stems: list[str] | None, device: torch.device, force_ocr: bool) -> None:
    sync_component_bboxes(config.paths, direction="to_masked")
    stage1 = load_stage1(config)
    args = make_stage1_args(config, stems, device, force_ocr)
    stage1.run_component_anchor_nodes(args)


def step_export_touches(config, stems: list[str] | None) -> None:
    touch_mod = import_module_from_path(
        "export_node_component_touches",
        config.paths.repo_root / "2export_node_component_touches.py",
    )
    config.paths.node_touches_dir.mkdir(parents=True, exist_ok=True)
    selected = selected_stems(config.paths, stems)
    for stem in selected:
        npz_path = config.paths.node_data_dir / f"{stem}.npz"
        json_path = config.paths.node_data_dir / f"{stem}.json"
        if not npz_path.exists() or not json_path.exists():
            print(f"[node-touch] {stem}: missing node data, skipping")
            continue
        out_path, n_touches, n_nodes, n_added, warnings = touch_mod.export_one(
            stem=stem,
            npz_path=npz_path,
            json_path=json_path,
            bbox_dir=config.paths.component_bbox_dir,
            output_dir=config.paths.node_touches_dir,
            extra_endpoint_extend_px=config.getfloat("params", "extra_endpoint_extend_px", 10.0),
        )
        print(
            f"[node-touch] {stem}: touches={n_touches} nodes={n_nodes} "
            f"added_extended={n_added} -> {out_path}"
        )
        if warnings:
            print(f"[node-touch] {stem}: warnings={len(warnings)}")
    step_visualize_touches(config, stems)


def step_visualize_touches(config, stems: list[str] | None) -> None:
    vis_mod = import_module_from_path(
        "visualize_node_component_touches",
        config.paths.repo_root / "helper" / "visualize_node_component_touches.py",
    )
    for stem in selected_stems(config.paths, stems):
        touch_path = config.paths.node_touches_dir / f"{stem}.json"
        if not touch_path.exists():
            continue
        out_path, n_touches = vis_mod.visualize_one(
            touch_path=touch_path,
            image_dir=config.paths.masked_dir,
            output_dir=config.paths.node_touch_vis_dir,
            draw_labels=True,
            show_endpoints=True,
        )
        print(f"[touch-vis] {stem}: touches={n_touches} -> {out_path}")


def step_build_incidence(config, stems: list[str] | None) -> None:
    inc_mod = import_module_from_path(
        "build_incidence_matrix", config.paths.repo_root / "4build_incidence_matrix.py"
    )
    inc_mod.BBOX_DIR = config.paths.component_bbox_dir
    inc_mod.ORIENT_DIR = config.paths.orientation_dir
    inc_mod.TOUCH_DIR = config.paths.node_touches_dir
    inc_mod.IMAGE_DIR = config.paths.images_dir
    inc_mod.OUT_DIR = config.paths.incidence_dir
    inc_mod.TEXT_BBOX_DIR = config.paths.masked_no_text_dir / "_text_bboxes"
    inc_mod.COMBINED_LINES_DIR = config.paths.combined_lines_dir
    inc_mod.NODE_DATA_DIR = config.paths.node_data_dir
    inc_mod.CLOSE_SAME_NODE_TOUCH_MERGE_PX = config.getfloat(
        "params",
        "incidence_close_touch_merge_px",
        8.0,
    )
    inc_mod.TEXT_TOUCH_MARGIN_PX = config.getfloat("params", "text_touch_margin_px", 3.0)
    inc_mod.NEARBY_LINE_PIN_RESCUE_PX = config.getfloat(
        "params",
        "nearby_line_pin_rescue_px",
        20.0,
    )
    inc_mod.G_SPLIT_RATIO = config.getfloat("params", "g_split_ratio", 0.30)
    inc_mod.SHORT_ENDPOINT_MARGIN_PX = config.getint("params", "short_endpoint_margin_px", 1)
    inc_mod.MIN_BLACK_THRESHOLD = config.getfloat("params", "min_black_threshold", 35)
    inc_mod.MAX_BLACK_THRESHOLD = config.getfloat("params", "max_black_threshold", 90)
    inc_mod.BLACK_THRESHOLD_AVG_RATIO = config.getfloat(
        "params",
        "black_threshold_avg_ratio",
        0.35,
    )
    inc_mod.CLOSE_BBOX_PIN_RESCUE_PX = config.getfloat(
        "params",
        "close_bbox_pin_rescue_px",
        30.0,
    )
    inc_mod.MOS_PASSTHROUGH_ALIGN_PX = config.getfloat(
        "params",
        "mos_passthrough_align_px",
        8.0,
    )
    inc_mod.AMP_DIFF_ALIGN_RATIO = config.getfloat(
        "params",
        "amp_diff_align_ratio",
        0.10,
    )
    inc_mod.SHORT_LINE_MAX_GAP_PX = config.getint(
        "params",
        "short_line_max_gap_px",
        3,
    )
    config.paths.incidence_dir.mkdir(parents=True, exist_ok=True)

    ids = stems if stems else sorted(p.stem for p in config.paths.component_bbox_dir.glob("*.txt"))
    for stem in ids:
        result = inc_mod.build_image(stem)
        out_path = config.paths.incidence_dir / f"{stem}.json"
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        n_red = sum(1 for c in result["components"] if c["red_flags"])
        print(
            f"[incidence] {stem}: {len(result['components'])} components, "
            f"{len(result['nodes'])} nodes, red_flags={n_red} -> {out_path}"
        )
    step_visualize_incidence(config, stems)


def step_visualize_incidence(config, stems: list[str] | None) -> None:
    vis_mod = import_module_from_path(
        "visualize_incidence_matrix",
        config.paths.repo_root / "helper" / "visualize_incidence_matrix.py",
    )
    vis_mod.IMAGE_DIR = config.paths.images_dir
    vis_mod.INCIDENCE_DIR = config.paths.incidence_dir
    vis_mod.TOUCH_DIR = config.paths.node_touches_dir
    ids = stems if stems else sorted(p.stem for p in config.paths.incidence_dir.glob("*.json"))
    for stem in ids:
        try:
            out_path = vis_mod.visualize_image(stem, config.paths.incidence_vis_dir)
        except Exception as exc:
            print(f"[incidence-vis] {stem}: {exc}, skipping")
            continue
        print(f"[incidence-vis] {stem} -> {out_path}")


def step_build_netlist(config, stems: list[str] | None) -> None:
    net_mod = import_module_from_path("build_netlist", config.paths.repo_root / "5build_netlist.py")
    config.paths.netlist_dir.mkdir(parents=True, exist_ok=True)
    ids = stems if stems else sorted(p.stem for p in config.paths.incidence_dir.glob("*.json"))
    generated = []
    skipped = []
    for stem in ids:
        in_path = config.paths.incidence_dir / f"{stem}.json"
        if not in_path.exists():
            print(f"[netlist] {stem}: missing {in_path}, skipping")
            continue
        data = json.loads(in_path.read_text(encoding="utf-8"))
        red = [c["name"] for c in data["components"] if c.get("red_flags")]
        if red:
            skipped.append((stem, red))
            print(f"[netlist] {stem}: skipped, red_flags on {', '.join(red)}")
            continue
        out_path = config.paths.netlist_dir / f"{stem}.cir"
        out_path.write_text(net_mod.generate_netlist(data), encoding="utf-8")
        generated.append(stem)
        print(f"[netlist] {stem} -> {out_path}")
    print(f"[netlist] generated={len(generated)} skipped={len(skipped)}")


def run_steps(config, steps: list[str], stems: list[str] | None, device: torch.device, force_ocr: bool) -> None:
    for step in steps:
        if step == "detect_components":
            step_detect_components(config, stems, device, force_ocr)
        elif step == "detect_orientation":
            step_detect_orientation(config, stems, device)
        elif step == "extract_lines":
            step_extract_lines(config, stems, device, force_ocr)
        elif step == "generate_nodes":
            step_generate_nodes(config, stems, device, force_ocr)
        elif step == "export_touches":
            step_export_touches(config, stems)
        elif step == "build_incidence":
            step_build_incidence(config, stems)
        elif step == "build_netlist":
            step_build_netlist(config, stems)
        else:
            raise ValueError(f"Unknown step: {step}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified schematic-to-netlist pipeline runner.")
    parser.add_argument("--config", default="pipeline_config.yaml", help="pipeline config file")
    parser.add_argument("--image", type=Path, help="single image to add/process")
    parser.add_argument("--input-dir", type=Path, help="directory of images to add/process")
    parser.add_argument(
        "--step",
        choices=("all", *STEPS),
        default="all",
        help="single stage to run, or all stages; default runs through build_netlist",
    )
    parser.add_argument(
        "--from-step",
        choices=STEPS,
        help="run from this stage through build_netlist",
    )
    parser.add_argument("--stem", action="append", help="only process this image stem")
    parser.add_argument("--device", help="override device from config, e.g. cpu or cuda:0")
    parser.add_argument("--force-ocr", action="store_true", help="recompute OCR text boxes")
    parser.add_argument(
        "--no-copy-inputs",
        action="store_true",
        help="require inputs to already live in result/images",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_pipeline_config(args.config)
    ensure_output_dirs(config.paths)

    device = resolve_device(args.device or config.get("params", "device", "auto"))
    print(f"[init] config={config.config_path}")
    print(f"[init] output_dir={config.paths.output_dir}")
    print(f"[init] device={device}")

    if args.from_step and args.step != "all":
        raise ValueError("Use either --step for one stage or --from-step for a stage range, not both.")

    if args.from_step:
        start = STEPS.index(args.from_step)
        steps = list(STEPS[start:])
    elif args.step == "all":
        steps = list(STEPS)
    else:
        steps = [args.step]

    if args.image or args.input_dir:
        images = collect_images(args.image, args.input_dir)
    elif steps[0] == "detect_components" and config.paths.input_dir.exists():
        images = collect_images(None, config.paths.input_dir)
    else:
        images = []
    prepared_stems = None
    if images:
        prepared_stems = prepare_input_images(
            images,
            config.paths.images_dir,
            copy_inputs=not args.no_copy_inputs and config.getbool("outputs", "copy_inputs", True),
        )
        print(f"[input] prepared {len(prepared_stems)} image(s) in {config.paths.images_dir}")

    stems = args.stem or prepared_stems
    if stems:
        stems = sorted(dict.fromkeys(stems))

    run_steps(config, steps, stems, device, args.force_ocr)


if __name__ == "__main__":
    main()
