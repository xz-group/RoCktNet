from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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


@dataclass(frozen=True)
class PipelinePaths:
    repo_root: Path
    input_dir: Path
    output_dir: Path
    images_dir: Path
    component_bbox_dir: Path
    masked_dir: Path
    masked_no_text_dir: Path
    orientation_dir: Path
    orientation_crop_dir: Path
    orientation_vis_dir: Path
    combined_lines_dir: Path
    combined_vis_dir: Path
    nodes_dir: Path
    node_data_dir: Path
    node_vis_dir: Path
    node_touches_dir: Path
    node_touch_vis_dir: Path
    incidence_dir: Path
    incidence_vis_dir: Path
    netlist_dir: Path
    logs_dir: Path
    component_weights: Path
    junction_jump_weights: Path
    junction_jump_weights_2cls: Path
    orientation_weights: Path
    hawp_dir: Path


@dataclass(frozen=True)
class PipelineConfig:
    config_path: Path
    raw: dict[str, Any]
    paths: PipelinePaths

    def get(self, section: str, key: str, fallback: str | None = None) -> str:
        value = self.raw.get(section, {}).get(key, fallback)
        return fallback if value is None else str(value)

    def getfloat(self, section: str, key: str, fallback: float) -> float:
        value = self.raw.get(section, {}).get(key, fallback)
        return float(value)

    def getint(self, section: str, key: str, fallback: int) -> int:
        value = self.raw.get(section, {}).get(key, fallback)
        return int(value)

    def getbool(self, section: str, key: str, fallback: bool) -> bool:
        value = self.raw.get(section, {}).get(key, fallback)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "yes", "true", "on"}
        return bool(value)


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _load_raw_config(config_path: Path) -> dict[str, Any]:
    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required for YAML config files. Install it in the active environment."
            ) from exc
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"YAML config must contain a mapping at the top level: {config_path}")
        return data

    import configparser

    parser = configparser.ConfigParser()
    parser.read(config_path, encoding="utf-8")
    return {
        section: {key: value for key, value in parser.items(section)}
        for section in parser.sections()
    }


def load_pipeline_config(config_path: str | Path = "pipeline_config.yaml") -> PipelineConfig:
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    raw = _load_raw_config(config_path)

    base = config_path.parent
    cfg = PipelineConfig(config_path=config_path, raw=raw, paths=None)  # type: ignore[arg-type]
    output_dir = _resolve(base, cfg.get("paths", "output_dir", "result"))
    paths = PipelinePaths(
        repo_root=base,
        input_dir=_resolve(base, cfg.get("paths", "input_dir", "images")),
        output_dir=output_dir,
        images_dir=output_dir / "images",
        component_bbox_dir=output_dir / "component_bbox",
        masked_dir=output_dir / "masked_images",
        masked_no_text_dir=output_dir / "masked_no_text_images",
        orientation_dir=output_dir / "orientation",
        orientation_crop_dir=output_dir / "orientation" / "crops",
        orientation_vis_dir=output_dir / "orientation" / "visualizations",
        combined_lines_dir=output_dir / "combined_lines",
        combined_vis_dir=output_dir / "combined_visualizations",
        nodes_dir=output_dir / "nodes",
        node_data_dir=output_dir / "nodes" / "data",
        node_vis_dir=output_dir / "nodes" / "vis",
        node_touches_dir=output_dir / "node_touches",
        node_touch_vis_dir=output_dir / "node_touch_visualizations",
        incidence_dir=output_dir / "incidence_matrix",
        incidence_vis_dir=output_dir / "incidence_visualization",
        netlist_dir=output_dir / "netlist",
        logs_dir=output_dir / "logs",
        component_weights=_resolve(
            base,
            cfg.get(
                "paths",
                "component_weights",
                "pretrainedWeights/componentDetection/best.pt",
            ),
        ),
        junction_jump_weights=_resolve(
            base,
            cfg.get(
                "paths",
                "junction_jump_weights",
                "pretrainedWeights/junctionJumpDetection/best.pt",
            ),
        ),
        junction_jump_weights_2cls=_resolve(
            base,
            cfg.get(
                "paths",
                "junction_jump_weights_2cls",
                "pretrainedWeights/junctionJumpDetection/best2cls.pt",
            ),
        ),
        orientation_weights=_resolve(
            base,
            cfg.get(
                "paths",
                "orientation_weights",
                "pretrainedWeights/orientationDetection/best_model.pt",
            ),
        ),
        hawp_dir=_resolve(base, cfg.get("paths", "hawp_dir", "pretrainedWeights/hawp")),
    )
    return PipelineConfig(config_path=config_path, raw=raw, paths=paths)


def ensure_output_dirs(paths: PipelinePaths) -> None:
    for path in (
        paths.output_dir,
        paths.images_dir,
        paths.component_bbox_dir,
        paths.masked_dir,
        paths.masked_no_text_dir,
        paths.orientation_dir,
        paths.orientation_crop_dir,
        paths.orientation_vis_dir,
        paths.combined_lines_dir,
        paths.combined_vis_dir,
        paths.nodes_dir,
        paths.node_data_dir,
        paths.node_vis_dir,
        paths.node_touches_dir,
        paths.node_touch_vis_dir,
        paths.incidence_dir,
        paths.incidence_vis_dir,
        paths.netlist_dir,
        paths.logs_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def collect_images(image: str | Path | None, input_dir: str | Path | None) -> list[Path]:
    if image and input_dir:
        raise ValueError("Provide either --image or --input-dir, not both.")
    if image:
        path = Path(image).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input image not found: {path}")
        if path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image extension: {path}")
        return [path]
    if input_dir:
        root = Path(input_dir).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Input directory not found: {root}")
        return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return []


def prepare_input_images(images: Iterable[Path], images_dir: Path, copy_inputs: bool = True) -> list[str]:
    images_dir.mkdir(parents=True, exist_ok=True)
    stems: list[str] = []
    for src in images:
        dst = images_dir / src.name
        stems.append(src.stem)
        if copy_inputs:
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
        elif src.resolve() != dst.resolve():
            raise ValueError("--no-copy-inputs requires input images to already live in result/images.")
    return stems


def import_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selected_stems(paths: PipelinePaths, stems: list[str] | None) -> list[str]:
    if stems:
        return stems
    return sorted(p.stem for p in paths.images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def sync_component_bboxes(paths: PipelinePaths, direction: str = "from_masked") -> None:
    if direction not in {"from_masked", "to_masked"}:
        raise ValueError(f"Unknown bbox sync direction: {direction}")
    masked_bbox_dir = paths.masked_dir / "_bboxes"
    masked_bbox_dir.mkdir(parents=True, exist_ok=True)
    paths.component_bbox_dir.mkdir(parents=True, exist_ok=True)

    if direction == "from_masked":
        src_dir, dst_dir = masked_bbox_dir, paths.component_bbox_dir
    else:
        src_dir, dst_dir = paths.component_bbox_dir, masked_bbox_dir

    for src in sorted(src_dir.glob("*.txt")):
        shutil.copy2(src, dst_dir / src.name)
