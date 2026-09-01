#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable


def _pin_pyside6_qt_plugins() -> None:
    # conn_gui has PyQt6 + opencv-python alongside PySide6; both ship their
    # own Qt plugins and will hijack QT_PLUGIN_PATH on import, causing
    # "Could not load the Qt platform plugin 'windows'" at startup.
    # Point Qt at PySide6's plugin dir explicitly before any QtCore import.
    try:
        import PySide6  # top-level package only — no Qt loading yet
    except ImportError:
        return
    plugins_dir = Path(PySide6.__file__).resolve().parent / "plugins"
    platforms_dir = plugins_dir / "platforms"
    if plugins_dir.is_dir():
        os.environ["QT_PLUGIN_PATH"] = str(plugins_dir)
    if platforms_dir.is_dir():
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platforms_dir)


_pin_pyside6_qt_plugins()

import colorsys

import numpy as np
import yaml
from PIL import Image, ImageDraw
from PySide6.QtCore import Qt, QLineF, QProcess, QProcessEnvironment, QRectF
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPen,
    QPixmap,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsLineItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

# Pull shared constants from pipeline_common (lives next to this file).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_common import CLASS_NAMES, IMAGE_EXTS  # noqa: E402

# ---------- paths ----------

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "pipeline_config.yaml"
RESULT_DIR = REPO_ROOT / "output"
# Session config must sit next to pipeline_config.yaml: pipeline_common.py
# resolves every relative path in the YAML against the config file's parent
# directory, so a config under result/ breaks repo_root / output_dir / etc.
SESSION_CONFIG = REPO_ROOT / ".gui_session_config.yaml"
RUN_PIPELINE = REPO_ROOT / "run_pipeline.py"


# Must match run_pipeline.STEPS exactly.
STEPS = (
    "detect_components",
    "detect_orientation",
    "extract_lines",
    "generate_nodes",
    "export_touches",
    "build_incidence",
    "build_netlist",
)


# ---------- parameter schema ----------
#
# step:    --from-step to use when this parameter changes (None = no step
#          association, applies on next run regardless).
# type: 'float' | 'int' | 'bool' | 'choice' | 'str'
# range:   (min, max, single_step[, decimals]) for numeric widgets
# choices: list of strings for 'choice'
# tip:     tooltip shown next to the field

PARAM_SCHEMA: dict[str, dict[str, Any]] = {
    # Runtime ------------------------------------------------------------
    "device": {
        "step": None,
        "group": "Runtime",
        "type": "choice",
        "choices": ["auto", "cpu", "cuda", "cuda:0"],
        "editable": True,
        "tip": "auto picks CUDA when available, else CPU. cuda:N selects a specific GPU.",
    },
    # detect_components --------------------------------------------------
    "component_conf": {
        "step": "detect_components",
        "type": "float",
        "range": (0.0, 1.0, 0.01, 3),
        "tip": "YOLO component confidence threshold.",
    },
    "component_iou": {
        "step": "detect_components",
        "type": "float",
        "range": (0.0, 1.0, 0.01, 3),
        "tip": "YOLO NMS IoU threshold.",
    },
    "mask_pad": {
        "step": "detect_components",
        "type": "int",
        "range": (0, 50, 1),
        "tip": "Extra pixels to pad each bbox before masking.",
    },
    "mask_color": {
        "step": "detect_components",
        "type": "choice",
        "choices": ["white", "black"],
        "tip": "Fill color used when erasing detected components.",
    },
    # detect_orientation -------------------------------------------------
    "orientation_image_size": {
        "step": "detect_orientation",
        "type": "int",
        "range": (32, 1024, 16),
        "tip": "Crop resize used by the orientation classifier.",
    },
    # extract_lines ------------------------------------------------------
    "ocr_conf": {
        "step": "extract_lines",
        "type": "float",
        "range": (0.0, 1.0, 0.05, 3),
        "tip": "OCR confidence threshold for the text-bbox cache.",
    },
    "ocr_pad": {
        "step": "extract_lines",
        "type": "int",
        "range": (0, 50, 1),
        "tip": "Padding around OCR text bboxes.",
    },
    "combined_hawp_threshold": {
        "step": "extract_lines",
        "type": "float",
        "range": (0.0, 1.0, 0.01, 3),
        "tip": "HAWPv3 line score threshold.",
    },
    "text_distance": {
        "step": "extract_lines",
        "type": "float",
        "range": (0.0, 200.0, 1.0, 2),
        "tip": "Clearance from OCR text required for extra-line extraction.",
    },
    "min_line_length": {
        "step": "extract_lines",
        "type": "float",
        "range": (0.0, 200.0, 1.0, 2),
        "tip": "Discard line subsegments shorter than this.",
    },
    "whitespace_threshold": {
        "step": "extract_lines",
        "type": "int",
        "range": (0, 255, 1),
        "tip": "Grayscale value above which a pixel counts as whitespace.",
    },
    "whitespace_fraction": {
        "step": "extract_lines",
        "type": "float",
        "range": (0.0, 1.0, 0.01, 3),
        "tip": "Drop a line if at least this fraction of its samples are whitespace.",
    },
    # generate_nodes -----------------------------------------------------
    "extra_endpoint_extend_px": {
        "step": "generate_nodes",
        "type": "float",
        "range": (0.0, 200.0, 1.0, 2),
        "tip": "Extra extension used while exporting node-to-component touches.",
    },
    "node_union_dist": {
        "step": "generate_nodes",
        "type": "float",
        "range": (0.0, 200.0, 1.0, 2),
        "tip": "Distance threshold for grouping nearby line segments into one node.",
    },
    "jj_conf": {
        "step": "generate_nodes",
        "type": "float",
        "range": (0.0, 1.0, 0.01, 3),
        "tip": "Junction/jump YOLO confidence threshold.",
    },
    "jj_bbox_pad": {
        "step": "generate_nodes",
        "type": "float",
        "range": (0.0, 200.0, 0.5, 2),
        "tip": "Padding around junction/jump bboxes.",
    },
    "perp_tol_deg": {
        "step": "generate_nodes",
        "type": "float",
        "range": (0.0, 90.0, 1.0, 1),
        "tip": "Tolerance for perpendicularity check at crossovers (degrees).",
    },
    "crossover_probe_dist": {
        "step": "generate_nodes",
        "type": "float",
        "range": (0.0, 200.0, 0.5, 2),
        "tip": "Distance to probe along a crossover candidate.",
    },
    "crossover_probe_tol": {
        "step": "generate_nodes",
        "type": "float",
        "range": (0.0, 200.0, 0.5, 2),
        "tip": "Perpendicular tolerance for crossover probing.",
    },
    "crossover_min_line_len": {
        "step": "generate_nodes",
        "type": "float",
        "range": (0.0, 200.0, 0.5, 2),
        "tip": "Min line length to consider for crossover detection.",
    },
    "jump_parallel_tol": {
        "step": "generate_nodes",
        "type": "float",
        "range": (0.0, 90.0, 1.0, 1),
        "tip": "Parallel tolerance for jump detection (degrees).",
    },
    # build_incidence ----------------------------------------------------
    "g_split_ratio": {
        "step": "build_incidence",
        "type": "float",
        "range": (0.0, 1.0, 0.01, 3),
        "tip": "Fraction of MOS/BJT bbox length (along orientation axis) treated as the G/B side.",
    },
    "incidence_close_touch_merge_px": {
        "step": "build_incidence",
        "type": "float",
        "range": (0.0, 200.0, 0.5, 2),
        "tip": "Merge duplicate same-node corner touches within this distance.",
    },
    "text_touch_margin_px": {
        "step": "build_incidence",
        "type": "float",
        "range": (0.0, 50.0, 0.5, 2),
        "tip": "Drop extra touches that overlap OCR text boxes, with this edge margin.",
    },
    "nearby_line_pin_rescue_px": {
        "step": "build_incidence",
        "type": "float",
        "range": (0.0, 200.0, 0.5, 2),
        "tip": "When a pin is missing, search this far outside the expected edge for a line.",
    },
    "short_endpoint_margin_px": {
        "step": "build_incidence",
        "type": "int",
        "range": (0, 50, 1),
        "tip": "Pixel margin used when deciding two pins inside a bbox are shorted.",
    },
    "min_black_threshold": {
        "step": "build_incidence",
        "type": "float",
        "range": (0.0, 255.0, 1.0, 2),
        "tip": "Lower bound on the adaptive black-pixel threshold.",
    },
    "max_black_threshold": {
        "step": "build_incidence",
        "type": "float",
        "range": (0.0, 255.0, 1.0, 2),
        "tip": "Upper bound on the adaptive black-pixel threshold.",
    },
    "black_threshold_avg_ratio": {
        "step": "build_incidence",
        "type": "float",
        "range": (0.0, 1.0, 0.01, 3),
        "tip": "Fraction of the per-bbox average grayscale used as the threshold.",
    },
}


# Step-section ordering for the parameter panel (top to bottom).
STEP_SECTION_ORDER = [
    ("Runtime", None, "Applies on the next run regardless of step."),
    ("detect_components", "detect_components", "YOLO component detection."),
    ("detect_orientation", "detect_orientation", "ResNet18 orientation classifier."),
    ("extract_lines", "extract_lines", "HAWPv3 line extraction + OCR text removal."),
    (
        "generate_nodes",
        "generate_nodes",
        "Anchor lines to components + union-find into nodes; junction/jump detection.",
    ),
    (
        "build_incidence",
        "build_incidence",
        "Assign touches to pins, decide nets, raise red_flags.",
    ),
]


# ---------- visualization tabs ----------
#
# (display label, subdir relative to result/, primary filename glob).
# Primary glob is matched first; if missing, the tab shows a "not generated"
# placeholder.

VIS_TABS: list[tuple[str, str, str]] = [
    ("Input", "images", "{stem}.*"),
    ("Masked", "masked_images", "{stem}.*"),
    ("Orientation", "orientation/visualizations", "{stem}_annotated.*"),
    ("Lines", "combined_visualizations", "{stem}.*"),
    ("Nodes", "nodes/vis", "{stem}.*"),
    ("Touches", "node_touch_visualizations", "{stem}.*"),
    ("Incidence", "incidence_visualization", "{stem}_incidence_overlay.*"),
]


# ---------- helpers ----------


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML must be a top-level mapping: {path}")
    return data


def _dump_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def ensure_session_config() -> dict:
    # One-shot migration: earlier builds placed the session yaml under result/,
    # which broke relative-path resolution in pipeline_common. Move any leftover
    # file from the old location so the user keeps their edits.
    legacy = RESULT_DIR / ".gui_session_config.yaml"
    if legacy.exists() and not SESSION_CONFIG.exists():
        SESSION_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy), str(SESSION_CONFIG))

    if not SESSION_CONFIG.exists():
        if not DEFAULT_CONFIG.exists():
            raise FileNotFoundError(f"Missing {DEFAULT_CONFIG}")
        SESSION_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(DEFAULT_CONFIG, SESSION_CONFIG)
    return _load_yaml(SESSION_CONFIG)


def reset_session_from_default() -> dict:
    if not DEFAULT_CONFIG.exists():
        raise FileNotFoundError(f"Missing {DEFAULT_CONFIG}")
    SESSION_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DEFAULT_CONFIG, SESSION_CONFIG)
    return _load_yaml(SESSION_CONFIG)


def scan_red_flags(result_dir: Path) -> dict[str, list[tuple[str, list[str]]]]:
    out: dict[str, list[tuple[str, list[str]]]] = {}
    inc_dir = result_dir / "incidence_matrix"
    if not inc_dir.exists():
        return out
    for p in sorted(inc_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        flagged = []
        for c in data.get("components", []):
            rf = c.get("red_flags")
            if rf:
                flagged.append((c.get("name", "?"), list(rf)))
        if flagged:
            out[p.stem] = flagged
    return out


def list_all_stems(result_dir: Path) -> list[str]:
    images_dir = result_dir / "images"
    if not images_dir.exists():
        return []
    stems = set()
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
            ".webp",
        }:
            stems.add(p.stem)
    return sorted(stems)


def find_vis_file(
    result_dir: Path, subdir: str, stem: str, pattern: str
) -> Path | None:
    folder = result_dir / subdir
    if not folder.exists():
        return None
    needle = pattern.format(stem=stem)
    # Glob and return the first non-aux match (prefer exact-stem match).
    matches = sorted(folder.glob(needle))
    if not matches:
        return None
    # Prefer the file whose stem exactly equals the requested stem; falls back
    # to the first match (covers the "{stem}_annotated.*" / "_incidence_overlay.*"
    # pattern case where the wanted match already encodes the suffix).
    for m in matches:
        if m.stem == stem:
            return m
    return matches[0]


def find_image(images_dir: Path, stem: str) -> Path | None:
    if not images_dir.exists():
        return None
    for ext in IMAGE_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


# ---------- bbox file IO + masked-image regen ----------


def read_bbox_file(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("x1"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            x1, y1, x2, y2 = (float(parts[i]) for i in range(4))
            cls = int(float(parts[4]))
        except ValueError:
            continue
        conf = None
        if len(parts) > 5:
            try:
                conf = float(parts[5])
            except ValueError:
                conf = None
        out.append({"xyxy": [x1, y1, x2, y2], "class_id": cls, "confidence": conf})
    return out


def write_bbox_file(path: Path, bboxes: list[dict]) -> None:
    lines = ["x1 y1 x2 y2 cls conf"]
    for b in bboxes:
        x1, y1, x2, y2 = b["xyxy"]
        cls = int(b["class_id"])
        conf = b.get("confidence")
        # Manually-drawn bboxes have no detection confidence; write 1.0 so the
        # downstream loader (run_pipeline.read_component_bboxes) keeps treating
        # the column as numeric.
        if conf is None:
            conf = 1.0
        lines.append(f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {cls} {float(conf):.4f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sync_bbox_to_mirror(bbox_path: Path, mirror_dir: Path) -> None:
    mirror_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bbox_path, mirror_dir / bbox_path.name)


def regenerate_masked_image(
    images_dir: Path,
    masked_dir: Path,
    masked_no_text_dir: Path,
    stem: str,
    bboxes: list[dict],
    pad: int = 1,
    mask_color: str = "white",
) -> Path:
    src = find_image(images_dir, stem)
    if src is None:
        raise FileNotFoundError(
            f"Original image missing for stem '{stem}' in {images_dir}"
        )
    img = Image.open(src).convert("RGB")
    fill = (255, 255, 255) if mask_color == "white" else (0, 0, 0)
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for b in bboxes:
        x1, y1, x2, y2 = b["xyxy"]
        x1 = max(0, min(w, int(round(x1 - pad))))
        y1 = max(0, min(h, int(round(y1 - pad))))
        x2 = max(0, min(w, int(round(x2 + pad))))
        y2 = max(0, min(h, int(round(y2 + pad))))
        if x2 > x1 and y2 > y1:
            draw.rectangle((x1, y1, x2, y2), fill=fill)
    masked_dir.mkdir(parents=True, exist_ok=True)
    out = masked_dir / f"{stem}.jpg"
    img.save(out, quality=95)
    stale = masked_no_text_dir / f"{stem}.jpg"
    if stale.exists():
        try:
            stale.unlink()
        except OSError:
            pass
    return out


# ---------- per-class color palette ----------


_CLASS_COLORS: dict[int, QColor] = {}


def class_color(class_id: int) -> QColor:
    if class_id not in _CLASS_COLORS:
        # Spread hues evenly; multiply by golden-ratio-ish step for variety.
        hue = (class_id * 0.137508) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.78, 0.92)
        _CLASS_COLORS[class_id] = QColor(int(r * 255), int(g * 255), int(b * 255))
    return _CLASS_COLORS[class_id]


_NODE_COLORS: dict[int, QColor] = {}


def node_color(node_id: int) -> QColor:
    
    if node_id < 0:
        return QColor("#7f8c8d")
    if node_id not in _NODE_COLORS:
        # offset hue from 0 so node 0 isn't bright red (collision with red flags)
        hue = (0.07 + node_id * 0.137508) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.82, 0.95)
        _NODE_COLORS[node_id] = QColor(int(r * 255), int(g * 255), int(b * 255))
    return _NODE_COLORS[node_id]


# ---------- nodes/data IO ----------


def read_nodes_data(npz_path: Path, json_path: Path) -> dict:
    if not npz_path.exists() or not json_path.exists():
        raise FileNotFoundError(
            f"Missing nodes data for npz={npz_path} json={json_path}"
        )
    z = np.load(npz_path, allow_pickle=True)

    def _get(key, default_shape, dtype):
        if key in z.files:
            return np.asarray(z[key]).copy()
        return np.zeros(default_shape, dtype=dtype)

    return {
        "lines": _get("lines", (0, 2, 2), np.float32),
        "line_node_ids": _get("line_node_ids", (0,), np.int32),
        "valid_anchor_indices": _get("valid_anchor_indices", (0,), np.int32),
        "extended_anchor_lines": _get("extended_anchor_lines", (0, 2, 2), np.float32),
        "contact_points": _get("contact_points", (0, 2), np.float32),
        "anchor_line_indices": _get("anchor_line_indices", (0,), np.int32),
        "anchor_bbox_indices": _get("anchor_bbox_indices", (0,), np.int32),
        "junction_bboxes": _get("junction_bboxes", (0, 4), np.float32),
        "jump_bboxes": _get("jump_bboxes", (0, 4), np.float32),
        "json_data": json.loads(json_path.read_text(encoding="utf-8")),
    }


def _coerce_int(value, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def write_nodes_data(npz_path: Path, json_path: Path, state: dict) -> None:
    
    lines = np.asarray(state["lines"], dtype=np.float32).reshape(-1, 2, 2)
    line_node_ids = np.asarray(state["line_node_ids"], dtype=np.int32)
    valid_anchor_indices = np.asarray(state["valid_anchor_indices"], dtype=np.int32)
    extended_anchor_lines = np.asarray(
        state["extended_anchor_lines"], dtype=np.float32
    ).reshape(-1, 2, 2)
    contact_points = np.asarray(state["contact_points"], dtype=np.float32).reshape(
        -1, 2
    )
    anchor_line_indices = np.asarray(state["anchor_line_indices"], dtype=np.int32)
    anchor_bbox_indices = np.asarray(state["anchor_bbox_indices"], dtype=np.int32)
    junction_bboxes = np.asarray(state["junction_bboxes"], dtype=np.float32).reshape(
        -1, 4
    )
    jump_bboxes = np.asarray(state["jump_bboxes"], dtype=np.float32).reshape(-1, 4)

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        lines=lines,
        line_node_ids=line_node_ids,
        valid_anchor_indices=valid_anchor_indices,
        extended_anchor_lines=extended_anchor_lines,
        contact_points=contact_points,
        anchor_line_indices=anchor_line_indices,
        anchor_bbox_indices=anchor_bbox_indices,
        junction_bboxes=junction_bboxes,
        jump_bboxes=jump_bboxes,
    )

    json_data = dict(state["json_data"])
    json_data["num_lines"] = int(len(lines))
    json_data["num_valid_anchors"] = int(len(valid_anchor_indices))
    json_data["num_node_lines"] = int((line_node_ids >= 0).sum())
    json_data["num_nodes"] = len({int(x) for x in line_node_ids if int(x) >= 0})
    json_data["num_junctions"] = int(len(junction_bboxes))
    json_data["num_jumps"] = int(len(jump_bboxes))
    json_data["junction_bboxes"] = junction_bboxes.tolist()
    json_data["jump_bboxes"] = jump_bboxes.tolist()
    json_data["line_node_ids"] = line_node_ids.tolist()

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(json_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def add_line_to_state(
    state: dict, x1: float, y1: float, x2: float, y2: float, node_id: int
) -> int:
    
    lines = np.asarray(state["lines"], dtype=np.float32).reshape(-1, 2, 2)
    new_line = np.array([[[x1, y1], [x2, y2]]], dtype=np.float32)
    state["lines"] = np.concatenate([lines, new_line], axis=0)
    state["line_node_ids"] = np.concatenate(
        [
            np.asarray(state["line_node_ids"], dtype=np.int32),
            np.array([int(node_id)], dtype=np.int32),
        ]
    )
    return int(state["lines"].shape[0] - 1)


def remove_line_from_state(state: dict, line_idx: int) -> bool:
    
    n = int(state["lines"].shape[0])
    if not (0 <= line_idx < n):
        return False
    keep = np.ones(n, dtype=bool)
    keep[line_idx] = False
    state["lines"] = state["lines"][keep]
    state["line_node_ids"] = state["line_node_ids"][keep]

    def _shift(i: int) -> int | None:
        if i == line_idx:
            return None
        return i - 1 if i > line_idx else i

    # anchor_line_indices + parallel arrays (extended_anchor_lines, contact_points, anchor_bbox_indices)
    keep_anchor = []
    new_line_indices = []
    for i, li in enumerate(state["anchor_line_indices"]):
        shifted = _shift(int(li))
        if shifted is None:
            continue
        keep_anchor.append(i)
        new_line_indices.append(shifted)
    keep_anchor_arr = np.asarray(keep_anchor, dtype=np.int64)
    state["anchor_line_indices"] = np.asarray(new_line_indices, dtype=np.int32)
    state["anchor_bbox_indices"] = (
        state["anchor_bbox_indices"][keep_anchor_arr]
        if len(keep_anchor_arr)
        else np.zeros((0,), dtype=np.int32)
    )
    state["extended_anchor_lines"] = (
        state["extended_anchor_lines"][keep_anchor_arr]
        if len(keep_anchor_arr)
        else np.zeros((0, 2, 2), dtype=np.float32)
    )
    state["contact_points"] = (
        state["contact_points"][keep_anchor_arr]
        if len(keep_anchor_arr)
        else np.zeros((0, 2), dtype=np.float32)
    )

    # valid_anchor_indices is parallel to anchor_line_indices in the data we
    # observe; recompute it identically so the two stay in sync.
    new_valid = []
    for vi in state["valid_anchor_indices"]:
        shifted = _shift(int(vi))
        if shifted is not None:
            new_valid.append(shifted)
    state["valid_anchor_indices"] = np.asarray(new_valid, dtype=np.int32)

    # JSON anchors list mirrors the npz anchor arrays.
    new_anchors = []
    for a in state["json_data"].get("anchors", []):
        shifted = _shift(_coerce_int(a.get("line_idx"), -1))
        if shifted is None or shifted < 0:
            continue
        new = dict(a)
        new["line_idx"] = shifted
        new_anchors.append(new)
    state["json_data"]["anchors"] = new_anchors
    return True


def reassign_line_in_state(state: dict, line_idx: int, new_node_id: int) -> bool:
    n = int(state["lines"].shape[0])
    if not (0 <= line_idx < n):
        return False
    state["line_node_ids"] = np.asarray(state["line_node_ids"], dtype=np.int32)
    if int(state["line_node_ids"][line_idx]) == int(new_node_id):
        return False
    state["line_node_ids"][line_idx] = int(new_node_id)
    return True


def next_node_id(state: dict) -> int:
    arr = np.asarray(state["line_node_ids"], dtype=np.int32)
    pos = arr[arr >= 0]
    return int(pos.max() + 1) if len(pos) else 0


def regenerate_nodes_vis(result_dir: Path, stem: str) -> Path | None:
    src = find_image(result_dir / "masked_images", stem)
    if src is None:
        src = find_image(result_dir / "images", stem)
    if src is None:
        return None
    npz_path = result_dir / "nodes" / "data" / f"{stem}.npz"
    if not npz_path.exists():
        return None
    try:
        z = np.load(npz_path, allow_pickle=True)
    except Exception:
        return None
    lines = np.asarray(z["lines"]).reshape(-1, 2, 2) if "lines" in z.files else None
    line_node_ids = (
        np.asarray(z["line_node_ids"]) if "line_node_ids" in z.files else None
    )
    junction_bboxes = (
        np.asarray(z["junction_bboxes"]).reshape(-1, 4)
        if "junction_bboxes" in z.files
        else None
    )
    jump_bboxes = (
        np.asarray(z["jump_bboxes"]).reshape(-1, 4)
        if "jump_bboxes" in z.files
        else None
    )

    img = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(img)

    bbox_path = result_dir / "component_bbox" / f"{stem}.txt"
    for b in read_bbox_file(bbox_path):
        x1, y1, x2, y2 = (int(round(v)) for v in b["xyxy"])
        draw.rectangle((x1, y1, x2, y2), outline=(180, 180, 180), width=1)

    if junction_bboxes is not None:
        for jb in junction_bboxes:
            x1, y1, x2, y2 = (int(round(float(v))) for v in jb)
            draw.rectangle((x1, y1, x2, y2), outline=(0, 200, 0), width=1)
    if jump_bboxes is not None:
        for jb in jump_bboxes:
            x1, y1, x2, y2 = (int(round(float(v))) for v in jb)
            draw.rectangle((x1, y1, x2, y2), outline=(220, 0, 0), width=1)

    # X-jumps live only in the override file — show them so the user can
    # see all their manual annotations reflected here.
    overrides = read_manual_jj_overrides(stem)
    for xb in overrides.get("added_x_jumps", []):
        if len(xb) < 4:
            continue
        x1, y1, x2, y2 = (int(round(float(v))) for v in xb[:4])
        draw.rectangle((x1, y1, x2, y2), outline=(142, 68, 173), width=2)

    if lines is not None and lines.size and line_node_ids is not None:
        for i in range(len(lines)):
            nid = int(line_node_ids[i])
            if nid < 0:
                color = (128, 128, 128)
            else:
                qc = node_color(nid)
                color = (qc.red(), qc.green(), qc.blue())
            (x1, y1), (x2, y2) = lines[i]
            draw.line(
                [
                    (int(round(float(x1))), int(round(float(y1)))),
                    (int(round(float(x2))), int(round(float(y2)))),
                ],
                fill=color,
                width=2,
            )

    out_path = result_dir / "nodes" / "vis" / f"{stem}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


# ---------- touches IO + edit helpers ----------


def read_node_touches(result_dir: Path, stem: str) -> dict | None:
    p = result_dir / "node_touches" / f"{stem}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_node_touches(result_dir: Path, stem: str, data: dict) -> Path:
    # Recompute the simple counts so they stay consistent with `nodes`.
    nodes = data.get("nodes") or []
    for n in nodes:
        n["num_touches"] = len(n.get("touches") or [])
    data["num_nodes"] = len(nodes)
    data["num_touches"] = sum(int(n.get("num_touches", 0)) for n in nodes)
    out = result_dir / "node_touches" / f"{stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return out


def find_or_make_node_entry(data: dict, node_id: int) -> dict:
    
    nodes = data.setdefault("nodes", [])
    for n in nodes:
        if int(n.get("node_id", -1)) == int(node_id):
            return n
    entry = {"node_id": int(node_id), "num_touches": 0, "touches": []}
    nodes.append(entry)
    nodes.sort(key=lambda x: int(x.get("node_id", 0)))
    return entry


def next_touches_node_id(data: dict) -> int:
    ids = [int(n.get("node_id", -1)) for n in data.get("nodes") or []]
    ids += [
        int(n.get("node_id", -1)) for n in data.get("removed_single_bbox_nodes") or []
    ]
    ids = [i for i in ids if i >= 0]
    return (max(ids) + 1) if ids else 0


def nearest_bbox_edge(
    click_xy: tuple[float, float], bboxes: list[dict]
) -> tuple[int, str, list[float], float] | None:
    if not bboxes:
        return None
    cx, cy = float(click_xy[0]), float(click_xy[1])
    best: tuple[int, str, list[float], float] | None = None
    for idx, b in enumerate(bboxes):
        x1, y1, x2, y2 = (float(v) for v in b["xyxy"])
        candidates = [
            ("left", x1, max(y1, min(y2, cy))),
            ("right", x2, max(y1, min(y2, cy))),
            ("top", max(x1, min(x2, cx)), y1),
            ("bottom", max(x1, min(x2, cx)), y2),
        ]
        for edge_name, ex, ey in candidates:
            d = ((cx - ex) ** 2 + (cy - ey) ** 2) ** 0.5
            if best is None or d < best[3]:
                best = (idx, edge_name, [ex, ey], d)
    return best


def build_manual_touch(
    node_id: int, bbox_idx: int, bbox: dict, edge_name: str, contact_xy: list[float]
) -> dict:
    return {
        "node_id": int(node_id),
        "line_idx": -1,  # manual, not derived from a line
        "endpoint_idx": 0,
        "endpoint_xy": [float(contact_xy[0]), float(contact_xy[1])],
        "endpoint_to_contact_dist": 0.0,
        "contact_xy": [float(contact_xy[0]), float(contact_xy[1])],
        "component_bbox_idx": int(bbox_idx),
        "component_bbox_xyxy": [float(v) for v in bbox["xyxy"]],
        "component_class": int(bbox["class_id"]),
        "component_conf": float(bbox.get("confidence") or 1.0),
        "edge": edge_name,
        "anchor_dist": None,
        "extension_dist": None,
        "source": "manual",
        "sources": ["manual"],
        "num_contributors": 1,
        "contributors": [
            {
                "line_idx": -1,
                "endpoint_idx": 0,
                "endpoint_xy": [float(contact_xy[0]), float(contact_xy[1])],
                "contact_xy": [float(contact_xy[0]), float(contact_xy[1])],
                "endpoint_to_contact_dist": 0.0,
                "anchor_dist": None,
                "extension_dist": None,
                "source": "manual",
            }
        ],
    }


# ---------- manual jump/junction overrides ----------
#
# The pipeline's generate_nodes re-runs junction/jump YOLO every time, which
# would clobber any bboxes the user adds in the GUI. To make manual additions
# stick *and* take effect in the union-find / crossover-split logic, we write
# them to output/manual_jj_overrides/{stem}.json;
# step1_component_line_detection._apply_manual_jj_overrides reads that file and
# unions the entries with the YOLO output before building line_node_ids. The
# override file is purely additive.

MANUAL_JJ_OVERRIDES_DIR = RESULT_DIR / "manual_jj_overrides"
_JJ_MATCH_EPSILON = 0.75  # pixels — tolerance for matching override entries to npz rows


def read_manual_jj_overrides(stem: str) -> dict:
    p = MANUAL_JJ_OVERRIDES_DIR / f"{stem}.json"
    empty = {"added_jumps": [], "added_junctions": [], "added_x_jumps": []}
    if not p.exists():
        return empty
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return empty
    return {
        "added_jumps": list(data.get("added_jumps") or []),
        "added_junctions": list(data.get("added_junctions") or []),
        "added_x_jumps": list(data.get("added_x_jumps") or []),
    }


def write_manual_jj_overrides(
    stem: str,
    added_jumps: list[list[float]],
    added_junctions: list[list[float]],
    added_x_jumps: list[list[float]] | None = None,
) -> Path:
    MANUAL_JJ_OVERRIDES_DIR.mkdir(parents=True, exist_ok=True)
    out = MANUAL_JJ_OVERRIDES_DIR / f"{stem}.json"
    payload = {
        "added_jumps": [list(map(float, row)) for row in added_jumps],
        "added_junctions": [list(map(float, row)) for row in added_junctions],
        "added_x_jumps": [list(map(float, row)) for row in (added_x_jumps or [])],
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def _bbox_matches(a, b, eps: float = _JJ_MATCH_EPSILON) -> bool:
    return all(abs(float(a[i]) - float(b[i])) <= eps for i in range(4))


def jj_source_lookup(stem: str, bboxes: np.ndarray, kind: str) -> list[str]:
    overrides = read_manual_jj_overrides(stem)
    key = "added_jumps" if kind == "jump" else "added_junctions"
    refs = overrides.get(key, [])
    out: list[str] = []
    for row in bboxes:
        tag = "yolo"
        for ref in refs:
            if _bbox_matches(row, ref):
                tag = "manual"
                break
        out.append(tag)
    return out


# ---------- widgets ----------


class StemListPanel(QWidget):
    

    def __init__(self, result_dir: Path, parent=None):
        super().__init__(parent)
        self.result_dir = result_dir
        self._red_map: dict[str, list[tuple[str, list[str]]]] = {}
        self._all_stems: list[str] = []
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.only_red = QCheckBox("Only red-flagged")
        self.only_red.setChecked(True)
        self.only_red.toggled.connect(self._populate_list)
        layout.addWidget(self.only_red)

        refresh_btn = QPushButton("Re-scan red flags")
        refresh_btn.clicked.connect(self.refresh)
        layout.addWidget(refresh_btn)

        self.summary = QLabel("...")
        self.summary.setStyleSheet("color: #666;")
        layout.addWidget(self.summary)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.ExtendedSelection)
        layout.addWidget(self.list_widget, 1)

        hint = QLabel("Shift/Ctrl-click for multi-select")
        hint.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(hint)

    def refresh(self) -> None:
        self._red_map = scan_red_flags(self.result_dir)
        self._all_stems = list_all_stems(self.result_dir)
        self._populate_list()

    def _populate_list(self) -> None:
        prev = self.current_selection()
        self.list_widget.clear()
        only_red = self.only_red.isChecked()
        stems = list(self._red_map.keys()) if only_red else self._all_stems
        stems = sorted(stems)
        for stem in stems:
            flags = self._red_map.get(stem)
            if flags:
                count = sum(len(rs) for _, rs in flags)
                item = QListWidgetItem(
                    f"{stem}    [{count} flag{'s' if count != 1 else ''}]"
                )
                item.setForeground(QColor("#c0392b"))
            else:
                item = QListWidgetItem(stem)
            item.setData(Qt.UserRole, stem)
            self.list_widget.addItem(item)
        # Re-select previously selected entries that survived the filter.
        if prev:
            for i in range(self.list_widget.count()):
                it = self.list_widget.item(i)
                if it.data(Qt.UserRole) in prev:
                    it.setSelected(True)
        n_red = len(self._red_map)
        n_total = len(self._all_stems)
        self.summary.setText(f"{n_red} red-flagged / {n_total} total")

    def red_flags_for(self, stem: str) -> list[tuple[str, list[str]]]:
        return self._red_map.get(stem, [])

    def current_selection(self) -> list[str]:
        return [it.data(Qt.UserRole) for it in self.list_widget.selectedItems()]

    def select_only(self, stem: str) -> None:
        self.list_widget.clearSelection()
        for i in range(self.list_widget.count()):
            it = self.list_widget.item(i)
            if it.data(Qt.UserRole) == stem:
                it.setSelected(True)
                self.list_widget.setCurrentItem(it)
                return


class ImageView(QScrollArea):
    

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._label = QLabel("(no image)")
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet("color: #888; padding: 24px;")
        self.setWidget(self._label)
        self._pixmap: QPixmap | None = None

    def set_image(self, path: Path | None) -> None:
        if path is None or not path.exists():
            self._pixmap = None
            self._label.setPixmap(QPixmap())
            self._label.setText("(not generated)")
            return
        pm = QPixmap(str(path))
        if pm.isNull():
            self._pixmap = None
            self._label.setText(f"(failed to load: {path.name})")
            return
        self._pixmap = pm
        self._apply_pixmap()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_pixmap()

    def _apply_pixmap(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        viewport_w = max(64, self.viewport().width() - 8)
        scaled = self._pixmap.scaledToWidth(viewport_w, Qt.SmoothTransformation)
        self._label.setPixmap(scaled)
        self._label.setText("")


# ---------- bbox editor ----------


class BboxRectItem(QGraphicsRectItem):
    def __init__(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        class_id: int,
        confidence: float | None = None,
    ):
        super().__init__(QRectF(x1, y1, x2 - x1, y2 - y1))
        self.class_id = int(class_id)
        self.confidence = confidence
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self._apply_pens()

    def set_class(self, class_id: int) -> None:
        self.class_id = int(class_id)
        self._apply_pens()

    def to_dict(self) -> dict:
        r = self.rect()
        x1, y1 = r.x(), r.y()
        x2, y2 = x1 + r.width(), y1 + r.height()
        # Normalize so x1<=x2 and y1<=y2 (negative-width rects can sneak in
        # if the user drag-draws right-to-left).
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return {
            "xyxy": [x1, y1, x2, y2],
            "class_id": self.class_id,
            "confidence": self.confidence,
        }

    def _apply_pens(self) -> None:
        color = class_color(self.class_id)
        outline_width = 3 if self.isSelected() else 2
        pen = QPen(color, outline_width)
        pen.setCosmetic(True)
        self.setPen(pen)
        fill = QColor(color)
        fill.setAlpha(48)
        self.setBrush(QBrush(fill))

    def itemChange(self, change, value):  # type: ignore[override]
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self._apply_pens()
        return super().itemChange(change, value)


class BboxCanvas(QGraphicsView):

    SELECT = 0
    DRAW = 1

    MIN_NEW_SIZE = 3.0  # px in scene coords; drops accidental zero-size drags.

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._tool: int = self.SELECT
        self._drawing = False
        self._draw_start: tuple[float, float] | None = None
        self._draw_preview: QGraphicsRectItem | None = None
        # Insertion-ordered bbox list. QGraphicsScene.items() sorts by z-stack,
        # not insertion order, so we track our own list for deterministic
        # serialization.
        self._bboxes: list[BboxRectItem] = []
        self.new_class_id: int = 8  # resistor by default; updated by toolbar
        # callbacks
        self.on_dirty: Callable[[], None] = lambda: None
        self.on_selection_changed: Callable[[BboxRectItem | None], None] = (
            lambda _: None
        )
        self._scene.selectionChanged.connect(self._handle_selection_change)
        self.set_tool(self.SELECT)

    # ---- image / bbox loading ----

    def reset(self, image_path: Path | None) -> None:
        self._drawing = False
        self._draw_preview = None
        self._draw_start = None
        self._scene.clear()
        self._bboxes.clear()
        self._pixmap_item = None
        if image_path is not None and image_path.exists():
            pm = QPixmap(str(image_path))
            if not pm.isNull():
                self._pixmap_item = self._scene.addPixmap(pm)
                self._pixmap_item.setZValue(-1)
                self._scene.setSceneRect(0, 0, pm.width(), pm.height())
                self.resetTransform()
                self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def add_bbox(self, bbox: dict) -> BboxRectItem:
        x1, y1, x2, y2 = bbox["xyxy"]
        item = BboxRectItem(x1, y1, x2, y2, bbox["class_id"], bbox.get("confidence"))
        self._scene.addItem(item)
        self._bboxes.append(item)
        return item

    def all_bboxes(self) -> list[BboxRectItem]:
        # Filter out items that were removed from the scene (paranoia: keeps
        # the list in sync even if delete paths are bypassed).
        self._bboxes = [it for it in self._bboxes if it.scene() is self._scene]
        return list(self._bboxes)

    def selected_bboxes(self) -> list[BboxRectItem]:
        return [
            it for it in self._scene.selectedItems() if isinstance(it, BboxRectItem)
        ]

    # ---- tool switching ----

    def set_tool(self, tool: int) -> None:
        self._tool = tool
        if tool == self.SELECT:
            self.setDragMode(QGraphicsView.RubberBandDrag)
            self.viewport().setCursor(Qt.ArrowCursor)
        else:
            self.setDragMode(QGraphicsView.NoDrag)
            self.viewport().setCursor(Qt.CrossCursor)

    # ---- mouse / keyboard ----

    def mousePressEvent(self, event):  # type: ignore[override]
        if self._tool == self.DRAW and event.button() == Qt.LeftButton:
            sp = self.mapToScene(event.position().toPoint())
            self._drawing = True
            self._draw_start = (sp.x(), sp.y())
            preview = QGraphicsRectItem(QRectF(sp.x(), sp.y(), 0, 0))
            pen = QPen(QColor("#c0392b"), 2, Qt.DashLine)
            pen.setCosmetic(True)
            preview.setPen(pen)
            self._scene.addItem(preview)
            self._draw_preview = preview
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # type: ignore[override]
        if (
            self._drawing
            and self._draw_preview is not None
            and self._draw_start is not None
        ):
            sp = self.mapToScene(event.position().toPoint())
            x0, y0 = self._draw_start
            x = min(x0, sp.x())
            y = min(y0, sp.y())
            w = abs(sp.x() - x0)
            h = abs(sp.y() - y0)
            self._draw_preview.setRect(QRectF(x, y, w, h))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # type: ignore[override]
        if self._drawing and event.button() == Qt.LeftButton:
            self._drawing = False
            r = self._draw_preview.rect() if self._draw_preview is not None else None
            if self._draw_preview is not None:
                self._scene.removeItem(self._draw_preview)
                self._draw_preview = None
            if (
                r is not None
                and r.width() >= self.MIN_NEW_SIZE
                and r.height() >= self.MIN_NEW_SIZE
            ):
                x1, y1 = r.x(), r.y()
                x2, y2 = x1 + r.width(), y1 + r.height()
                # Clamp to image bounds.
                if self._pixmap_item is not None:
                    pm = self._pixmap_item.pixmap()
                    x1 = max(0.0, min(pm.width(), x1))
                    y1 = max(0.0, min(pm.height(), y1))
                    x2 = max(0.0, min(pm.width(), x2))
                    y2 = max(0.0, min(pm.height(), y2))
                if (x2 - x1) >= self.MIN_NEW_SIZE and (y2 - y1) >= self.MIN_NEW_SIZE:
                    item = self.add_bbox(
                        {
                            "xyxy": [x1, y1, x2, y2],
                            "class_id": self.new_class_id,
                            "confidence": None,
                        }
                    )
                    # auto-select the newly drawn box so the user can change
                    # class or hit Delete without an extra click
                    self._scene.clearSelection()
                    item.setSelected(True)
                    self.on_dirty()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):  # type: ignore[override]
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if self.delete_selected() > 0:
                event.accept()
                return
        super().keyPressEvent(event)

    def wheelEvent(self, event):  # type: ignore[override]
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            factor = 1.15 if delta > 0 else 1 / 1.15
            self.scale(factor, factor)
            event.accept()
            return
        super().wheelEvent(event)

    # ---- mutators ----

    def delete_selected(self) -> int:
        sel = self.selected_bboxes()
        for it in sel:
            self._scene.removeItem(it)
            try:
                self._bboxes.remove(it)
            except ValueError:
                pass
        if sel:
            self.on_dirty()
        return len(sel)

    def set_class_of_selected(self, class_id: int) -> bool:
        changed = False
        for it in self.selected_bboxes():
            if it.class_id != class_id:
                it.set_class(class_id)
                changed = True
        if changed:
            self.on_dirty()
        return changed

    def _handle_selection_change(self) -> None:
        sel = self.selected_bboxes()
        self.on_selection_changed(sel[0] if sel else None)


class BboxEditor(QWidget):

    def __init__(
        self, result_dir: Path, on_commit: Callable[[str, str], None], parent=None
    ):
        super().__init__(parent)
        self.result_dir = result_dir
        self.on_commit = on_commit
        self._stem: str | None = None
        self._dirty = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        tb = QHBoxLayout()
        self.tool_select = QPushButton("Select")
        self.tool_select.setCheckable(True)
        self.tool_select.setChecked(True)
        self.tool_draw = QPushButton("Draw")
        self.tool_draw.setCheckable(True)
        group = QButtonGroup(self)
        group.setExclusive(True)
        group.addButton(self.tool_select)
        group.addButton(self.tool_draw)
        self.tool_select.toggled.connect(self._on_tool_changed)
        self.tool_draw.toggled.connect(self._on_tool_changed)
        tb.addWidget(self.tool_select)
        tb.addWidget(self.tool_draw)

        tb.addSpacing(12)
        tb.addWidget(QLabel("Class:"))
        self.class_combo = QComboBox()
        for cid in sorted(CLASS_NAMES.keys()):
            self.class_combo.addItem(f"{cid:>2}  {CLASS_NAMES[cid]}", cid)
        self.class_combo.currentIndexChanged.connect(self._on_class_changed)
        tb.addWidget(self.class_combo)

        tb.addStretch(1)

        self.dirty_label = QLabel("clean")
        self.dirty_label.setStyleSheet("color: #888;")
        tb.addWidget(self.dirty_label)

        self.revert_btn = QPushButton("Revert")
        self.revert_btn.clicked.connect(self._revert)
        tb.addWidget(self.revert_btn)

        self.commit_btn = QPushButton("Commit + re-run")
        self.commit_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 4px 10px; }"
        )
        self.commit_btn.clicked.connect(self._commit)
        tb.addWidget(self.commit_btn)

        layout.addLayout(tb)

        self.canvas = BboxCanvas()
        self.canvas.new_class_id = int(self.class_combo.currentData())
        self.canvas.on_dirty = self._mark_dirty
        self.canvas.on_selection_changed = self._on_selection_changed
        layout.addWidget(self.canvas, 1)

        hint = QLabel(
            "Select tool: click bbox → change class via dropdown or press Delete. "
            "Draw tool: click-drag a rectangle. Ctrl+wheel to zoom."
        )
        hint.setStyleSheet("color: #888; font-size: 10px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

    # ---- public API ----

    def stem(self) -> str | None:
        return self._stem

    def is_dirty(self) -> bool:
        return self._dirty

    def load_stem(self, stem: str) -> None:
        self._stem = stem
        image_path = find_image(self.result_dir / "images", stem)
        self.canvas.reset(image_path)
        bbox_path = self.result_dir / "component_bbox" / f"{stem}.txt"
        for b in read_bbox_file(bbox_path):
            self.canvas.add_bbox(b)
        self._dirty = False
        self._update_dirty_label()

    # ---- callbacks ----

    def _on_tool_changed(self) -> None:
        if self.tool_draw.isChecked():
            self.canvas.set_tool(BboxCanvas.DRAW)
        else:
            self.canvas.set_tool(BboxCanvas.SELECT)

    def _on_class_changed(self) -> None:
        cid = self.class_combo.currentData()
        if cid is None:
            return
        cid = int(cid)
        self.canvas.new_class_id = cid
        # If a bbox is currently selected, treat the dropdown as a class editor
        # for that bbox too.
        if self.canvas.selected_bboxes():
            self.canvas.set_class_of_selected(cid)

    def _on_selection_changed(self, bbox: BboxRectItem | None) -> None:
        if bbox is None:
            return
        idx = self.class_combo.findData(bbox.class_id)
        if idx >= 0:
            self.class_combo.blockSignals(True)
            self.class_combo.setCurrentIndex(idx)
            self.class_combo.blockSignals(False)

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._update_dirty_label()

    def _update_dirty_label(self) -> None:
        if self._dirty:
            self.dirty_label.setText("● unsaved edits")
            self.dirty_label.setStyleSheet("color: #c0392b; font-weight: bold;")
        else:
            self.dirty_label.setText("clean")
            self.dirty_label.setStyleSheet("color: #888;")

    def _revert(self) -> None:
        if self._stem is None:
            return
        if self._dirty:
            ok = QMessageBox.question(
                self,
                "Revert bbox edits",
                "Discard pending bbox edits and reload from disk?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if ok != QMessageBox.Yes:
                return
        self.load_stem(self._stem)

    def _commit(self) -> None:
        if self._stem is None:
            return
        if not self._dirty:
            QMessageBox.information(self, "Nothing to commit", "No bbox edits pending.")
            return
        n_bboxes = len(self.canvas.all_bboxes())
        msg = (
            f"Commit bbox edits for {self._stem} and re-run pipeline "
            f"from extract_lines?\n\n"
            f"Will:\n"
            f"  • write component_bbox/{self._stem}.txt ({n_bboxes} bboxes)\n"
            f"  • regenerate masked_images/{self._stem}.jpg from the original image\n"
            f"  • delete masked_no_text_images/{self._stem}.jpg (regenerated by pipeline)\n"
            f"  • re-run extract_lines → … → build_netlist\n\n"
            f"WARNING: any later-stage manual edits (lines / nodes / touches) "
            f"for {self._stem} will be overwritten."
        )
        ok = QMessageBox.question(
            self,
            "Commit + re-run",
            msg,
            QMessageBox.Yes | QMessageBox.No,
        )
        if ok != QMessageBox.Yes:
            return
        try:
            self._do_commit()
        except Exception as exc:
            QMessageBox.critical(self, "Commit failed", str(exc))
            return
        self._dirty = False
        self._update_dirty_label()
        self.on_commit(self._stem, "extract_lines")

    def _do_commit(self) -> None:
        assert self._stem is not None
        bboxes = [it.to_dict() for it in self.canvas.all_bboxes()]
        bbox_path = self.result_dir / "component_bbox" / f"{self._stem}.txt"
        write_bbox_file(bbox_path, bboxes)
        mirror_dir = self.result_dir / "masked_images" / "_bboxes"
        sync_bbox_to_mirror(bbox_path, mirror_dir)
        # mask_pad / mask_color come from the live session config so the
        # re-mask matches what the pipeline would have produced.
        cfg = _load_yaml(SESSION_CONFIG) if SESSION_CONFIG.exists() else {}
        params = cfg.get("params", {}) or {}
        try:
            pad = int(params.get("mask_pad", 1))
        except (TypeError, ValueError):
            pad = 1
        mask_color = str(params.get("mask_color", "white"))
        regenerate_masked_image(
            images_dir=self.result_dir / "images",
            masked_dir=self.result_dir / "masked_images",
            masked_no_text_dir=self.result_dir / "masked_no_text_images",
            stem=self._stem,
            bboxes=bboxes,
            pad=pad,
            mask_color=mask_color,
        )


class MaskedTab(QWidget):

    def __init__(
        self, result_dir: Path, on_commit: Callable[[str, str], None], parent=None
    ):
        super().__init__(parent)
        self.result_dir = result_dir
        self._stem: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        tb = QHBoxLayout()
        self.edit_toggle = QCheckBox("Edit bboxes")
        self.edit_toggle.toggled.connect(self._on_mode_changed)
        tb.addWidget(self.edit_toggle)
        tb.addStretch(1)
        layout.addLayout(tb)

        self.stack = QStackedWidget()
        self.view = ImageView()
        self.editor = BboxEditor(result_dir, on_commit)
        self.stack.addWidget(self.view)
        self.stack.addWidget(self.editor)
        self.stack.setCurrentWidget(self.view)
        layout.addWidget(self.stack, 1)

    def show_stem(self, stem: str | None) -> None:
        if stem is None:
            self._stem = None
            self.view.set_image(None)
            return
        # If user has unsaved edits and we're switching to a different stem,
        # warn before clobbering.
        if (
            self.edit_toggle.isChecked()
            and self.editor.is_dirty()
            and self.editor.stem() is not None
            and self.editor.stem() != stem
        ):
            QMessageBox.warning(
                self,
                "Edits discarded",
                f"Switching away from {self.editor.stem()} with unsaved bbox "
                f"edits — they have been discarded.",
            )
        self._stem = stem
        path = find_vis_file(self.result_dir, "masked_images", stem, "{stem}.*")
        self.view.set_image(path)
        if self.edit_toggle.isChecked():
            self.editor.load_stem(stem)

    def _on_mode_changed(self, checked: bool) -> None:
        if checked:
            self.stack.setCurrentWidget(self.editor)
            if self._stem is not None:
                self.editor.load_stem(self._stem)
        else:
            self.stack.setCurrentWidget(self.view)


# ---------- orientation editor ----------


class OrientationBboxItem(QGraphicsRectItem):

    def __init__(self, component: dict, required: bool):
        bx = component["bbox_xyxy"]
        x1, y1, x2, y2 = bx["x1"], bx["y1"], bx["x2"], bx["y2"]
        super().__init__(QRectF(x1, y1, x2 - x1, y2 - y1))
        self.component_id = int(component["component_id"])
        self.component_class = str(component["component_class"])
        self.orientation: str | None = component.get("orientation")
        self.required = required
        self.original_orientation = self.orientation
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)

        # Letter overlay (large, centered, with outline-ish contrast).
        self._letter = QGraphicsTextItem(parent=self)
        font = QFont("Arial", 22, QFont.Bold)
        self._letter.setFont(font)
        self._apply_visuals()

    def set_orientation(self, orientation: str | None) -> None:
        self.orientation = orientation
        self._apply_visuals()

    def is_modified(self) -> bool:
        return self.orientation != self.original_orientation

    def _apply_visuals(self) -> None:
        r = self.rect()
        if self.required:
            if self.orientation:
                # Green if set; mildly darker if it's been edited by user.
                color = (
                    QColor("#27ae60") if not self.is_modified() else QColor("#1e8449")
                )
                label = self.orientation.upper()
            else:
                color = QColor("#c0392b")
                label = "?"
            self._letter.setVisible(True)
            self._letter.setPlainText(label)
            self._letter.setDefaultTextColor(color)
            br = self._letter.boundingRect()
            self._letter.setPos(
                r.x() + (r.width() - br.width()) / 2,
                r.y() + (r.height() - br.height()) / 2,
            )
        else:
            color = QColor("#7f8c8d")
            self._letter.setVisible(False)

        outline = 4 if self.isSelected() else 2
        pen = QPen(color, outline)
        pen.setCosmetic(True)
        self.setPen(pen)
        fill = QColor(color)
        fill.setAlpha(48 if self.required else 24)
        self.setBrush(QBrush(fill))

    def itemChange(self, change, value):  # type: ignore[override]
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self._apply_visuals()
        return super().itemChange(change, value)


class OrientationCanvas(QGraphicsView):
    

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._components: list[OrientationBboxItem] = []
        self.on_dirty: Callable[[], None] = lambda: None
        self.on_selection_changed: Callable[[OrientationBboxItem | None], None] = (
            lambda _: None
        )
        self._scene.selectionChanged.connect(self._handle_selection_change)
        self.setDragMode(QGraphicsView.RubberBandDrag)

    def reset(self, image_path: Path | None) -> None:
        self._scene.clear()
        self._components.clear()
        self._pixmap_item = None
        if image_path is not None and image_path.exists():
            pm = QPixmap(str(image_path))
            if not pm.isNull():
                self._pixmap_item = self._scene.addPixmap(pm)
                self._pixmap_item.setZValue(-1)
                self._scene.setSceneRect(0, 0, pm.width(), pm.height())
                self.resetTransform()
                self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def add_component(self, component: dict, required: bool) -> OrientationBboxItem:
        item = OrientationBboxItem(component, required=required)
        self._scene.addItem(item)
        self._components.append(item)
        return item

    def all_components(self) -> list[OrientationBboxItem]:
        return list(self._components)

    def selected_component(self) -> OrientationBboxItem | None:
        for it in self._scene.selectedItems():
            if isinstance(it, OrientationBboxItem):
                return it
        return None

    def set_orientation_of_selected(self, orientation: str | None) -> bool:
        sel = self.selected_component()
        if sel is None or not sel.required:
            return False
        if sel.orientation == orientation:
            return False
        sel.set_orientation(orientation)
        self.on_dirty()
        return True

    def wheelEvent(self, event):  # type: ignore[override]
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            factor = 1.15 if delta > 0 else 1 / 1.15
            self.scale(factor, factor)
            event.accept()
            return
        super().wheelEvent(event)

    def _handle_selection_change(self) -> None:
        self.on_selection_changed(self.selected_component())


class OrientationEditor(QWidget):

    def __init__(
        self, result_dir: Path, on_commit: Callable[[str, str], None], parent=None
    ):
        super().__init__(parent)
        self.result_dir = result_dir
        self.on_commit = on_commit
        self._stem: str | None = None
        self._data: dict | None = None
        self._class_to_idx: dict[str, int] = {}
        self._required_classes: set[str] = set()
        self._dirty = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        tb = QHBoxLayout()
        tb.addWidget(QLabel("Orient:"))
        self._dir_buttons: list[tuple[QPushButton, str]] = []
        for orientation, label in (
            ("u", "U  ↑"),
            ("r", "R  →"),
            ("d", "D  ↓"),
            ("l", "L  ←"),
        ):
            b = QPushButton(label)
            b.setEnabled(False)
            b.clicked.connect(lambda _=False, o=orientation: self._set_orientation(o))
            tb.addWidget(b)
            self._dir_buttons.append((b, orientation))
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setEnabled(False)
        self._clear_btn.clicked.connect(lambda: self._set_orientation(None))
        tb.addWidget(self._clear_btn)

        tb.addSpacing(12)
        self._selection_label = QLabel("(no selection)")
        self._selection_label.setStyleSheet("color: #555;")
        tb.addWidget(self._selection_label, 1)

        self.dirty_label = QLabel("clean")
        self.dirty_label.setStyleSheet("color: #888;")
        tb.addWidget(self.dirty_label)

        self.revert_btn = QPushButton("Revert")
        self.revert_btn.clicked.connect(self._revert)
        tb.addWidget(self.revert_btn)

        self.commit_btn = QPushButton("Commit + re-run")
        self.commit_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 4px 10px; }"
        )
        self.commit_btn.clicked.connect(self._commit)
        tb.addWidget(self.commit_btn)

        layout.addLayout(tb)

        self.canvas = OrientationCanvas()
        self.canvas.on_dirty = self._mark_dirty
        self.canvas.on_selection_changed = self._on_selection_changed
        layout.addWidget(self.canvas, 1)

        hint = QLabel(
            "Click a component bbox to select. Required components show U/R/D/L (green) or "
            "'?' (red); non-required components are gray and not editable. Ctrl+wheel to zoom."
        )
        hint.setStyleSheet("color: #888; font-size: 10px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

    # ---- public API ----

    def stem(self) -> str | None:
        return self._stem

    def is_dirty(self) -> bool:
        return self._dirty

    def load_stem(self, stem: str) -> None:
        self._stem = stem
        ori_path = self.result_dir / "orientation" / f"{stem}.json"
        if not ori_path.exists():
            QMessageBox.warning(
                self,
                "No orientation data",
                f"orientation/{stem}.json is missing. Run detect_orientation first "
                f"(re-run the pipeline at least up through that step).",
            )
            self._data = None
            self.canvas.reset(None)
            self._dirty = False
            self._update_dirty_label()
            self._refresh_buttons()
            return
        try:
            self._data = json.loads(ori_path.read_text(encoding="utf-8"))
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", f"Cannot parse {ori_path}: {exc}")
            self._data = None
            return
        self._class_to_idx = dict(self._data.get("resnet_orientation_classes") or {})
        self._required_classes = set(
            self._data.get("orientation_required_classes") or []
        )

        image_path = find_image(self.result_dir / "images", stem)
        self.canvas.reset(image_path)
        for c in self._data.get("components", []):
            required = c.get("component_class") in self._required_classes
            self.canvas.add_component(c, required=required)

        self._dirty = False
        self._update_dirty_label()
        self._refresh_buttons()

    # ---- callbacks ----

    def _set_orientation(self, orientation: str | None) -> None:
        if self.canvas.set_orientation_of_selected(orientation):
            self._refresh_buttons()

    def _on_selection_changed(self, _item) -> None:
        self._refresh_buttons()

    def _refresh_buttons(self) -> None:
        sel = self.canvas.selected_component()
        enabled = sel is not None and sel.required
        for b, _ in self._dir_buttons:
            b.setEnabled(enabled)
        self._clear_btn.setEnabled(enabled)
        if sel is None:
            self._selection_label.setText("(no selection)")
            return
        req = "required" if sel.required else "not required"
        cur = sel.orientation or "—"
        suffix = "  ●" if sel.is_modified() else ""
        self._selection_label.setText(
            f"{sel.component_class} #{sel.component_id}  ({req})   current: {cur}{suffix}"
        )

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._update_dirty_label()

    def _update_dirty_label(self) -> None:
        if self._dirty:
            self.dirty_label.setText("● unsaved edits")
            self.dirty_label.setStyleSheet("color: #c0392b; font-weight: bold;")
        else:
            self.dirty_label.setText("clean")
            self.dirty_label.setStyleSheet("color: #888;")

    def _revert(self) -> None:
        if self._stem is None:
            return
        if self._dirty:
            ok = QMessageBox.question(
                self,
                "Revert orientation edits",
                "Discard pending orientation edits and reload from disk?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if ok != QMessageBox.Yes:
                return
        self.load_stem(self._stem)

    def _commit(self) -> None:
        if self._stem is None or self._data is None:
            return
        if not self._dirty:
            QMessageBox.information(
                self, "Nothing to commit", "No orientation edits pending."
            )
            return
        canvas_by_id = {it.component_id: it for it in self.canvas.all_components()}
        # Diff against the in-memory data (which still reflects what's on disk
        # because _do_commit hasn't run yet).
        changes: list[tuple[int, str | None, str | None, str]] = []
        for c in self._data.get("components", []):
            cid = int(c["component_id"])
            it = canvas_by_id.get(cid)
            if it is None:
                continue
            old = c.get("orientation")
            new = it.orientation
            if new != old:
                changes.append((cid, old, new, c.get("component_class", "?")))
        if not changes:
            self._dirty = False
            self._update_dirty_label()
            QMessageBox.information(self, "Nothing to commit", "No effective changes.")
            return

        msg_lines = [
            f"Commit {len(changes)} orientation change(s) for {self._stem} "
            f"and re-run from build_incidence?\n",
        ]
        for cid, old, new, cls in changes[:10]:
            msg_lines.append(f"  • #{cid} {cls}: {old or '—'} → {new or '—'}")
        if len(changes) > 10:
            msg_lines.append(f"  … and {len(changes) - 10} more")
        msg_lines.append(
            "\nWARNING: any later-stage manual edits (touches) for this stem "
            "will be overwritten by the re-run."
        )
        ok = QMessageBox.question(
            self,
            "Commit + re-run",
            "\n".join(msg_lines),
            QMessageBox.Yes | QMessageBox.No,
        )
        if ok != QMessageBox.Yes:
            return
        try:
            self._do_commit(canvas_by_id)
        except Exception as exc:
            QMessageBox.critical(self, "Commit failed", str(exc))
            return
        self._dirty = False
        self._update_dirty_label()
        self.on_commit(self._stem, "build_incidence")

    def _do_commit(self, canvas_by_id: dict[int, OrientationBboxItem]) -> None:
        assert self._stem is not None and self._data is not None
        for c in self._data["components"]:
            cid = int(c["component_id"])
            it = canvas_by_id.get(cid)
            if it is None:
                continue
            new = it.orientation
            c["orientation"] = new
            if new is None:
                c["orientation_id"] = None
                c["orientation_confidence"] = None
                c["orientation_probabilities"] = None
            else:
                c["orientation_id"] = self._class_to_idx.get(new)
                # User-confirmed: full confidence, no probability distribution.
                c["orientation_confidence"] = 1.0
                c["orientation_probabilities"] = None
            # Status: leave "orientation_not_required" alone, else "ok".
            if c.get("status") != "orientation_not_required":
                c["status"] = "ok"
        # Refresh the prediction count.
        self._data["num_orientation_predictions"] = sum(
            1 for c in self._data["components"] if c.get("orientation") is not None
        )

        out_path = self.result_dir / "orientation" / f"{self._stem}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


class OrientationTab(QWidget):

    def __init__(
        self, result_dir: Path, on_commit: Callable[[str, str], None], parent=None
    ):
        super().__init__(parent)
        self.result_dir = result_dir
        self._stem: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        tb = QHBoxLayout()
        self.edit_toggle = QCheckBox("Edit orientations")
        self.edit_toggle.toggled.connect(self._on_mode_changed)
        tb.addWidget(self.edit_toggle)
        tb.addStretch(1)
        layout.addLayout(tb)

        self.stack = QStackedWidget()
        self.view = ImageView()
        self.editor = OrientationEditor(result_dir, on_commit)
        self.stack.addWidget(self.view)
        self.stack.addWidget(self.editor)
        self.stack.setCurrentWidget(self.view)
        layout.addWidget(self.stack, 1)

    def show_stem(self, stem: str | None) -> None:
        if stem is None:
            self._stem = None
            self.view.set_image(None)
            return
        if (
            self.edit_toggle.isChecked()
            and self.editor.is_dirty()
            and self.editor.stem() is not None
            and self.editor.stem() != stem
        ):
            QMessageBox.warning(
                self,
                "Edits discarded",
                f"Switching away from {self.editor.stem()} with unsaved orientation "
                f"edits — they have been discarded.",
            )
        self._stem = stem
        path = find_vis_file(
            self.result_dir, "orientation/visualizations", stem, "{stem}_annotated.*"
        )
        self.view.set_image(path)
        if self.edit_toggle.isChecked():
            self.editor.load_stem(stem)

    def _on_mode_changed(self, checked: bool) -> None:
        if checked:
            self.stack.setCurrentWidget(self.editor)
            if self._stem is not None:
                self.editor.load_stem(self._stem)
        else:
            self.stack.setCurrentWidget(self.view)


# ---------- nodes editor ----------


class LineItem(QGraphicsLineItem):
    

    def __init__(
        self, x1: float, y1: float, x2: float, y2: float, node_id: int, line_idx: int
    ):
        super().__init__(QLineF(x1, y1, x2, y2))
        self.node_id = int(node_id)
        self.line_idx = int(line_idx)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self._apply_pen()

    def set_node_id(self, node_id: int) -> None:
        self.node_id = int(node_id)
        self._apply_pen()

    def set_interactive(self, interactive: bool) -> None:
        self.setFlag(QGraphicsItem.ItemIsSelectable, interactive)
        # When non-interactive, the line is still visible but faint.
        self._apply_pen(faint=not interactive)

    def _apply_pen(self, faint: bool = False) -> None:
        color = node_color(self.node_id)
        if faint:
            color = QColor(color)
            color.setAlpha(70)
            width = 1.5
        else:
            width = 4.0 if self.isSelected() else 2.5
        pen = QPen(color, width)
        pen.setCosmetic(True)
        pen.setCapStyle(Qt.RoundCap)
        self.setPen(pen)

    def itemChange(self, change, value):  # type: ignore[override]
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self._apply_pen(faint=not (self.flags() & QGraphicsItem.ItemIsSelectable))
        return super().itemChange(change, value)


class JumpItem(QGraphicsRectItem):

    BASE_COLOR = QColor("#f1c40f")  # yellow

    def __init__(
        self, x1: float, y1: float, x2: float, y2: float, idx: int, source: str = "yolo"
    ):
        super().__init__(QRectF(x1, y1, x2 - x1, y2 - y1))
        self.idx = int(idx)
        self.source = source  # 'yolo' | 'manual'
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self._apply_pen()

    def set_interactive(self, interactive: bool) -> None:
        self.setFlag(QGraphicsItem.ItemIsSelectable, interactive)
        self._apply_pen(faint=not interactive)

    def _apply_pen(self, faint: bool = False) -> None:
        color = QColor(self.BASE_COLOR)
        if faint:
            color.setAlpha(80)
            width = 1.5
        else:
            width = 4 if self.isSelected() else 2
        pen = QPen(color, width)
        pen.setCosmetic(True)
        if self.source == "manual":
            pen.setStyle(Qt.DashLine)
        self.setPen(pen)
        fill = QColor(color)
        fill.setAlpha(40 if not faint else 20)
        self.setBrush(QBrush(fill))

    def itemChange(self, change, value):  # type: ignore[override]
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self._apply_pen(faint=not (self.flags() & QGraphicsItem.ItemIsSelectable))
        return super().itemChange(change, value)


class JunctionItem(QGraphicsRectItem):
    

    BASE_COLOR = QColor("#27ae60")

    def __init__(
        self, x1: float, y1: float, x2: float, y2: float, idx: int, source: str = "yolo"
    ):
        super().__init__(QRectF(x1, y1, x2 - x1, y2 - y1))
        self.idx = int(idx)
        self.source = source
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self._apply_pen()

    def set_interactive(self, interactive: bool) -> None:
        self.setFlag(QGraphicsItem.ItemIsSelectable, interactive)
        self._apply_pen(faint=not interactive)

    def _apply_pen(self, faint: bool = False) -> None:
        color = QColor(self.BASE_COLOR)
        if faint:
            color.setAlpha(80)
            width = 1.5
        else:
            width = 4 if self.isSelected() else 2
        pen = QPen(color, width)
        pen.setCosmetic(True)
        if self.source == "manual":
            pen.setStyle(Qt.DashLine)
        self.setPen(pen)
        fill = QColor(color)
        fill.setAlpha(40 if not faint else 20)
        self.setBrush(QBrush(fill))

    def itemChange(self, change, value):  # type: ignore[override]
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self._apply_pen(faint=not (self.flags() & QGraphicsItem.ItemIsSelectable))
        return super().itemChange(change, value)


class XJumpItem(QGraphicsRectItem):

    BASE_COLOR = QColor("#8e44ad")  # purple

    def __init__(self, x1: float, y1: float, x2: float, y2: float, idx: int):
        super().__init__(QRectF(x1, y1, x2 - x1, y2 - y1))
        self.idx = int(idx)
        self.source = "manual"  # X-jumps are always manual
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        # Centered 'X' label.
        self._label = QGraphicsTextItem(parent=self)
        font = QFont("Arial", 18, QFont.Bold)
        self._label.setFont(font)
        self._label.setPlainText("X")
        self._label.setDefaultTextColor(self.BASE_COLOR)
        self._apply_pen()

    def set_interactive(self, interactive: bool) -> None:
        self.setFlag(QGraphicsItem.ItemIsSelectable, interactive)
        self._apply_pen(faint=not interactive)

    def _apply_pen(self, faint: bool = False) -> None:
        color = QColor(self.BASE_COLOR)
        if faint:
            color.setAlpha(80)
            width = 1.5
        else:
            width = 4 if self.isSelected() else 2
        pen = QPen(color, width)
        pen.setCosmetic(True)
        pen.setStyle(Qt.DashLine)
        self.setPen(pen)
        fill = QColor(color)
        fill.setAlpha(40 if not faint else 18)
        self.setBrush(QBrush(fill))
        # Re-center the X label inside the (possibly resized) rect.
        r = self.rect()
        br = self._label.boundingRect()
        self._label.setPos(
            r.x() + (r.width() - br.width()) / 2,
            r.y() + (r.height() - br.height()) / 2,
        )
        self._label.setDefaultTextColor(color)
        self._label.setVisible(not faint or True)  # always show

    def itemChange(self, change, value):  # type: ignore[override]
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self._apply_pen(faint=not (self.flags() & QGraphicsItem.ItemIsSelectable))
        return super().itemChange(change, value)


class NodesCanvas(QGraphicsView):

    MODE_JJ = "jumps_junctions"
    MODE_LN = "lines_nodes"

    # tool ids
    TOOL_SELECT = 0
    TOOL_DRAW_JUMP = 1
    TOOL_DRAW_JUNCTION = 2
    TOOL_DRAW_LINE = 3
    TOOL_DRAW_X_JUMP = 4

    MIN_NEW_SIZE = 3.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._line_items: list[LineItem] = []
        self._jump_items: list[JumpItem] = []
        self._junction_items: list[JunctionItem] = []
        self._x_jump_items: list[XJumpItem] = []

        self._mode = self.MODE_JJ
        self._tool = self.TOOL_SELECT

        # In-progress draw state.
        self._drawing = False
        self._draw_start: tuple[float, float] | None = None
        self._draw_preview: QGraphicsRectItem | QGraphicsLineItem | None = None

        # callbacks
        self.on_dirty: Callable[[], None] = lambda: None
        self.on_selection_changed: Callable[[], None] = lambda: None
        self.on_line_drawn: Callable[[float, float, float, float], None] = (
            lambda *_: None
        )
        self.on_bbox_drawn: Callable[[str, float, float, float, float], None] = (
            lambda *_: None
        )

        self._scene.selectionChanged.connect(self._handle_selection_change)
        self.setDragMode(QGraphicsView.RubberBandDrag)

    # ---- scene/data management ----

    def reset(self, image_path: Path | None) -> None:
        self._drawing = False
        self._draw_preview = None
        self._draw_start = None
        self._scene.clear()
        self._line_items.clear()
        self._jump_items.clear()
        self._junction_items.clear()
        self._x_jump_items.clear()
        self._pixmap_item = None
        if image_path is not None and image_path.exists():
            pm = QPixmap(str(image_path))
            if not pm.isNull():
                self._pixmap_item = self._scene.addPixmap(pm)
                self._pixmap_item.setZValue(-1)
                self._scene.setSceneRect(0, 0, pm.width(), pm.height())
                self.resetTransform()
                self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def add_line(
        self, x1: float, y1: float, x2: float, y2: float, node_id: int, line_idx: int
    ) -> LineItem:
        item = LineItem(x1, y1, x2, y2, node_id, line_idx)
        item.setZValue(1.0)
        self._scene.addItem(item)
        self._line_items.append(item)
        self._refresh_interactivity_for(item)
        return item

    def add_jump(
        self, x1: float, y1: float, x2: float, y2: float, idx: int, source: str = "yolo"
    ) -> JumpItem:
        item = JumpItem(x1, y1, x2, y2, idx, source=source)
        item.setZValue(2.0)
        self._scene.addItem(item)
        self._jump_items.append(item)
        self._refresh_interactivity_for(item)
        return item

    def add_junction(
        self, x1: float, y1: float, x2: float, y2: float, idx: int, source: str = "yolo"
    ) -> JunctionItem:
        item = JunctionItem(x1, y1, x2, y2, idx, source=source)
        item.setZValue(2.0)
        self._scene.addItem(item)
        self._junction_items.append(item)
        self._refresh_interactivity_for(item)
        return item

    def add_x_jump(
        self, x1: float, y1: float, x2: float, y2: float, idx: int
    ) -> XJumpItem:
        item = XJumpItem(x1, y1, x2, y2, idx)
        item.setZValue(2.0)
        self._scene.addItem(item)
        self._x_jump_items.append(item)
        self._refresh_interactivity_for(item)
        return item

    def remove_item(self, item) -> None:
        if isinstance(item, LineItem) and item in self._line_items:
            self._line_items.remove(item)
        elif isinstance(item, JumpItem) and item in self._jump_items:
            self._jump_items.remove(item)
        elif isinstance(item, JunctionItem) and item in self._junction_items:
            self._junction_items.remove(item)
        elif isinstance(item, XJumpItem) and item in self._x_jump_items:
            self._x_jump_items.remove(item)
        self._scene.removeItem(item)

    def all_lines(self) -> list[LineItem]:
        return list(self._line_items)

    def all_jumps(self) -> list[JumpItem]:
        return list(self._jump_items)

    def all_junctions(self) -> list[JunctionItem]:
        return list(self._junction_items)

    def all_x_jumps(self) -> list[XJumpItem]:
        return list(self._x_jump_items)

    def selected_lines(self) -> list[LineItem]:
        return [it for it in self._scene.selectedItems() if isinstance(it, LineItem)]

    def selected_jumps(self) -> list[JumpItem]:
        return [it for it in self._scene.selectedItems() if isinstance(it, JumpItem)]

    def selected_junctions(self) -> list[JunctionItem]:
        return [
            it for it in self._scene.selectedItems() if isinstance(it, JunctionItem)
        ]

    def selected_x_jumps(self) -> list[XJumpItem]:
        return [it for it in self._scene.selectedItems() if isinstance(it, XJumpItem)]

    # ---- mode / tool ----

    def set_mode(self, mode: str) -> None:
        if mode not in (self.MODE_JJ, self.MODE_LN):
            return
        self._mode = mode
        self._scene.clearSelection()
        for it in self._line_items:
            self._refresh_interactivity_for(it)
        for it in self._jump_items:
            self._refresh_interactivity_for(it)
        for it in self._junction_items:
            self._refresh_interactivity_for(it)
        for it in self._x_jump_items:
            self._refresh_interactivity_for(it)
        self.set_tool(self.TOOL_SELECT)

    def _refresh_interactivity_for(self, item) -> None:
        if self._mode == self.MODE_JJ:
            if isinstance(item, LineItem):
                item.set_interactive(False)
            else:
                item.set_interactive(True)
        else:  # MODE_LN
            if isinstance(item, LineItem):
                item.set_interactive(True)
            else:
                item.set_interactive(False)

    def set_tool(self, tool: int) -> None:
        self._tool = tool
        if tool == self.TOOL_SELECT:
            self.setDragMode(QGraphicsView.RubberBandDrag)
            self.viewport().setCursor(Qt.ArrowCursor)
        else:
            self.setDragMode(QGraphicsView.NoDrag)
            self.viewport().setCursor(Qt.CrossCursor)

    # ---- mouse / key ----

    def mousePressEvent(self, event):  # type: ignore[override]
        if event.button() == Qt.LeftButton and self._tool != self.TOOL_SELECT:
            sp = self.mapToScene(event.position().toPoint())
            self._drawing = True
            self._draw_start = (sp.x(), sp.y())
            if self._tool == self.TOOL_DRAW_LINE:
                preview = QGraphicsLineItem(QLineF(sp.x(), sp.y(), sp.x(), sp.y()))
                pen = QPen(QColor("#c0392b"), 3, Qt.DashLine)
                pen.setCosmetic(True)
                preview.setPen(pen)
            else:
                preview = QGraphicsRectItem(QRectF(sp.x(), sp.y(), 0, 0))
                pen = QPen(QColor("#c0392b"), 2, Qt.DashLine)
                pen.setCosmetic(True)
                preview.setPen(pen)
            preview.setZValue(10)
            self._scene.addItem(preview)
            self._draw_preview = preview
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # type: ignore[override]
        if (
            self._drawing
            and self._draw_preview is not None
            and self._draw_start is not None
        ):
            sp = self.mapToScene(event.position().toPoint())
            x0, y0 = self._draw_start
            if self._tool == self.TOOL_DRAW_LINE:
                self._draw_preview.setLine(QLineF(x0, y0, sp.x(), sp.y()))
            else:
                x = min(x0, sp.x())
                y = min(y0, sp.y())
                w = abs(sp.x() - x0)
                h = abs(sp.y() - y0)
                self._draw_preview.setRect(QRectF(x, y, w, h))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # type: ignore[override]
        if self._drawing and event.button() == Qt.LeftButton:
            self._drawing = False
            preview = self._draw_preview
            tool = self._tool
            self._draw_preview = None
            if preview is not None:
                if tool == self.TOOL_DRAW_LINE:
                    line = preview.line()
                    self._scene.removeItem(preview)
                    if (abs(line.dx()) + abs(line.dy())) >= self.MIN_NEW_SIZE:
                        self.on_line_drawn(line.x1(), line.y1(), line.x2(), line.y2())
                else:
                    r = preview.rect()
                    self._scene.removeItem(preview)
                    if (
                        r.width() >= self.MIN_NEW_SIZE
                        and r.height() >= self.MIN_NEW_SIZE
                    ):
                        if tool == self.TOOL_DRAW_JUMP:
                            kind = "jump"
                        elif tool == self.TOOL_DRAW_JUNCTION:
                            kind = "junction"
                        else:
                            kind = "x_jump"
                        self.on_bbox_drawn(
                            kind, r.x(), r.y(), r.x() + r.width(), r.y() + r.height()
                        )
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):  # type: ignore[override]
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            # The editor handles deletes — it needs to update the state object
            # alongside the canvas item.
            self.on_selection_changed()
            event.accept()
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event):  # type: ignore[override]
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            factor = 1.15 if delta > 0 else 1 / 1.15
            self.scale(factor, factor)
            event.accept()
            return
        super().wheelEvent(event)

    def _handle_selection_change(self) -> None:
        self.on_selection_changed()


class NodesEditor(QWidget):

    def __init__(
        self, result_dir: Path, on_commit: Callable[[str, str], None], parent=None
    ):
        super().__init__(parent)
        self.result_dir = result_dir
        self.on_commit = on_commit
        self._stem: str | None = None
        self._state: dict | None = None
        # Dirty flags, per-mode.
        self._dirty_jj = False
        self._dirty_ln = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        # Top toolbar: mode selector + dirty + revert + commit
        tb1 = QHBoxLayout()
        tb1.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem(
            "1 · Jumps / Junctions  →  re-run generate_nodes", NodesCanvas.MODE_JJ
        )
        self.mode_combo.addItem(
            "2 · Lines / Nodes  →  re-run export_touches", NodesCanvas.MODE_LN
        )
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        tb1.addWidget(self.mode_combo)

        tb1.addStretch(1)

        self.dirty_label = QLabel("clean")
        self.dirty_label.setStyleSheet("color: #888;")
        tb1.addWidget(self.dirty_label)

        self.revert_btn = QPushButton("Revert")
        self.revert_btn.clicked.connect(self._revert)
        tb1.addWidget(self.revert_btn)

        self.commit_btn = QPushButton("Commit + re-run")
        self.commit_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 4px 10px; }"
        )
        self.commit_btn.clicked.connect(self._commit)
        tb1.addWidget(self.commit_btn)

        layout.addLayout(tb1)

        # Per-mode sub-toolbar (stacked widget).
        self.tools_stack = QStackedWidget()

        # --- Jumps/Junctions toolbar ---
        jj_box = QWidget()
        jj_layout = QHBoxLayout(jj_box)
        jj_layout.setContentsMargins(0, 0, 0, 0)
        self.jj_select = QPushButton("Select")
        self.jj_select.setCheckable(True)
        self.jj_select.setChecked(True)
        self.jj_draw_jump = QPushButton("Draw Jump")
        self.jj_draw_jump.setCheckable(True)
        self.jj_draw_junction = QPushButton("Draw Junction")
        self.jj_draw_junction.setCheckable(True)
        self.jj_draw_x_jump = QPushButton("Draw X-Jump")
        self.jj_draw_x_jump.setCheckable(True)
        self.jj_draw_x_jump.setToolTip(
            "Diagonal crossover: lines running top-left↔bottom-right form one "
            "node, top-right↔bottom-left another. Manual-only — no YOLO."
        )
        jj_group = QButtonGroup(self)
        jj_group.setExclusive(True)
        for b in (
            self.jj_select,
            self.jj_draw_jump,
            self.jj_draw_junction,
            self.jj_draw_x_jump,
        ):
            jj_group.addButton(b)
            jj_layout.addWidget(b)
        self.jj_select.toggled.connect(self._on_jj_tool_changed)
        self.jj_draw_jump.toggled.connect(self._on_jj_tool_changed)
        self.jj_draw_junction.toggled.connect(self._on_jj_tool_changed)
        self.jj_draw_x_jump.toggled.connect(self._on_jj_tool_changed)
        self.jj_delete_btn = QPushButton("Delete selected")
        self.jj_delete_btn.clicked.connect(self._delete_selected_jj)
        jj_layout.addWidget(self.jj_delete_btn)
        jj_layout.addStretch(1)
        self.jj_info = QLabel("(no selection)")
        self.jj_info.setStyleSheet("color: #555;")
        jj_layout.addWidget(self.jj_info)
        self.tools_stack.addWidget(jj_box)

        # --- Lines/Nodes toolbar ---
        ln_box = QWidget()
        ln_layout = QHBoxLayout(ln_box)
        ln_layout.setContentsMargins(0, 0, 0, 0)
        self.ln_select = QPushButton("Select")
        self.ln_select.setCheckable(True)
        self.ln_select.setChecked(True)
        self.ln_draw = QPushButton("Draw Line")
        self.ln_draw.setCheckable(True)
        ln_group = QButtonGroup(self)
        ln_group.setExclusive(True)
        for b in (self.ln_select, self.ln_draw):
            ln_group.addButton(b)
            ln_layout.addWidget(b)
        self.ln_select.toggled.connect(self._on_ln_tool_changed)
        self.ln_draw.toggled.connect(self._on_ln_tool_changed)

        ln_layout.addSpacing(12)
        ln_layout.addWidget(QLabel("Target node:"))
        self.ln_node_combo = QComboBox()
        ln_layout.addWidget(self.ln_node_combo)

        self.ln_reassign_btn = QPushButton("Move selected → target")
        self.ln_reassign_btn.clicked.connect(self._reassign_selected_lines)
        ln_layout.addWidget(self.ln_reassign_btn)

        self.ln_delete_btn = QPushButton("Delete selected line(s)")
        self.ln_delete_btn.clicked.connect(self._delete_selected_lines)
        ln_layout.addWidget(self.ln_delete_btn)

        ln_layout.addStretch(1)
        self.ln_info = QLabel("(no selection)")
        self.ln_info.setStyleSheet("color: #555;")
        ln_layout.addWidget(self.ln_info)
        self.tools_stack.addWidget(ln_box)

        layout.addWidget(self.tools_stack)

        # Canvas.
        self.canvas = NodesCanvas()
        self.canvas.on_dirty = self._mark_dirty_current_mode
        self.canvas.on_selection_changed = self._on_selection_changed
        self.canvas.on_line_drawn = self._on_line_drawn
        self.canvas.on_bbox_drawn = self._on_bbox_drawn
        layout.addWidget(self.canvas, 1)

        hint = QLabel(
            "Mode 1 edits jump/junction bboxes (commit forces a re-run from "
            "generate_nodes, which rebuilds everything downstream). "
            "Mode 2 edits lines + node assignments (commit re-runs from "
            "export_touches). Ctrl+wheel to zoom."
        )
        hint.setStyleSheet("color: #888; font-size: 10px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._apply_mode()  # syncs UI with default mode

    # ---- public API ----

    def stem(self) -> str | None:
        return self._stem

    def is_dirty(self) -> bool:
        return self._dirty_jj or self._dirty_ln

    def load_stem(self, stem: str) -> None:
        self._stem = stem
        npz = self.result_dir / "nodes" / "data" / f"{stem}.npz"
        jsn = self.result_dir / "nodes" / "data" / f"{stem}.json"
        if not npz.exists() or not jsn.exists():
            QMessageBox.warning(
                self,
                "No node data",
                f"nodes/data/{stem}.{{npz,json}} is missing. Run generate_nodes first.",
            )
            self._state = None
            self.canvas.reset(None)
            self._dirty_jj = False
            self._dirty_ln = False
            self._update_dirty_label()
            self._refresh_info()
            return
        try:
            self._state = read_nodes_data(npz, jsn)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", f"Cannot parse {npz}: {exc}")
            self._state = None
            return

        image_path = find_image(self.result_dir / "images", stem)
        self.canvas.reset(image_path)
        self._populate_canvas_from_state()
        self._populate_node_combo()
        self._dirty_jj = False
        self._dirty_ln = False
        self._update_dirty_label()
        self._refresh_info()

    # ---- canvas population ----

    def _populate_canvas_from_state(self) -> None:
        assert self._state is not None and self._stem is not None
        lines = np.asarray(self._state["lines"]).reshape(-1, 2, 2)
        node_ids = np.asarray(self._state["line_node_ids"])
        for i in range(len(lines)):
            (x1, y1), (x2, y2) = lines[i]
            self.canvas.add_line(
                float(x1), float(y1), float(x2), float(y2), int(node_ids[i]), line_idx=i
            )
        jb = np.asarray(self._state["jump_bboxes"]).reshape(-1, 4)
        jump_sources = jj_source_lookup(self._stem, jb, "jump")
        for i in range(len(jb)):
            x1, y1, x2, y2 = jb[i]
            self.canvas.add_jump(
                float(x1),
                float(y1),
                float(x2),
                float(y2),
                idx=i,
                source=jump_sources[i],
            )
        jcb = np.asarray(self._state["junction_bboxes"]).reshape(-1, 4)
        jct_sources = jj_source_lookup(self._stem, jcb, "junction")
        for i in range(len(jcb)):
            x1, y1, x2, y2 = jcb[i]
            self.canvas.add_junction(
                float(x1), float(y1), float(x2), float(y2), idx=i, source=jct_sources[i]
            )
        # X-jumps come only from the override file (no YOLO, no npz storage).
        overrides = read_manual_jj_overrides(self._stem)
        for i, row in enumerate(overrides.get("added_x_jumps", [])):
            if len(row) < 4:
                continue
            x1, y1, x2, y2 = row[:4]
            self.canvas.add_x_jump(float(x1), float(y1), float(x2), float(y2), idx=i)

    def _populate_node_combo(self) -> None:
        self.ln_node_combo.blockSignals(True)
        self.ln_node_combo.clear()
        if self._state is None:
            self.ln_node_combo.blockSignals(False)
            return
        ids = sorted({int(x) for x in self._state["line_node_ids"] if int(x) >= 0})
        for nid in ids:
            self.ln_node_combo.addItem(f"node {nid}", nid)
        new_id = next_node_id(self._state)
        self.ln_node_combo.addItem(f"+ new node ({new_id})", new_id)
        self.ln_node_combo.blockSignals(False)

    # ---- mode/tool changes ----

    def _apply_mode(self) -> None:
        mode = self.mode_combo.currentData()
        self.canvas.set_mode(mode)
        idx = 0 if mode == NodesCanvas.MODE_JJ else 1
        self.tools_stack.setCurrentIndex(idx)
        # reset both sub-toolbars to Select
        if mode == NodesCanvas.MODE_JJ:
            self.jj_select.setChecked(True)
        else:
            self.ln_select.setChecked(True)
        self._update_dirty_label()
        self._refresh_info()

    def _on_mode_changed(self, _idx: int) -> None:
        if self._dirty_jj or self._dirty_ln:
            # Warn user about pending edits when switching mode.
            dirty_mode = "jumps/junctions" if self._dirty_jj else "lines/nodes"
            ok = QMessageBox.question(
                self,
                "Pending edits",
                f"You have unsaved {dirty_mode} edits. Switching mode keeps them "
                f"in memory but you must commit each mode separately. Continue?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if ok != QMessageBox.Yes:
                # revert
                self.mode_combo.blockSignals(True)
                other = 1 if self.mode_combo.currentIndex() == 0 else 0
                self.mode_combo.setCurrentIndex(other)
                self.mode_combo.blockSignals(False)
        self._apply_mode()

    def _on_jj_tool_changed(self) -> None:
        if self.jj_draw_jump.isChecked():
            self.canvas.set_tool(NodesCanvas.TOOL_DRAW_JUMP)
        elif self.jj_draw_junction.isChecked():
            self.canvas.set_tool(NodesCanvas.TOOL_DRAW_JUNCTION)
        elif self.jj_draw_x_jump.isChecked():
            self.canvas.set_tool(NodesCanvas.TOOL_DRAW_X_JUMP)
        else:
            self.canvas.set_tool(NodesCanvas.TOOL_SELECT)

    def _on_ln_tool_changed(self) -> None:
        if self.ln_draw.isChecked():
            self.canvas.set_tool(NodesCanvas.TOOL_DRAW_LINE)
        else:
            self.canvas.set_tool(NodesCanvas.TOOL_SELECT)

    # ---- drawn-shape callbacks from canvas ----

    def _on_line_drawn(self, x1: float, y1: float, x2: float, y2: float) -> None:
        if self._state is None:
            return
        node_id = self.ln_node_combo.currentData()
        if node_id is None:
            QMessageBox.warning(self, "Pick a node", "Set a target node first.")
            return
        node_id = int(node_id)
        msg = (
            f"Add a new line from ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f}) "
            f"and assign it to node {node_id}?"
        )
        ok = QMessageBox.question(
            self,
            "Confirm new line",
            msg,
            QMessageBox.Yes | QMessageBox.Cancel,
        )
        if ok != QMessageBox.Yes:
            return
        new_idx = add_line_to_state(self._state, x1, y1, x2, y2, node_id)
        self.canvas.add_line(x1, y1, x2, y2, node_id, line_idx=new_idx)
        self._dirty_ln = True
        self._update_dirty_label()
        self._populate_node_combo()  # may need to add the new node id
        # Keep the same target so successive draws all go to the same node.
        idx = self.ln_node_combo.findData(node_id)
        if idx >= 0:
            self.ln_node_combo.setCurrentIndex(idx)

    def _on_bbox_drawn(
        self, kind: str, x1: float, y1: float, x2: float, y2: float
    ) -> None:
        if self._state is None:
            return
        if kind == "x_jump":
            # X-jumps live only in the override file, not in the npz. Track
            # them on the canvas; on commit they get written to
            # manual_jj_overrides/{stem}.json.added_x_jumps.
            new_idx = len(self.canvas.all_x_jumps())
            self.canvas.add_x_jump(x1, y1, x2, y2, idx=new_idx)
            self._dirty_jj = True
            self._update_dirty_label()
            return
        if kind not in ("jump", "junction"):
            return
        key = "jump_bboxes" if kind == "jump" else "junction_bboxes"
        arr = np.asarray(self._state[key], dtype=np.float32).reshape(-1, 4)
        new_row = np.array([[x1, y1, x2, y2]], dtype=np.float32)
        self._state[key] = np.concatenate([arr, new_row], axis=0)
        new_idx = int(self._state[key].shape[0] - 1)
        # New shapes are always user-authored, so tagged 'manual' — these are
        # the entries that get written to manual_jj_overrides/{stem}.json on
        # commit and survive the next generate_nodes re-run.
        if kind == "jump":
            self.canvas.add_jump(x1, y1, x2, y2, idx=new_idx, source="manual")
        else:
            self.canvas.add_junction(x1, y1, x2, y2, idx=new_idx, source="manual")
        self._dirty_jj = True
        self._update_dirty_label()

    # ---- selection/delete/reassign ----

    def _on_selection_changed(self) -> None:
        self._refresh_info()

    def _refresh_info(self) -> None:
        mode = self.mode_combo.currentData()
        if mode == NodesCanvas.MODE_JJ:
            n_jump = len(self.canvas.selected_jumps())
            n_jct = len(self.canvas.selected_junctions())
            n_x = len(self.canvas.selected_x_jumps())
            if n_jump + n_jct + n_x == 0:
                self.jj_info.setText("(no selection)")
            else:
                self.jj_info.setText(
                    f"selected: {n_jump} jump(s), {n_jct} junction(s), {n_x} X-jump(s)"
                )
        else:
            sel = self.canvas.selected_lines()
            if not sel:
                self.ln_info.setText("(no selection)")
            else:
                if len(sel) == 1:
                    s = sel[0]
                    self.ln_info.setText(f"line #{s.line_idx}  node={s.node_id}")
                else:
                    nodes = {it.node_id for it in sel}
                    self.ln_info.setText(
                        f"{len(sel)} lines selected, nodes={sorted(nodes)}"
                    )

    def _delete_selected_jj(self) -> None:
        if self._state is None:
            return
        jumps = self.canvas.selected_jumps()
        jcns = self.canvas.selected_junctions()
        x_jumps = self.canvas.selected_x_jumps()
        if not jumps and not jcns and not x_jumps:
            return
        # Distinguish YOLO-sourced deletions: they don't survive re-runs
        # because the YOLO detector keeps re-finding them. Warn the user.
        n_yolo = sum(1 for it in jumps if it.source == "yolo") + sum(
            1 for it in jcns if it.source == "yolo"
        )
        n_manual = (len(jumps) + len(jcns) + len(x_jumps)) - n_yolo
        warn = ""
        if n_yolo:
            warn = (
                f"\n\nNote: {n_yolo} of these were YOLO-detected (solid outline). "
                f"Removing them locally is fine, but the next re-run will "
                f"regenerate them via YOLO. To suppress them permanently, raise "
                f"jj_conf in the parameter panel."
            )
        ok = QMessageBox.question(
            self,
            "Delete",
            f"Delete {len(jumps)} jump + {len(jcns)} junction + {len(x_jumps)} "
            f"X-jump bbox(es)?  (manual={n_manual}, yolo={n_yolo}){warn}",
            QMessageBox.Yes | QMessageBox.No,
        )
        if ok != QMessageBox.Yes:
            return
        # delete from state by filtering by idx values
        jump_indices_to_remove = {int(it.idx) for it in jumps}
        jct_indices_to_remove = {int(it.idx) for it in jcns}
        jb = np.asarray(self._state["jump_bboxes"]).reshape(-1, 4)
        new_jb = np.array(
            [row for i, row in enumerate(jb) if i not in jump_indices_to_remove],
            dtype=np.float32,
        ).reshape(-1, 4)
        self._state["jump_bboxes"] = new_jb
        jcb = np.asarray(self._state["junction_bboxes"]).reshape(-1, 4)
        new_jcb = np.array(
            [row for i, row in enumerate(jcb) if i not in jct_indices_to_remove],
            dtype=np.float32,
        ).reshape(-1, 4)
        self._state["junction_bboxes"] = new_jcb
        # Remove from canvas. X-jumps don't live in state arrays, so just
        # drop them from the canvas; reindex the survivors.
        for it in list(jumps + jcns + x_jumps):
            self.canvas.remove_item(it)
        for new_idx, it in enumerate(self.canvas.all_jumps()):
            it.idx = new_idx
        for new_idx, it in enumerate(self.canvas.all_junctions()):
            it.idx = new_idx
        for new_idx, it in enumerate(self.canvas.all_x_jumps()):
            it.idx = new_idx
        self._dirty_jj = True
        self._update_dirty_label()
        self._refresh_info()

    def _delete_selected_lines(self) -> None:
        if self._state is None:
            return
        sel = self.canvas.selected_lines()
        if not sel:
            return
        ok = QMessageBox.question(
            self,
            "Delete lines",
            f"Delete {len(sel)} line(s)?\n\nLines are removed from the state and "
            f"any anchor records that reference them are dropped.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if ok != QMessageBox.Yes:
            return
        # Sort selected line_idx descending so removal doesn't shift indices we
        # haven't processed yet.
        to_remove = sorted({int(it.line_idx) for it in sel}, reverse=True)
        for li in to_remove:
            remove_line_from_state(self._state, li)
        # Rebuild canvas line items.
        for it in self.canvas.all_lines():
            self.canvas._scene.removeItem(it)
        self.canvas._line_items.clear()
        lines = np.asarray(self._state["lines"]).reshape(-1, 2, 2)
        node_ids = np.asarray(self._state["line_node_ids"])
        for i in range(len(lines)):
            (x1, y1), (x2, y2) = lines[i]
            self.canvas.add_line(
                float(x1), float(y1), float(x2), float(y2), int(node_ids[i]), line_idx=i
            )
        self._populate_node_combo()
        self._dirty_ln = True
        self._update_dirty_label()
        self._refresh_info()

    def _reassign_selected_lines(self) -> None:
        if self._state is None:
            return
        sel = self.canvas.selected_lines()
        if not sel:
            return
        target = self.ln_node_combo.currentData()
        if target is None:
            QMessageBox.warning(self, "Pick a node", "Set a target node first.")
            return
        target = int(target)
        any_changed = False
        for it in sel:
            if reassign_line_in_state(self._state, it.line_idx, target):
                it.set_node_id(target)
                any_changed = True
        if any_changed:
            self._populate_node_combo()
            idx = self.ln_node_combo.findData(target)
            if idx >= 0:
                self.ln_node_combo.setCurrentIndex(idx)
            self._dirty_ln = True
            self._update_dirty_label()
            self._refresh_info()

    # ---- dirty / commit ----

    def _mark_dirty_current_mode(self) -> None:
        if self.mode_combo.currentData() == NodesCanvas.MODE_JJ:
            self._dirty_jj = True
        else:
            self._dirty_ln = True
        self._update_dirty_label()

    def _update_dirty_label(self) -> None:
        parts = []
        if self._dirty_jj:
            parts.append("jump/junction")
        if self._dirty_ln:
            parts.append("lines/nodes")
        if not parts:
            self.dirty_label.setText("clean")
            self.dirty_label.setStyleSheet("color: #888;")
        else:
            self.dirty_label.setText("● unsaved: " + " + ".join(parts))
            self.dirty_label.setStyleSheet("color: #c0392b; font-weight: bold;")
        # Commit button label reflects which step would run.
        mode = self.mode_combo.currentData()
        step = "generate_nodes" if mode == NodesCanvas.MODE_JJ else "export_touches"
        self.commit_btn.setText(f"Commit + re-run {step}")

    def _revert(self) -> None:
        if self._stem is None:
            return
        if self._dirty_jj or self._dirty_ln:
            ok = QMessageBox.question(
                self,
                "Revert",
                "Discard pending edits (in both modes) and reload from disk?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if ok != QMessageBox.Yes:
                return
        self.load_stem(self._stem)

    def _commit(self) -> None:
        if self._stem is None or self._state is None:
            return
        mode = self.mode_combo.currentData()
        is_jj = mode == NodesCanvas.MODE_JJ
        if is_jj and not self._dirty_jj:
            QMessageBox.information(
                self, "Nothing to commit", "No jump/junction edits pending."
            )
            return
        if not is_jj and not self._dirty_ln:
            QMessageBox.information(
                self, "Nothing to commit", "No line/node edits pending."
            )
            return
        # Phase-2c-specific guardrail: if user is in mode-2 but mode-1 is also
        # dirty, warn — committing mode-2 first means jump/junction edits won't
        # be applied at the right step in the pipeline order.
        if not is_jj and self._dirty_jj:
            warn = QMessageBox.warning(
                self,
                "Out-of-order commit",
                "You have pending jump/junction edits as well. They will NOT "
                "take effect via this re-run (which starts from export_touches, "
                "after the step that consumes jump/junction).\n\n"
                "Recommended workflow: commit jump/junction first (re-runs "
                "generate_nodes), then redo your line/node edits.\n\n"
                "Continue anyway?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if warn != QMessageBox.Yes:
                return

        from_step = "generate_nodes" if is_jj else "export_touches"
        downstream = (
            "generate_nodes → export_touches → build_incidence → build_netlist"
            if is_jj
            else "export_touches → build_incidence → build_netlist"
        )
        if is_jj:
            # Mode 1 writes to the override file (the npz will be regenerated
            # by the pipeline, which now unions YOLO + override entries).
            n_manual_jumps = sum(
                1 for it in self.canvas.all_jumps() if it.source == "manual"
            )
            n_manual_jcts = sum(
                1 for it in self.canvas.all_junctions() if it.source == "manual"
            )
            n_x_jumps = len(self.canvas.all_x_jumps())
            msg_extra = (
                f"\n\nWill write {n_manual_jumps} manual jump(s) + "
                f"{n_manual_jcts} manual junction(s) + {n_x_jumps} X-jump(s) "
                f"to manual_jj_overrides/{self._stem}.json so they survive "
                f"the next generate_nodes re-run."
            )
        else:
            msg_extra = ""
        msg = (
            f"Commit edits for {self._stem} and re-run pipeline from {from_step}?\n\n"
            f"This will rebuild: {downstream}{msg_extra}\n\n"
            f"WARNING: any later-stage manual edits (touches, etc.) for this stem "
            f"will be overwritten."
        )
        ok = QMessageBox.question(
            self,
            "Commit + re-run",
            msg,
            QMessageBox.Yes | QMessageBox.No,
        )
        if ok != QMessageBox.Yes:
            return
        try:
            if is_jj:
                # Collect 'manual'-tagged items from the canvas and write the
                # override file. Don't touch the npz — generate_nodes will
                # rebuild it from YOLO + the overrides on re-run.
                manual_jumps = [
                    self._item_xyxy(it)
                    for it in self.canvas.all_jumps()
                    if it.source == "manual"
                ]
                manual_jcts = [
                    self._item_xyxy(it)
                    for it in self.canvas.all_junctions()
                    if it.source == "manual"
                ]
                manual_x_jumps = [
                    self._item_xyxy(it) for it in self.canvas.all_x_jumps()
                ]
                write_manual_jj_overrides(
                    self._stem, manual_jumps, manual_jcts, manual_x_jumps
                )
            else:
                npz = self.result_dir / "nodes" / "data" / f"{self._stem}.npz"
                jsn = self.result_dir / "nodes" / "data" / f"{self._stem}.json"
                write_nodes_data(npz, jsn, self._state)
                # The upcoming re-run starts at export_touches, which never
                # regenerates nodes/vis. Re-render it now so the user sees
                # their new lines reflected in the read-only Nodes view.
                regenerate_nodes_vis(self.result_dir, self._stem)
        except Exception as exc:
            QMessageBox.critical(self, "Commit failed", str(exc))
            return
        # Clear the dirty flag of the committed mode; the other mode's edits
        # will be discarded when the re-run reloads the canvas.
        if is_jj:
            self._dirty_jj = False
        else:
            self._dirty_ln = False
        self._update_dirty_label()
        self.on_commit(self._stem, from_step)

    @staticmethod
    def _item_xyxy(item) -> list[float]:
        r = item.rect()
        return [
            float(r.x()),
            float(r.y()),
            float(r.x() + r.width()),
            float(r.y() + r.height()),
        ]


class NodesTab(QWidget):

    def __init__(
        self, result_dir: Path, on_commit: Callable[[str, str], None], parent=None
    ):
        super().__init__(parent)
        self.result_dir = result_dir
        self._stem: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        tb = QHBoxLayout()
        self.edit_toggle = QCheckBox("Edit lines / nodes / jumps / junctions")
        self.edit_toggle.toggled.connect(self._on_mode_changed)
        tb.addWidget(self.edit_toggle)
        tb.addStretch(1)
        layout.addLayout(tb)

        self.stack = QStackedWidget()
        self.view = ImageView()
        self.editor = NodesEditor(result_dir, on_commit)
        self.stack.addWidget(self.view)
        self.stack.addWidget(self.editor)
        self.stack.setCurrentWidget(self.view)
        layout.addWidget(self.stack, 1)

    def show_stem(self, stem: str | None) -> None:
        if stem is None:
            self._stem = None
            self.view.set_image(None)
            return
        if (
            self.edit_toggle.isChecked()
            and self.editor.is_dirty()
            and self.editor.stem() is not None
            and self.editor.stem() != stem
        ):
            QMessageBox.warning(
                self,
                "Edits discarded",
                f"Switching away from {self.editor.stem()} with unsaved node "
                f"edits — they have been discarded.",
            )
        self._stem = stem
        path = find_vis_file(self.result_dir, "nodes/vis", stem, "{stem}.*")
        self.view.set_image(path)
        if self.edit_toggle.isChecked():
            self.editor.load_stem(stem)

    def _on_mode_changed(self, checked: bool) -> None:
        if checked:
            self.stack.setCurrentWidget(self.editor)
            if self._stem is not None:
                self.editor.load_stem(self._stem)
        else:
            self.stack.setCurrentWidget(self.view)


# ---------- touches editor ----------


class TouchItem(QGraphicsEllipseItem):

    RADIUS = 4.0  # scene-coords radius; cosmetic pen keeps it crisp at any zoom

    def __init__(
        self,
        contact_xy: list[float],
        node_id: int,
        node_pos_in_data: int,
        touch_pos_in_node: int,
        source: str = "anchor",
    ):
        r = self.RADIUS
        cx, cy = float(contact_xy[0]), float(contact_xy[1])
        super().__init__(QRectF(cx - r, cy - r, 2 * r, 2 * r))
        self.node_id = int(node_id)
        self.node_pos = int(node_pos_in_data)  # index into data["nodes"]
        self.touch_pos = int(touch_pos_in_node)  # index into nodes[i]["touches"]
        self.source = source
        self.contact_xy = [cx, cy]
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self._apply_pen()

    def _apply_pen(self) -> None:
        color = node_color(self.node_id)
        outline = QColor("#222222") if self.source == "manual" else QColor(color)
        outline.setAlpha(255)
        width = 3 if self.isSelected() else 1.5
        pen = QPen(outline, width)
        pen.setCosmetic(True)
        self.setPen(pen)
        fill = QColor(color)
        fill.setAlpha(220)
        self.setBrush(QBrush(fill))

    def itemChange(self, change, value):  # type: ignore[override]
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self._apply_pen()
        return super().itemChange(change, value)


class TouchesCanvas(QGraphicsView):

    TOOL_SELECT = 0
    TOOL_ADD = 1

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._bbox_items: list[QGraphicsRectItem] = []
        self._touch_items: list[TouchItem] = []
        self._tool = self.TOOL_SELECT
        self.bboxes: list[dict] = []
        # callbacks
        self.on_selection_changed: Callable[[], None] = lambda: None
        self.on_add_request: Callable[[int, str, list[float]], None] = lambda *_: None
        self._scene.selectionChanged.connect(lambda: self.on_selection_changed())
        self.setDragMode(QGraphicsView.RubberBandDrag)

    def reset(self, image_path: Path | None) -> None:
        self._scene.clear()
        self._bbox_items.clear()
        self._touch_items.clear()
        self._pixmap_item = None
        if image_path is not None and image_path.exists():
            pm = QPixmap(str(image_path))
            if not pm.isNull():
                self._pixmap_item = self._scene.addPixmap(pm)
                self._pixmap_item.setZValue(-1)
                self._scene.setSceneRect(0, 0, pm.width(), pm.height())
                self.resetTransform()
                self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def set_bboxes(self, bboxes: list[dict]) -> None:
        self.bboxes = list(bboxes)
        for b in self.bboxes:
            x1, y1, x2, y2 = b["xyxy"]
            r = QGraphicsRectItem(QRectF(x1, y1, x2 - x1, y2 - y1))
            color = class_color(int(b["class_id"]))
            pen = QPen(color, 1.5)
            pen.setCosmetic(True)
            r.setPen(pen)
            fill = QColor(color)
            fill.setAlpha(22)
            r.setBrush(QBrush(fill))
            r.setZValue(0)
            self._scene.addItem(r)
            self._bbox_items.append(r)

    def add_touch(
        self,
        contact_xy: list[float],
        node_id: int,
        node_pos: int,
        touch_pos: int,
        source: str,
    ) -> TouchItem:
        item = TouchItem(contact_xy, node_id, node_pos, touch_pos, source)
        item.setZValue(3)
        self._scene.addItem(item)
        self._touch_items.append(item)
        return item

    def all_touches(self) -> list[TouchItem]:
        return list(self._touch_items)

    def selected_touches(self) -> list[TouchItem]:
        return [it for it in self._scene.selectedItems() if isinstance(it, TouchItem)]

    def set_tool(self, tool: int) -> None:
        self._tool = tool
        if tool == self.TOOL_SELECT:
            self.setDragMode(QGraphicsView.RubberBandDrag)
            self.viewport().setCursor(Qt.ArrowCursor)
        else:
            self.setDragMode(QGraphicsView.NoDrag)
            self.viewport().setCursor(Qt.CrossCursor)

    def mousePressEvent(self, event):  # type: ignore[override]
        if self._tool == self.TOOL_ADD and event.button() == Qt.LeftButton:
            sp = self.mapToScene(event.position().toPoint())
            best = nearest_bbox_edge((sp.x(), sp.y()), self.bboxes)
            if best is not None:
                bbox_idx, edge_name, contact_xy, _dist = best
                self.on_add_request(bbox_idx, edge_name, contact_xy)
            event.accept()
            return
        super().mousePressEvent(event)

    def wheelEvent(self, event):  # type: ignore[override]
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            factor = 1.15 if delta > 0 else 1 / 1.15
            self.scale(factor, factor)
            event.accept()
            return
        super().wheelEvent(event)


class TouchesEditor(QWidget):

    def __init__(
        self, result_dir: Path, on_commit: Callable[[str, str], None], parent=None
    ):
        super().__init__(parent)
        self.result_dir = result_dir
        self.on_commit = on_commit
        self._stem: str | None = None
        self._data: dict | None = None
        self._bboxes: list[dict] = []
        self._dirty = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        tb = QHBoxLayout()
        self.tool_select = QPushButton("Select")
        self.tool_select.setCheckable(True)
        self.tool_select.setChecked(True)
        self.tool_add = QPushButton("Add Touch")
        self.tool_add.setCheckable(True)
        grp = QButtonGroup(self)
        grp.setExclusive(True)
        for b in (self.tool_select, self.tool_add):
            grp.addButton(b)
            tb.addWidget(b)
        self.tool_select.toggled.connect(self._on_tool_changed)
        self.tool_add.toggled.connect(self._on_tool_changed)

        tb.addSpacing(12)
        tb.addWidget(QLabel("Assign to node:"))
        self.node_combo = QComboBox()
        tb.addWidget(self.node_combo)

        self.delete_btn = QPushButton("Delete selected")
        self.delete_btn.clicked.connect(self._delete_selected)
        tb.addWidget(self.delete_btn)

        tb.addStretch(1)

        self.info_label = QLabel("(no selection)")
        self.info_label.setStyleSheet("color: #555;")
        tb.addWidget(self.info_label)

        self.dirty_label = QLabel("clean")
        self.dirty_label.setStyleSheet("color: #888;")
        tb.addWidget(self.dirty_label)

        self.revert_btn = QPushButton("Revert")
        self.revert_btn.clicked.connect(self._revert)
        tb.addWidget(self.revert_btn)

        self.commit_btn = QPushButton("Commit + re-run build_incidence")
        self.commit_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 4px 10px; }"
        )
        self.commit_btn.clicked.connect(self._commit)
        tb.addWidget(self.commit_btn)

        layout.addLayout(tb)

        self.canvas = TouchesCanvas()
        self.canvas.on_selection_changed = self._refresh_info
        self.canvas.on_add_request = self._on_add_request
        layout.addWidget(self.canvas, 1)

        hint = QLabel(
            "Click near a component bbox edge while 'Add Touch' is active — "
            "the contact point auto-snaps to the nearest edge. Select an existing "
            "touch and press Delete to remove. Ctrl+wheel to zoom."
        )
        hint.setStyleSheet("color: #888; font-size: 10px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

    def stem(self) -> str | None:
        return self._stem

    def is_dirty(self) -> bool:
        return self._dirty

    def load_stem(self, stem: str) -> None:
        self._stem = stem
        self._data = read_node_touches(self.result_dir, stem)
        if self._data is None:
            QMessageBox.warning(
                self,
                "No touches data",
                f"node_touches/{stem}.json is missing. Run export_touches first.",
            )
            self.canvas.reset(None)
            self._dirty = False
            self._update_dirty_label()
            return
        # Load bboxes (used both for drawing context and for click-snapping).
        self._bboxes = read_bbox_file(
            self.result_dir / "component_bbox" / f"{stem}.txt"
        )

        image_path = find_image(self.result_dir / "images", stem)
        self.canvas.reset(image_path)
        self.canvas.set_bboxes(self._bboxes)
        self._render_touches()
        self._populate_node_combo()
        self._dirty = False
        self._update_dirty_label()
        self._refresh_info()

    def _render_touches(self) -> None:
        assert self._data is not None
        for node_pos, node in enumerate(self._data.get("nodes") or []):
            nid = int(node.get("node_id", -1))
            for touch_pos, t in enumerate(node.get("touches") or []):
                contact = t.get("contact_xy") or [0.0, 0.0]
                src = str(t.get("source", "anchor"))
                self.canvas.add_touch(contact, nid, node_pos, touch_pos, src)

    def _populate_node_combo(self) -> None:
        self.node_combo.blockSignals(True)
        self.node_combo.clear()
        if self._data is None:
            self.node_combo.blockSignals(False)
            return
        ids = sorted(
            {
                int(n.get("node_id", -1))
                for n in (self._data.get("nodes") or [])
                if int(n.get("node_id", -1)) >= 0
            }
        )
        for nid in ids:
            self.node_combo.addItem(f"node {nid}", nid)
        new_id = next_touches_node_id(self._data)
        self.node_combo.addItem(f"+ new node ({new_id})", new_id)
        self.node_combo.blockSignals(False)

    # ---- callbacks ----

    def _on_tool_changed(self) -> None:
        if self.tool_add.isChecked():
            self.canvas.set_tool(TouchesCanvas.TOOL_ADD)
        else:
            self.canvas.set_tool(TouchesCanvas.TOOL_SELECT)

    def _on_add_request(
        self, bbox_idx: int, edge_name: str, contact_xy: list[float]
    ) -> None:
        if self._data is None:
            return
        target_node = self.node_combo.currentData()
        if target_node is None:
            QMessageBox.warning(self, "Pick a node", "Set a target node first.")
            return
        target_node = int(target_node)
        bbox = self._bboxes[bbox_idx]
        cls_name = CLASS_NAMES.get(int(bbox["class_id"]), f"class_{bbox['class_id']}")
        msg = (
            f"Add a manual touch on component #{bbox_idx} "
            f"({cls_name}) edge '{edge_name}' at "
            f"({contact_xy[0]:.0f}, {contact_xy[1]:.0f}), assigned to node {target_node}?"
        )
        ok = QMessageBox.question(
            self,
            "Confirm new touch",
            msg,
            QMessageBox.Yes | QMessageBox.Cancel,
        )
        if ok != QMessageBox.Yes:
            return
        node_entry = find_or_make_node_entry(self._data, target_node)
        touch = build_manual_touch(target_node, bbox_idx, bbox, edge_name, contact_xy)
        node_entry.setdefault("touches", []).append(touch)
        # Place the new TouchItem on canvas.
        nodes_list = self._data.get("nodes") or []
        node_pos = nodes_list.index(node_entry)
        touch_pos = len(node_entry["touches"]) - 1
        item = self.canvas.add_touch(
            touch["contact_xy"], target_node, node_pos, touch_pos, source="manual"
        )
        self.canvas._scene.clearSelection()
        item.setSelected(True)
        self._populate_node_combo()
        idx = self.node_combo.findData(target_node)
        if idx >= 0:
            self.node_combo.setCurrentIndex(idx)
        self._dirty = True
        self._update_dirty_label()
        self._refresh_info()

    def _delete_selected(self) -> None:
        if self._data is None:
            return
        sel = self.canvas.selected_touches()
        if not sel:
            return
        ok = QMessageBox.question(
            self,
            "Delete",
            f"Delete {len(sel)} touch(es)? Empty nodes will be pruned.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if ok != QMessageBox.Yes:
            return
        # Group selections by node_pos to delete safely in reverse touch_pos
        # order without shifting indices.
        by_node: dict[int, list[int]] = {}
        for it in sel:
            by_node.setdefault(it.node_pos, []).append(it.touch_pos)
        nodes_list = self._data.setdefault("nodes", [])
        for node_pos, positions in by_node.items():
            for tp in sorted(positions, reverse=True):
                try:
                    del nodes_list[node_pos]["touches"][tp]
                except (IndexError, KeyError):
                    pass
        # Prune nodes that now have 0 touches.
        self._data["nodes"] = [n for n in nodes_list if (n.get("touches") or [])]
        # Re-render touches from scratch so node_pos/touch_pos indices stay
        # consistent with the new state.
        self.canvas.reset(find_image(self.result_dir / "images", self._stem))
        self.canvas.set_bboxes(self._bboxes)
        self._render_touches()
        self._populate_node_combo()
        self._dirty = True
        self._update_dirty_label()
        self._refresh_info()

    def _refresh_info(self) -> None:
        sel = self.canvas.selected_touches()
        if not sel:
            self.info_label.setText("(no selection)")
            return
        if len(sel) == 1:
            t = sel[0]
            self.info_label.setText(
                f"touch  node={t.node_id}  source={t.source}  "
                f"contact=({t.contact_xy[0]:.0f},{t.contact_xy[1]:.0f})"
            )
        else:
            self.info_label.setText(
                f"{len(sel)} touches selected, nodes={sorted({it.node_id for it in sel})}"
            )

    def _update_dirty_label(self) -> None:
        if self._dirty:
            self.dirty_label.setText("● unsaved edits")
            self.dirty_label.setStyleSheet("color: #c0392b; font-weight: bold;")
        else:
            self.dirty_label.setText("clean")
            self.dirty_label.setStyleSheet("color: #888;")

    def _revert(self) -> None:
        if self._stem is None:
            return
        if self._dirty:
            ok = QMessageBox.question(
                self,
                "Revert",
                "Discard pending touch edits and reload from disk?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if ok != QMessageBox.Yes:
                return
        self.load_stem(self._stem)

    def _commit(self) -> None:
        if self._stem is None or self._data is None:
            return
        if not self._dirty:
            QMessageBox.information(
                self, "Nothing to commit", "No touch edits pending."
            )
            return
        n_touches = sum(
            len(n.get("touches") or []) for n in self._data.get("nodes") or []
        )
        msg = (
            f"Commit touch edits for {self._stem} and re-run from build_incidence?\n\n"
            f"Touches now: {n_touches} across {len(self._data.get('nodes') or [])} node(s).\n\n"
            f"Will rebuild: build_incidence → build_netlist."
        )
        ok = QMessageBox.question(
            self,
            "Commit + re-run",
            msg,
            QMessageBox.Yes | QMessageBox.No,
        )
        if ok != QMessageBox.Yes:
            return
        try:
            write_node_touches(self.result_dir, self._stem, self._data)
        except Exception as exc:
            QMessageBox.critical(self, "Commit failed", str(exc))
            return
        self._dirty = False
        self._update_dirty_label()
        self.on_commit(self._stem, "build_incidence")


class TouchesTab(QWidget):

    def __init__(
        self, result_dir: Path, on_commit: Callable[[str, str], None], parent=None
    ):
        super().__init__(parent)
        self.result_dir = result_dir
        self._stem: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        tb = QHBoxLayout()
        self.edit_toggle = QCheckBox("Edit touches")
        self.edit_toggle.toggled.connect(self._on_mode_changed)
        tb.addWidget(self.edit_toggle)
        tb.addStretch(1)
        layout.addLayout(tb)

        self.stack = QStackedWidget()
        self.view = ImageView()
        self.editor = TouchesEditor(result_dir, on_commit)
        self.stack.addWidget(self.view)
        self.stack.addWidget(self.editor)
        self.stack.setCurrentWidget(self.view)
        layout.addWidget(self.stack, 1)

    def show_stem(self, stem: str | None) -> None:
        if stem is None:
            self._stem = None
            self.view.set_image(None)
            return
        if (
            self.edit_toggle.isChecked()
            and self.editor.is_dirty()
            and self.editor.stem() is not None
            and self.editor.stem() != stem
        ):
            QMessageBox.warning(
                self,
                "Edits discarded",
                f"Switching away from {self.editor.stem()} with unsaved touch "
                f"edits — they have been discarded.",
            )
        self._stem = stem
        path = find_vis_file(
            self.result_dir, "node_touch_visualizations", stem, "{stem}.*"
        )
        self.view.set_image(path)
        if self.edit_toggle.isChecked():
            self.editor.load_stem(stem)

    def _on_mode_changed(self, checked: bool) -> None:
        if checked:
            self.stack.setCurrentWidget(self.editor)
            if self._stem is not None:
                self.editor.load_stem(self._stem)
        else:
            self.stack.setCurrentWidget(self.view)


class VisTabsPanel(QWidget):
    

    def __init__(
        self, result_dir: Path, on_edit_commit: Callable[[str, str], None], parent=None
    ):
        super().__init__(parent)
        self.result_dir = result_dir
        self.on_edit_commit = on_edit_commit
        self._stem: str | None = None
        self.masked_tab: MaskedTab | None = None
        self.orientation_tab: OrientationTab | None = None
        self.nodes_tab: NodesTab | None = None
        self.touches_tab: TouchesTab | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.header = QLabel("(no image selected)")
        self.header.setStyleSheet("font-weight: bold; padding: 4px;")
        layout.addWidget(self.header)

        self.flag_box = QPlainTextEdit()
        self.flag_box.setReadOnly(True)
        self.flag_box.setFixedHeight(80)
        self.flag_box.setPlaceholderText("red_flags will appear here")
        layout.addWidget(self.flag_box)

        self.tabs = QTabWidget()
        self.views: dict[str, ImageView] = {}
        for label, subdir, pattern in VIS_TABS:
            if label == "Masked":
                # bbox editor.
                self.masked_tab = MaskedTab(self.result_dir, self.on_edit_commit)
                self.tabs.addTab(self.masked_tab, label)
            elif label == "Orientation":
                # orientation editor.
                self.orientation_tab = OrientationTab(
                    self.result_dir, self.on_edit_commit
                )
                self.tabs.addTab(self.orientation_tab, label)
            elif label == "Nodes":
                # lines / nodes / jumps / junctions editor.
                self.nodes_tab = NodesTab(self.result_dir, self.on_edit_commit)
                self.tabs.addTab(self.nodes_tab, label)
            elif label == "Touches":
                # touch points editor.
                self.touches_tab = TouchesTab(self.result_dir, self.on_edit_commit)
                self.tabs.addTab(self.touches_tab, label)
            else:
                view = ImageView()
                self.tabs.addTab(view, label)
                self.views[label] = view
        # Netlist text tab.
        self.netlist_text = QPlainTextEdit()
        self.netlist_text.setReadOnly(True)
        mono = QFont("Consolas", 10)
        mono.setStyleHint(QFont.Monospace)
        self.netlist_text.setFont(mono)
        self.tabs.addTab(self.netlist_text, "Netlist")
        layout.addWidget(self.tabs, 1)

    def show_stem(
        self, stem: str | None, red_flags: list[tuple[str, list[str]]]
    ) -> None:
        self._stem = stem
        if stem is None:
            self.header.setText("(no image selected)")
            self.header.setStyleSheet("font-weight: bold; padding: 4px;")
            self.flag_box.clear()
            for view in self.views.values():
                view.set_image(None)
            if self.masked_tab is not None:
                self.masked_tab.show_stem(None)
            if self.orientation_tab is not None:
                self.orientation_tab.show_stem(None)
            if self.nodes_tab is not None:
                self.nodes_tab.show_stem(None)
            if self.touches_tab is not None:
                self.touches_tab.show_stem(None)
            self.netlist_text.clear()
            return

        if red_flags:
            self.header.setText(f"{stem}     RED FLAGS")
            self.header.setStyleSheet(
                "font-weight: bold; padding: 4px; color: #c0392b;"
            )
            lines = []
            for name, flags in red_flags:
                lines.append(f"{name}:")
                for f in flags:
                    lines.append(f"    - {f}")
            self.flag_box.setPlainText("\n".join(lines))
        else:
            self.header.setText(f"{stem}     clean (no red flags)")
            self.header.setStyleSheet(
                "font-weight: bold; padding: 4px; color: #1e8449;"
            )
            self.flag_box.setPlainText("(no red_flags)")

        for label, subdir, pattern in VIS_TABS:
            if label == "Masked":
                if self.masked_tab is not None:
                    self.masked_tab.show_stem(stem)
            elif label == "Orientation":
                if self.orientation_tab is not None:
                    self.orientation_tab.show_stem(stem)
            elif label == "Nodes":
                if self.nodes_tab is not None:
                    self.nodes_tab.show_stem(stem)
            elif label == "Touches":
                if self.touches_tab is not None:
                    self.touches_tab.show_stem(stem)
            else:
                path = find_vis_file(self.result_dir, subdir, stem, pattern)
                self.views[label].set_image(path)

        netlist_path = self.result_dir / "netlist" / f"{stem}.cir"
        if netlist_path.exists():
            try:
                self.netlist_text.setPlainText(netlist_path.read_text(encoding="utf-8"))
            except Exception as exc:
                self.netlist_text.setPlainText(f"(error reading netlist: {exc})")
        else:
            self.netlist_text.setPlainText(
                "(no netlist produced — likely red-flagged or not yet run)"
            )


class ParamPanel(QWidget):
    

    def __init__(
        self,
        config: dict,
        on_value_changed: Callable[[], None],
        on_run_from: Callable[[str | None], None],
        parent=None,
    ):
        super().__init__(parent)
        self.config = config
        self.on_value_changed = on_value_changed
        self.on_run_from = on_run_from
        self.widgets: dict[str, QWidget] = {}
        self._build_ui()
        self.load_from_config(config)

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        body = QWidget()
        scroll.setWidget(body)
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(4, 4, 4, 4)
        body_layout.setSpacing(10)

        # Bucket params by section heading.
        by_section: dict[str, list[str]] = {
            label: [] for label, _, _ in STEP_SECTION_ORDER
        }
        for name, spec in PARAM_SCHEMA.items():
            section = spec.get("group") or spec.get("step") or "Runtime"
            by_section.setdefault(section, []).append(name)

        for label, step_key, blurb in STEP_SECTION_ORDER:
            params = by_section.get(label, [])
            if not params:
                continue
            box = QGroupBox(label)
            box.setStyleSheet("QGroupBox { font-weight: bold; }")
            form = QFormLayout(box)
            form.setLabelAlignment(Qt.AlignRight)
            form.setFormAlignment(Qt.AlignTop)

            if blurb:
                hint = QLabel(blurb)
                hint.setStyleSheet("color: #666; font-size: 10px;")
                hint.setWordWrap(True)
                form.addRow(hint)

            for name in params:
                spec = PARAM_SCHEMA[name]
                widget = self._make_widget(name, spec)
                tip = spec.get("tip")
                if tip:
                    widget.setToolTip(tip)
                self.widgets[name] = widget
                form.addRow(QLabel(name), widget)

            if step_key is not None:
                btn = QPushButton(f"Re-run from {step_key}")
                btn.setStyleSheet("QPushButton { font-weight: bold; padding: 6px; }")
                btn.clicked.connect(lambda _=False, s=step_key: self.on_run_from(s))
                form.addRow(btn)

            body_layout.addWidget(box)

        # Trailing "run everything" button.
        all_btn = QPushButton("Run full pipeline on selected stems")
        all_btn.setStyleSheet("QPushButton { padding: 6px; }")
        all_btn.clicked.connect(lambda: self.on_run_from(None))
        body_layout.addWidget(all_btn)

        body_layout.addStretch(1)

    def _make_widget(self, name: str, spec: dict) -> QWidget:
        kind = spec["type"]
        if kind == "float":
            w = QDoubleSpinBox()
            mn, mx, step, *rest = spec["range"]
            decimals = rest[0] if rest else 3
            w.setRange(float(mn), float(mx))
            w.setSingleStep(float(step))
            w.setDecimals(int(decimals))
            w.valueChanged.connect(lambda _: self.on_value_changed())
            return w
        if kind == "int":
            w = QSpinBox()
            mn, mx, step = spec["range"]
            w.setRange(int(mn), int(mx))
            w.setSingleStep(int(step))
            w.valueChanged.connect(lambda _: self.on_value_changed())
            return w
        if kind == "bool":
            w = QCheckBox()
            w.toggled.connect(lambda _: self.on_value_changed())
            return w
        if kind == "choice":
            w = QComboBox()
            w.addItems(list(spec.get("choices", [])))
            if spec.get("editable"):
                w.setEditable(True)
            w.currentTextChanged.connect(lambda _: self.on_value_changed())
            return w
        # fallback: plain text
        w = QLineEdit()
        w.textChanged.connect(lambda _: self.on_value_changed())
        return w

    def load_from_config(self, config: dict) -> None:
        params = config.get("params", {}) or {}
        for name, widget in self.widgets.items():
            spec = PARAM_SCHEMA[name]
            value = params.get(name)
            if value is None:
                continue
            kind = spec["type"]
            try:
                if kind == "float":
                    widget.blockSignals(True)
                    widget.setValue(float(value))
                    widget.blockSignals(False)
                elif kind == "int":
                    widget.blockSignals(True)
                    widget.setValue(int(value))
                    widget.blockSignals(False)
                elif kind == "bool":
                    widget.blockSignals(True)
                    widget.setChecked(bool(value))
                    widget.blockSignals(False)
                elif kind == "choice":
                    widget.blockSignals(True)
                    text = str(value)
                    idx = widget.findText(text)
                    if idx >= 0:
                        widget.setCurrentIndex(idx)
                    elif widget.isEditable():
                        widget.setEditText(text)
                    widget.blockSignals(False)
                else:
                    widget.blockSignals(True)
                    widget.setText(str(value))
                    widget.blockSignals(False)
            except Exception:
                pass

    def collect_values(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, widget in self.widgets.items():
            spec = PARAM_SCHEMA[name]
            kind = spec["type"]
            if kind == "float":
                out[name] = float(widget.value())
            elif kind == "int":
                out[name] = int(widget.value())
            elif kind == "bool":
                out[name] = bool(widget.isChecked())
            elif kind == "choice":
                out[name] = widget.currentText()
            else:
                out[name] = widget.text()
        return out


# ---------- main window ----------


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Schematic Pipeline – manual annotation")
        self.resize(1500, 950)

        self.config = ensure_session_config()
        self.process: QProcess | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        reset_btn = QPushButton("Reset session to defaults")
        reset_btn.clicked.connect(self._on_reset)
        toolbar.addWidget(reset_btn)

        save_btn = QPushButton("Save current as defaults")
        save_btn.clicked.connect(self._on_save_as_defaults)
        toolbar.addWidget(save_btn)

        self.session_label = QLabel(f"  session: {SESSION_CONFIG.name}")
        self.session_label.setStyleSheet("color: #555;")
        toolbar.addWidget(self.session_label)

        central = QSplitter(Qt.Vertical)
        self.setCentralWidget(central)

        upper = QSplitter(Qt.Horizontal)
        central.addWidget(upper)

        self.stem_panel = StemListPanel(RESULT_DIR)
        self.stem_panel.list_widget.currentItemChanged.connect(self._on_stem_changed)
        upper.addWidget(self.stem_panel)

        self.vis_panel = VisTabsPanel(
            RESULT_DIR, on_edit_commit=self._on_edit_committed
        )
        upper.addWidget(self.vis_panel)

        self.param_panel = ParamPanel(
            self.config,
            on_value_changed=self._on_param_changed,
            on_run_from=self._on_run_requested,
        )
        upper.addWidget(self.param_panel)

        upper.setStretchFactor(0, 1)
        upper.setStretchFactor(1, 4)
        upper.setStretchFactor(2, 2)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        log_font = QFont("Consolas", 9)
        log_font.setStyleHint(QFont.Monospace)
        self.log_view.setFont(log_font)
        self.log_view.setPlaceholderText("Subprocess output appears here.")
        central.addWidget(self.log_view)

        central.setStretchFactor(0, 5)
        central.setStretchFactor(1, 2)

        self.statusBar().showMessage("Ready.")

        # Auto-select the first item if any.
        if self.stem_panel.list_widget.count():
            self.stem_panel.list_widget.setCurrentRow(0)

    # ---- event handlers ----

    def _on_stem_changed(self, current: QListWidgetItem | None, _previous):
        if current is None:
            self.vis_panel.show_stem(None, [])
            return
        stem = current.data(Qt.UserRole)
        self.vis_panel.show_stem(stem, self.stem_panel.red_flags_for(stem))

    def _on_param_changed(self) -> None:
        # Persist every edit straight to the session yaml.
        try:
            values = self.param_panel.collect_values()
            self.config.setdefault("params", {}).update(values)
            _dump_yaml(self.config, SESSION_CONFIG)
            self.statusBar().showMessage(
                f"Saved session config @ {SESSION_CONFIG}", 2000
            )
        except Exception as exc:
            self.statusBar().showMessage(f"Save failed: {exc}", 4000)

    def _on_reset(self) -> None:
        confirm = QMessageBox.question(
            self,
            "Reset session",
            "Reload session config from pipeline_config.yaml (discard current edits)?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        self.config = reset_session_from_default()
        self.param_panel.load_from_config(self.config)
        self.statusBar().showMessage("Session reset to defaults.", 3000)

    def _on_save_as_defaults(self) -> None:
        confirm = QMessageBox.question(
            self,
            "Overwrite defaults",
            f"Overwrite {DEFAULT_CONFIG.name} with the current session values?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        try:
            self._on_param_changed()  # ensure session yaml is current
            shutil.copy2(SESSION_CONFIG, DEFAULT_CONFIG)
            self.statusBar().showMessage(f"Wrote {DEFAULT_CONFIG.name}.", 3000)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def _on_run_requested(self, from_step: str | None) -> None:
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "Busy", "A pipeline run is already in progress.")
            return
        stems = self.stem_panel.current_selection()
        if not stems:
            QMessageBox.information(
                self,
                "No selection",
                "Select one or more stems in the left panel first.",
            )
            return
        # Make sure session yaml reflects the latest widget values.
        self._on_param_changed()
        self._launch_pipeline(from_step, stems)

    def _on_edit_committed(self, stem: str, from_step: str) -> None:
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            QMessageBox.warning(
                self,
                "Busy",
                "A pipeline run is already in progress; commit completed but "
                "re-run was not started.",
            )
            return
        # Make sure session yaml reflects the latest widget values.
        self._on_param_changed()
        self._launch_pipeline(from_step, [stem])

    def _launch_pipeline(self, from_step: str | None, stems: list[str]) -> None:
        # `-u` forces unbuffered stdout so the log panel updates live.
        args = ["-u", str(RUN_PIPELINE), "--config", str(SESSION_CONFIG)]
        if from_step is not None:
            args += ["--from-step", from_step]
        for stem in stems:
            args += ["--stem", stem]
        # Stop pipeline from copying inputs (they already live in result/images).
        args += ["--no-copy-inputs"]

        scope = "all steps" if from_step is None else f"--from-step {from_step}"
        target = ", ".join(stems[:5]) + (" …" if len(stems) > 5 else "")
        banner = f"\n>>> running pipeline ({scope}) on {len(stems)} stem(s): {target}\n"
        self.log_view.appendPlainText(banner)

        process = QProcess(self)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.setWorkingDirectory(str(REPO_ROOT))
        # Inherit parent env (USERPROFILE, PATH, CUDA_PATH, ...) — calling
        # processEnvironment() returns an EMPTY object, not a copy of the
        # current env, so we must seed from systemEnvironment() explicitly,
        # otherwise the child process loses HOME/USERPROFILE and matplotlib
        # crashes on `Path("~").expanduser()`.
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("KMP_DUPLICATE_LIB_OK", "TRUE")
        env.insert("EASYOCR_MODULE_PATH", str(REPO_ROOT / ".cache" / "easyocr"))
        env.insert("MPLCONFIGDIR", str(REPO_ROOT / ".cache" / "matplotlib"))
        env.insert("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))
        env.insert("YOLO_CONFIG_DIR", str(REPO_ROOT / ".cache" / "ultralytics"))
        process.setProcessEnvironment(env)
        process.readyReadStandardOutput.connect(self._on_proc_output)
        process.finished.connect(self._on_proc_finished)
        process.errorOccurred.connect(self._on_proc_error)

        self.process = process
        self._pending_stems = stems
        self.statusBar().showMessage(f"Running pipeline on {len(stems)} stem(s)...")
        process.start(sys.executable, args)

    def _on_proc_output(self) -> None:
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        if data:
            self.log_view.moveCursor(QTextCursor.End)
            self.log_view.insertPlainText(data)
            self.log_view.moveCursor(QTextCursor.End)

    def _on_proc_error(self, _err) -> None:
        if self.process is None:
            return
        self.log_view.appendPlainText(f"[error] {self.process.errorString()}")

    def _on_proc_finished(self, exit_code: int, _status) -> None:
        self.log_view.appendPlainText(f"<<< pipeline exited with code {exit_code}\n")
        self.process = None
        self.statusBar().showMessage(f"Pipeline finished (exit={exit_code}).", 5000)
        # Refresh red-flag map + viz for the active stem.
        self.stem_panel.refresh()
        current = self.stem_panel.list_widget.currentItem()
        if current is not None:
            stem = current.data(Qt.UserRole)
            self.vis_panel.show_stem(stem, self.stem_panel.red_flags_for(stem))


# ---------- entry point ----------


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
