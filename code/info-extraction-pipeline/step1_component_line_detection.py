#!/usr/bin/env python3

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import torch

# Placeholder defaults. run_pipeline.load_step1() overwrites every constant in
# this block with paths derived from pipeline_config.yaml, so the values here
# only document the expected layout.
ROOT = Path(__file__).resolve().parent
YOLO_WEIGHTS = ROOT / "componentDetection" / "best.pt"
JJ_YOLO_WEIGHTS = ROOT / "junction_jump_detection" / "best.pt"
# Older 2-class (junction, jump) model, unioned in to recover basic
# junction/jump detections the 5-class model misses. Optional: if the file is
# absent, detection falls back to the 5-class model alone.
JJ_YOLO_WEIGHTS_2CLS = ROOT / "junction_jump_detection" / "best2cls.pt"
TEST_IMAGES = ROOT / "testData"

MASKED_DIR = ROOT / "maskedImages"
MASKED_NOTEXT_DIR = ROOT / "maskedNoTextImages"

HAWP_DIR = ROOT / "hawp"
HAWP_WEIGHTS = HAWP_DIR / "bestv3.pth"
HAWP_CONFIG = HAWP_DIR / "hawp" / "ssl" / "config" / "hawpv3.yaml"

COMBINED_RESULTS_DIR = ROOT / "testDataResults"
COMBINED_LINES_DIR = COMBINED_RESULTS_DIR / "lines"
COMBINED_VIS_DIR = COMBINED_RESULTS_DIR / "visualizations"

NODE_DIR = COMBINED_LINES_DIR.parent / "nodes_results"
NODE_VIS_DIR = NODE_DIR / "vis"
NODE_DATA_DIR = NODE_DIR / "data"

# Set by run_pipeline to result/manual_jj_overrides. When a {stem}.json exists
# there, its added_jumps / added_junctions are unioned with the YOLO output in
# run_component_anchor_nodes. None disables the hook.
MANUAL_JJ_OVERRIDES_DIR = None


PLTOPTS = {"color": "#33FFFF", "s": 15, "edgecolors": "none", "zorder": 5}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _filter_by_stems(paths, stems):
    if not stems:
        return list(paths)
    selected = set(stems)
    return [p for p in paths if p.stem in selected]


# --------------------------------------------------------------------------- #
# YOLO component detection + component masking                                 #
# --------------------------------------------------------------------------- #
def run_yolo_and_mask(args, device):
    from ultralytics import YOLO

    print(f"[YOLO] loading {YOLO_WEIGHTS}")
    model = YOLO(str(YOLO_WEIGHTS))

    images = sorted([p for p in TEST_IMAGES.iterdir() if p.suffix.lower() in IMG_EXTS])
    images = _filter_by_stems(images, getattr(args, "stem", None))
    print(f"[YOLO] found {len(images)} images in {TEST_IMAGES}")

    MASKED_DIR.mkdir(parents=True, exist_ok=True)
    bbox_dir = MASKED_DIR / "_bboxes"
    bbox_dir.mkdir(exist_ok=True)

    fill_value = 255 if args.mask_color == "white" else 0

    for img_path in images:
        result = model.predict(
            source=str(img_path),
            conf=args.conf,
            iou=args.iou,
            device=device,
            agnostic_nms=True,
            verbose=False,
        )[0]

        im = skimage.io.imread(str(img_path))
        if im.ndim == 2:
            im = np.repeat(im[:, :, None], 3, 2)
        im = im[:, :, :3].copy()
        H, W = im.shape[:2]

        boxes = (
            result.boxes.xyxy.cpu().numpy()
            if result.boxes is not None
            else np.zeros((0, 4))
        )
        cls = (
            result.boxes.cls.cpu().numpy().astype(int)
            if result.boxes is not None
            else np.zeros((0,), dtype=int)
        )
        confs = (
            result.boxes.conf.cpu().numpy()
            if result.boxes is not None
            else np.zeros((0,))
        )

        masked = im.copy()
        for x1, y1, x2, y2 in boxes:
            x1 = max(0, int(round(x1)) - args.pad)
            y1 = max(0, int(round(y1)) - args.pad)
            x2 = min(W, int(round(x2)) + args.pad)
            y2 = min(H, int(round(y2)) + args.pad)
            masked[y1:y2, x1:x2] = fill_value

        out_path = MASKED_DIR / img_path.name
        skimage.io.imsave(str(out_path), masked, check_contrast=False)

        # persist bbox metadata so the masking step is reproducible / inspectable
        meta = np.zeros((len(boxes), 6), dtype=np.float32)
        if len(boxes):
            meta[:, :4] = boxes
            meta[:, 4] = cls
            meta[:, 5] = confs
        np.savetxt(
            bbox_dir / f"{img_path.stem}.txt",
            meta,
            fmt="%.2f %.2f %.2f %.2f %d %.4f",
            header="x1 y1 x2 y2 cls conf",
            comments="",
        )

        print(f"[YOLO] {img_path.name}: {len(boxes)} components -> {out_path.name}")


# --------------------------------------------------------------------------- #
# OCR text removal (called by run_combined_stage)                              #
# --------------------------------------------------------------------------- #
def run_ocr_text_removal(device, pad=2, conf_thresh=0.2, force=False, stems=None):
    import cv2

    images = sorted(
        [
            p
            for p in MASKED_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ]
    )
    images = _filter_by_stems(images, stems)

    MASKED_NOTEXT_DIR.mkdir(parents=True, exist_ok=True)
    bbox_dir = MASKED_NOTEXT_DIR / "_text_bboxes"
    bbox_dir.mkdir(parents=True, exist_ok=True)

    # Decide which images actually need OCR. An image counts as cached only if
    # both the cleaned image AND its bbox metadata are already on disk.
    if force:
        todo = images
        cached_count = 0
    else:
        todo = []
        cached_count = 0
        for p in images:
            out_img = MASKED_NOTEXT_DIR / p.name
            out_meta = bbox_dir / f"{p.stem}.txt"
            if out_img.exists() and out_meta.exists():
                cached_count += 1
            else:
                todo.append(p)

    if not todo:
        print(
            f"[OCR] all {cached_count} image(s) already cached in "
            f"{MASKED_NOTEXT_DIR} - pass --force-ocr to regenerate"
        )
        return

    try:
        import easyocr
    except ImportError as e:
        raise RuntimeError(
            "easyocr is required for OCR text removal. "
            "Install with: pip install easyocr"
        ) from e

    use_gpu = device.type == "cuda"
    print(f"[OCR] initializing EasyOCR (gpu={use_gpu})...")
    reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)

    print(
        f"[OCR] processing {len(todo)} image(s) "
        f"({cached_count} cached, pad={pad} px, conf>={conf_thresh})"
    )

    for img_path in todo:
        im = cv2.imread(str(img_path))
        if im is None:
            print(f"[OCR] skip unreadable {img_path}")
            continue
        H, W = im.shape[:2]

        # EasyOCR returns [(quad_4pts, text, conf), ...]; quads are upright on
        # the masked schematics so taking the axis-aligned outer rect is fine.
        # results = reader.readtext(str(img_path))
        # Force image into 2D grayscale for EasyOCR
        if im.ndim == 3:
            if im.shape[2] == 4:
                gray = cv2.cvtColor(im, cv2.COLOR_BGRA2GRAY)
            else:
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        elif im.ndim == 2:
            gray = im
        else:
            print(
                f"[OCR] unsupported image shape {im.shape}, skipping: {img_path.name}"
            )
            continue

        # EasyOCR prefers uint8
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        try:
            results = reader.readtext(gray)
        except Exception as e:
            print(f"[OCR] EasyOCR failed on {img_path.name}: {e}")
            continue
        kept = []
        for bbox, text, conf in results:
            if conf < conf_thresh:
                continue
            xs = [int(round(p[0])) for p in bbox]
            ys = [int(round(p[1])) for p in bbox]
            x1 = max(0, min(xs) - pad)
            y1 = max(0, min(ys) - pad)
            x2 = min(W, max(xs) + pad)
            y2 = min(H, max(ys) + pad)
            if x2 <= x1 or y2 <= y1:
                continue
            im[y1:y2, x1:x2] = 255
            kept.append((x1, y1, x2, y2, conf, text))

        cv2.imwrite(str(MASKED_NOTEXT_DIR / img_path.name), im)
        with open(bbox_dir / f"{img_path.stem}.txt", "w", encoding="utf-8") as f:
            f.write("# x1 y1 x2 y2 conf text\n")
            for x1, y1, x2, y2, conf, text in kept:
                clean = text.replace("\t", " ").replace("\n", " ")
                f.write(f"{x1} {y1} {x2} {y2} {conf:.4f} {clean}\n")

        print(f"[OCR] {img_path.name}: removed {len(kept)} text region(s)")


def _read_bbox_meta(stem):
    path = MASKED_DIR / "_bboxes" / f"{stem}.txt"
    if not path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    arr = np.loadtxt(str(path), skiprows=1, ndmin=2)
    if arr.size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    # columns: x1 y1 x2 y2 cls conf -> we only need x1 y1 x2 y2 cls
    return arr[:, :5].astype(np.float32)


# --------------------------------------------------------------------------- #
# HAWPv3 wireframe parsing                                                     #
# --------------------------------------------------------------------------- #
def build_hawp(device):
    # Lazy: HAWP brings yacs / easydict / hafm csrc - only import when running.
    sys.path.insert(0, str(HAWP_DIR))
    from hawp.fsl.config import cfg as hawp_cfg
    from hawp.ssl.models import MODELS

    print(f"[HAWP] loading config {HAWP_CONFIG}")
    hawp_cfg.merge_from_file(str(HAWP_CONFIG))
    print(f"[HAWP] loading checkpoint {HAWP_WEIGHTS}")
    state_dict = torch.load(str(HAWP_WEIGHTS), map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    # Real HAWPv3 was trained with USE_LINE_HEATMAP=True + gray_scale input.
    model = MODELS["HAWP-heatmap"](hawp_cfg, gray_scale=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(
            f"[HAWP] WARNING missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    if unexpected:
        print(
            f"[HAWP] WARNING unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
        )
    model = model.to(device).eval()
    return model


def run_hawp_on_image(model, im_gray, device, fixed_size=512):
    import cv2  # local: keep top-level imports light

    H, W = im_gray.shape[:2]
    im_resized = cv2.resize(im_gray, (fixed_size, fixed_size))
    tensor = torch.from_numpy(im_resized).float().div_(255.0)[None, None].to(device)
    meta = {"width": W, "height": H, "filename": ""}
    try:
        with torch.no_grad():
            outputs, _ = model(tensor, [meta])
    except (IndexError, RuntimeError) as e:
        # No junctions found -> wireframe matcher reduces over an empty dim. Treat
        # as "model saw nothing here" rather than crashing the whole stage.
        print(f"[HAWP] no wireframe found (matcher returned empty): {e}")
        return np.zeros((0, 2, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    if "lines_pred" not in outputs or outputs["lines_pred"].numel() == 0:
        return np.zeros((0, 2, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    # outputs["lines_pred"] shape (N, 4): x1, y1, x2, y2 in original-image pixels.
    flat = outputs["lines_pred"].cpu().numpy()
    scores = outputs["lines_score"].cpu().numpy()
    lines = flat.reshape(-1, 2, 2)  # [[x1,y1],[x2,y2]]
    return lines.astype(np.float32), scores.astype(np.float32)


# --------------------------------------------------------------------------- #
# Combined: HAWPv3 -> clip vs component bboxes -> drop lines near OCR text    #
# --------------------------------------------------------------------------- #
def _read_text_bboxes(stem):
    path = MASKED_NOTEXT_DIR / "_text_bboxes" / f"{stem}.txt"
    if not path.exists():
        return np.zeros((0, 4), dtype=np.float32)
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                out.append((int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])))
            except ValueError:
                continue
    if not out:
        return np.zeros((0, 4), dtype=np.float32)
    return np.array(out, dtype=np.float32)


def _clip_lines_outside_bboxes(lines, bboxes, min_len=4.0):
    from shapely.geometry import LineString, MultiLineString, box as shp_box
    from shapely.ops import unary_union

    if len(lines) == 0:
        return np.zeros((0, 2, 2), dtype=np.float32)
    if len(bboxes) == 0:
        return np.asarray(lines, dtype=np.float32).reshape(-1, 2, 2)

    bbox_union = unary_union(
        [shp_box(float(b[0]), float(b[1]), float(b[2]), float(b[3])) for b in bboxes]
    )

    pieces = []
    for ln in lines:
        seg = LineString(
            [(float(ln[0, 0]), float(ln[0, 1])), (float(ln[1, 0]), float(ln[1, 1]))]
        )
        if seg.length <= 0:
            continue
        outside = seg.difference(bbox_union)
        if outside.is_empty:
            continue
        if isinstance(outside, LineString):
            geoms = [outside]
        elif isinstance(outside, MultiLineString):
            geoms = list(outside.geoms)
        else:
            # GeometryCollection or other - extract any LineStrings
            geoms = [
                g for g in getattr(outside, "geoms", []) if isinstance(g, LineString)
            ]
        for g in geoms:
            coords = list(g.coords)
            if len(coords) < 2:
                continue
            p0 = coords[0]
            p1 = coords[-1]
            if abs(p0[0] - p1[0]) + abs(p0[1] - p1[1]) < min_len:
                continue
            pieces.append([[p0[0], p0[1]], [p1[0], p1[1]]])

    if not pieces:
        return np.zeros((0, 2, 2), dtype=np.float32)
    return np.asarray(pieces, dtype=np.float32)


def _drop_lines_on_whitespace(
    lines,
    im_gray,
    white_threshold=240,
    fraction=0.8,
    min_len=4.0,
):
    import numpy as np

    lines = np.asarray(lines, dtype=np.float32).reshape(-1, 2, 2)

    if len(lines) == 0:
        return np.zeros((0, 2, 2), dtype=np.float32)

    H, W = im_gray.shape[:2]
    pieces = []

    for ln in lines:
        x1, y1 = float(ln[0, 0]), float(ln[0, 1])
        x2, y2 = float(ln[1, 0]), float(ln[1, 1])

        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))

        if length <= 0:
            continue

        # One sample approximately per pixel of line length.
        # +1 makes endpoint handling a little more stable.
        n_samples = max(2, int(np.ceil(length)) + 1)

        ts = np.linspace(0.0, 1.0, n_samples)

        xs = np.clip(np.rint(x1 + ts * dx).astype(int), 0, W - 1)
        ys = np.clip(np.rint(y1 + ts * dy).astype(int), 0, H - 1)

        vals = im_gray[ys, xs]

        # True means this sample is background and should be cut away.
        white_mask = vals >= white_threshold

        # True means this sample should be kept.
        keep_mask = ~white_mask

        if not np.any(keep_mask):
            continue

        # Convert sample centers into t-interval cells.
        # Example:
        #   sample t:  t0, t1, t2, ...
        #   interval: [edge0, edge1], [edge1, edge2], ...
        edges = np.zeros(n_samples + 1, dtype=np.float32)
        edges[0] = 0.0
        edges[-1] = 1.0
        if n_samples > 1:
            edges[1:-1] = (ts[:-1] + ts[1:]) / 2.0

        # Find continuous runs of keep_mask == True
        i = 0
        while i < n_samples:
            if not keep_mask[i]:
                i += 1
                continue

            start = i
            while i + 1 < n_samples and keep_mask[i + 1]:
                i += 1
            end = i

            t0 = float(edges[start])
            t1 = float(edges[end + 1])

            seg_len = (t1 - t0) * length

            if seg_len >= min_len:
                q0 = np.array([x1 + t0 * dx, y1 + t0 * dy], dtype=np.float32)
                q1 = np.array([x1 + t1 * dx, y1 + t1 * dy], dtype=np.float32)

                pieces.append(
                    [
                        [float(q0[0]), float(q0[1])],
                        [float(q1[0]), float(q1[1])],
                    ]
                )

            i += 1

    if not pieces:
        return np.zeros((0, 2, 2), dtype=np.float32)

    return np.asarray(pieces, dtype=np.float32)


def _point_segment_closest(p, a, b):
    """
    Return distance and closest point from point p to segment ab.
    """
    import numpy as np

    p = np.asarray(p, dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-6:
        q = a.copy()
        return float(np.linalg.norm(p - q)), q

    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    q = a + t * ab
    return float(np.linalg.norm(p - q)), q


def _ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a, b, c, d):
    a = tuple(a)
    b = tuple(b)
    c = tuple(c)
    d = tuple(d)
    return (_ccw(a, c, d) != _ccw(b, c, d)) and (_ccw(a, b, c) != _ccw(a, b, d))


def _segment_segment_distance(a, b, c, d):
    import numpy as np

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    c = np.asarray(c, dtype=np.float32)
    d = np.asarray(d, dtype=np.float32)

    if _segments_intersect(a, b, c, d):
        return 0.0

    d1, _ = _point_segment_closest(a, c, d)
    d2, _ = _point_segment_closest(b, c, d)
    d3, _ = _point_segment_closest(c, a, b)
    d4, _ = _point_segment_closest(d, a, b)

    return float(min(d1, d2, d3, d4))


def _segment_segment_closest_points(a, b, c, d):
    import numpy as np

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    c = np.asarray(c, dtype=np.float32)
    d = np.asarray(d, dtype=np.float32)

    if _segments_intersect(a, b, c, d):
        r = b - a
        s = d - c
        denom = float(r[0] * s[1] - r[1] * s[0])
        if abs(denom) > 1e-9:
            t = float((c[0] - a[0]) * s[1] - (c[1] - a[1]) * s[0]) / denom
            t = max(0.0, min(1.0, t))
            p = a + r * t
            return 0.0, p.astype(np.float32)
        return 0.0, ((a + b + c + d) * 0.25).astype(np.float32)

    candidates = []
    da, qa = _point_segment_closest(a, c, d)
    candidates.append((da, a, qa))
    db, qb = _point_segment_closest(b, c, d)
    candidates.append((db, b, qb))
    dc, qc = _point_segment_closest(c, a, b)
    candidates.append((dc, qc, c))
    dd, qd = _point_segment_closest(d, a, b)
    candidates.append((dd, qd, d))

    dist, p_on_ab, p_on_cd = min(candidates, key=lambda t: t[0])
    contact = (
        np.asarray(p_on_ab, dtype=np.float32) + np.asarray(p_on_cd, dtype=np.float32)
    ) * 0.5
    return float(dist), contact.astype(np.float32)


def _segment_segment_closest_pair(a, b, c, d):
    import numpy as np

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    c = np.asarray(c, dtype=np.float32)
    d = np.asarray(d, dtype=np.float32)

    if _segments_intersect(a, b, c, d):
        _, p = _segment_segment_closest_points(a, b, c, d)
        return 0.0, p, p

    candidates = []
    da, qa = _point_segment_closest(a, c, d)
    candidates.append((da, a, qa))
    db, qb = _point_segment_closest(b, c, d)
    candidates.append((db, b, qb))
    dc, qc = _point_segment_closest(c, a, b)
    candidates.append((dc, qc, c))
    dd, qd = _point_segment_closest(d, a, b)
    candidates.append((dd, qd, d))

    dist, p_on_ab, p_on_cd = min(candidates, key=lambda t: t[0])
    return (
        float(dist),
        np.asarray(p_on_ab, dtype=np.float32),
        np.asarray(p_on_cd, dtype=np.float32),
    )


def _segment_black_fill_fraction(im_bin, p1, p2, radius=1, black_threshold=128):
    import numpy as np

    if im_bin is None:
        return 0.0
    H, W = im_bin.shape[:2]
    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)
    dist = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    n = max(2, int(round(dist)) + 1)
    r = int(max(0, radius))

    hit = 0
    total = 0
    for t in np.linspace(0.0, 1.0, n):
        x = int(round(float(p1[0] + (p2[0] - p1[0]) * t)))
        y = int(round(float(p1[1] + (p2[1] - p1[1]) * t)))
        if x < 0 or y < 0 or x >= W or y >= H:
            continue
        total += 1
        x0, x1 = max(0, x - r), min(W, x + r + 1)
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        if np.any(im_bin[y0:y1, x0:x1] < black_threshold):
            hit += 1

    if total == 0:
        return 0.0
    return hit / total


def _segments_perpendicular_deviation(line_a, line_b):
    import math
    import numpy as np

    la = np.asarray(line_a, dtype=np.float32)
    lb = np.asarray(line_b, dtype=np.float32)

    va = la[1] - la[0]
    vb = lb[1] - lb[0]

    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-6 or nb < 1e-6:
        return 90.0

    cos_abs = abs(float(np.dot(va, vb)) / (na * nb))
    cos_abs = max(0.0, min(1.0, cos_abs))
    angle_deg = math.degrees(math.acos(cos_abs))
    return abs(angle_deg - 90.0)


def _point_in_any_bbox(pt, bboxes, pad=0.0):
    if bboxes is None or len(bboxes) == 0:
        return False
    for b in bboxes:
        if _point_in_bbox(pt, b, pad=pad):
            return True
    return False


def _node_has_line_at_probe(
    node_line_indices, lines, probe, min_line_length, probe_tol
):
    import numpy as np

    probe = np.asarray(probe, dtype=np.float32)
    min_len = float(min_line_length)
    tol = float(probe_tol)

    for idx in node_line_indices:
        p1 = np.asarray(lines[idx][0], dtype=np.float32)
        p2 = np.asarray(lines[idx][1], dtype=np.float32)
        seg_len = float(np.linalg.norm(p2 - p1))
        if seg_len < min_len:
            continue
        d, _ = _point_segment_closest(probe, p1, p2)
        if d <= tol:
            return True
    return False


def _node_extends_past_contact(
    node_line_indices,
    lines,
    line,
    contact,
    probe_dist,
    probe_tol,
    min_line_length,
):
    import numpy as np

    line = np.asarray(line, dtype=np.float32)
    direction = line[1] - line[0]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return False
    u = direction / norm
    contact = np.asarray(contact, dtype=np.float32)
    pd = float(probe_dist)
    probe_plus = contact + u * pd
    probe_minus = contact - u * pd

    return _node_has_line_at_probe(
        node_line_indices, lines, probe_plus, min_line_length, probe_tol
    ) and _node_has_line_at_probe(
        node_line_indices, lines, probe_minus, min_line_length, probe_tol
    )


def _find_lines_touching_bbox(lines, bbox, pad=0.0):
    if pad > 0:
        padded = (
            float(bbox[0]) - pad,
            float(bbox[1]) - pad,
            float(bbox[2]) + pad,
            float(bbox[3]) + pad,
        )
    else:
        padded = bbox

    out = []
    for i, ln in enumerate(lines):
        if _segment_intersects_bbox(ln[0], ln[1], padded):
            out.append(i)
    return out


def _bbox_edges_with_inward_normals(bbox):
    import numpy as np

    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return [
        (
            "left",
            np.array([x1, y1], dtype=np.float32),
            np.array([x1, y2], dtype=np.float32),
            np.array([1.0, 0.0], dtype=np.float32),
        ),
        (
            "right",
            np.array([x2, y1], dtype=np.float32),
            np.array([x2, y2], dtype=np.float32),
            np.array([-1.0, 0.0], dtype=np.float32),
        ),
        (
            "top",
            np.array([x1, y1], dtype=np.float32),
            np.array([x2, y1], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ),
        (
            "bottom",
            np.array([x1, y2], dtype=np.float32),
            np.array([x2, y2], dtype=np.float32),
            np.array([0.0, -1.0], dtype=np.float32),
        ),
    ]


def _point_in_bbox(p, bbox, pad=0.0):
    x, y = float(p[0]), float(p[1])
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return (x1 - pad <= x <= x2 + pad) and (y1 - pad <= y <= y2 + pad)


def _segment_intersects_bbox(p1, p2, bbox):
    if _point_in_bbox(p1, bbox) or _point_in_bbox(p2, bbox):
        return True

    for _, e1, e2, _ in _bbox_edges_with_inward_normals(bbox):
        if _segments_intersect(p1, p2, e1, e2):
            return True

    return False


def _has_black_pixels_inside_bbox_near_contact(
    im_bin,
    contact_xy,
    inward_normal,
    black_threshold=128,
    inside_depth=4,
    patch_radius=1,
):
    import numpy as np

    h, w = im_bin.shape[:2]
    contact_xy = np.asarray(contact_xy, dtype=np.float32)
    inward_normal = np.asarray(inward_normal, dtype=np.float32)

    for step in range(0, inside_depth + 1):
        p = contact_xy + inward_normal * float(step)
        x = int(round(float(p[0])))
        y = int(round(float(p[1])))

        x0 = max(0, x - patch_radius)
        x1 = min(w, x + patch_radius + 1)
        y0 = max(0, y - patch_radius)
        y1 = min(h, y + patch_radius + 1)

        if x0 >= x1 or y0 >= y1:
            continue

        patch = im_bin[y0:y1, x0:x1]
        if np.any(patch < black_threshold):
            return True

    return False


def _select_nearest_lines_to_component_edges(
    lines,
    comp_bboxes,
    max_anchor_dist=None,
):
    import numpy as np

    selected = {}

    if len(lines) == 0 or len(comp_bboxes) == 0:
        print("no box edges being detected")
        return selected

    for bbox_idx, bbox in enumerate(comp_bboxes):
        for edge_name, e1, e2, inward in _bbox_edges_with_inward_normals(bbox):
            best_i = None
            best_d = float("inf")

            for i, ln in enumerate(lines):
                p1 = ln[0]
                p2 = ln[1]
                d = _segment_segment_distance(p1, p2, e1, e2)

                if d < best_d:
                    best_d = d
                    best_i = i

            if best_i is None:
                continue

            if max_anchor_dist is not None and best_d > max_anchor_dist:
                continue

            old = selected.get(best_i)
            if old is None or best_d < old["dist"]:
                selected[best_i] = {
                    "line_idx": int(best_i),
                    "bbox_idx": int(bbox_idx),
                    "bbox": np.asarray(bbox, dtype=np.float32),
                    "edge_name": edge_name,
                    "edge_start": e1,
                    "edge_end": e2,
                    "inward": inward,
                    "dist": float(best_d),
                }
    if len(selected) == 0:
        print("no line is near any box edge")

    return selected


def _extend_line_endpoint_toward_edge(
    line,
    edge_start,
    edge_end,
    inward,
    max_extend_px=12.0,
    extra_into_bbox_px=2.0,
):
    import numpy as np

    ln = np.asarray(line, dtype=np.float32).copy()
    p0 = ln[0]
    p1 = ln[1]

    d0, q0 = _point_segment_closest(p0, edge_start, edge_end)
    d1, q1 = _point_segment_closest(p1, edge_start, edge_end)

    if d0 <= d1:
        endpoint_idx = 0
        dist = d0
        contact = q0
    else:
        endpoint_idx = 1
        dist = d1
        contact = q1

    if dist > max_extend_px:
        return None, None

    target = contact + np.asarray(inward, dtype=np.float32) * float(extra_into_bbox_px)
    ln[endpoint_idx] = target

    return ln, contact


def _find_valid_component_anchor_lines(
    lines,
    im_bin,
    comp_bboxes,
    max_anchor_dist=None,
    max_extend_px=12.0,
    extra_into_bbox_px=2.0,
    black_threshold=128,
    inside_depth=4,
    patch_radius=1,
):
    nearest = _select_nearest_lines_to_component_edges(
        lines,
        comp_bboxes,
        max_anchor_dist=max_anchor_dist,
    )

    valid_anchor_indices = []
    valid_anchor_infos = []

    never_touch = True
    never_black = True

    for line_idx, info in nearest.items():
        bbox = info["bbox"]

        extended, contact = _extend_line_endpoint_toward_edge(
            lines[line_idx],
            info["edge_start"],
            info["edge_end"],
            info["inward"],
            max_extend_px=max_extend_px,
            extra_into_bbox_px=extra_into_bbox_px,
        )

        if extended is None:
            continue

        touches_bbox = _segment_intersects_bbox(extended[0], extended[1], bbox)
        if not touches_bbox:
            continue

        never_touch = False
        black_ok = _has_black_pixels_inside_bbox_near_contact(
            im_bin,
            contact,
            info["inward"],
            black_threshold=black_threshold,
            inside_depth=inside_depth,
            patch_radius=patch_radius,
        )

        if not black_ok:
            continue

        never_black = False
        valid_anchor_indices.append(int(line_idx))
        valid_anchor_infos.append(
            {
                "line_idx": int(line_idx),
                "bbox_idx": int(info["bbox_idx"]),
                "edge_name": info["edge_name"],
                "dist": float(info["dist"]),
                "contact": contact.astype(float).tolist(),
                "extended_line": extended.astype(float).tolist(),
            }
        )
    if never_touch:
        print("WARNING: no lines could be extended to touch any box edge")
    if never_black:
        print(
            "WARNING: no candidate anchor lines had black pixels inside the bbox near contact"
        )
    return valid_anchor_indices, valid_anchor_infos


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def _line_aabb(line):
    import numpy as np

    ln = np.asarray(line, dtype=np.float32)
    x1 = float(np.min(ln[:, 0]))
    y1 = float(np.min(ln[:, 1]))
    x2 = float(np.max(ln[:, 0]))
    y2 = float(np.max(ln[:, 1]))
    return x1, y1, x2, y2


def _aabb_disjoint(a, b, pad=0.0):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    return ax2 + pad < bx1 or bx2 + pad < ax1 or ay2 + pad < by1 or by2 + pad < ay1


def _bbox_overlaps_any(bbox, bboxes, pad=0.0):
    if bboxes is None or len(bboxes) == 0:
        return False
    return any(not _aabb_disjoint(bbox, other, pad=pad) for other in bboxes)


# Models are cached by weights path so each file is loaded once per process.
_JJ_YOLO_MODELS = {}


def _load_jj_yolo(weights):
    if weights is None:
        return None
    key = str(weights)
    if key in _JJ_YOLO_MODELS:
        return _JJ_YOLO_MODELS[key]
    if not Path(weights).exists():
        print(f"[jj-yolo] weights not found at {weights}; skipping that model")
        _JJ_YOLO_MODELS[key] = None
        return None
    from ultralytics import YOLO

    print(f"[jj-yolo] loading {weights}")
    model = YOLO(str(weights))
    _JJ_YOLO_MODELS[key] = model
    return model


# Center-edge slack (px) used when deciding whether a 2-class box duplicates /
# conflicts with a 5-class box. Junctions on a schematic sit well apart, so a
# small pad only absorbs the slight box offset between the two models.
_JJ_MERGE_PAD = 2.0


def _augment_missing(base_arr, base_conf, extra_boxes, extra_conf, exclude_boxes, pad=_JJ_MERGE_PAD):
    import numpy as np

    keep = [i for i, b in enumerate(extra_boxes) if not _bbox_overlaps_any(b, exclude_boxes, pad=pad)]
    if not keep:
        return base_arr, base_conf
    add = np.asarray([extra_boxes[i] for i in keep], dtype=np.float32).reshape(-1, 4)
    addc = np.asarray([extra_conf[i] for i in keep], dtype=np.float32).reshape(-1)
    if len(base_arr) == 0:
        return add, addc
    return (
        np.concatenate([base_arr, add], axis=0),
        np.concatenate([base_conf, addc], axis=0),
    )


def _run_jj_model(weights, img_path, num_classes, conf, device):
    import numpy as np

    out = [np.empty((0, 4), dtype=np.float32) for _ in range(num_classes)]
    out_conf = [np.empty((0,), dtype=np.float32) for _ in range(num_classes)]
    model = _load_jj_yolo(weights)
    if model is None:
        return out, out_conf
    result = model.predict(
        source=str(img_path), conf=float(conf), device=device, verbose=False
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return out, out_conf
    xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float32)
    cls = result.boxes.cls.cpu().numpy().astype(int)
    cfd = result.boxes.conf.cpu().numpy().astype(np.float32)
    for c in range(num_classes):
        mask = cls == c
        if mask.any():
            out[c] = xyxy[mask].copy()
            out_conf[c] = cfd[mask].copy()
    return out, out_conf


def _detect_junctions_and_jumps(img_path, conf=0.25, device=None, trust_yolo=False):

    import numpy as np

    b5, c5 = _run_jj_model(JJ_YOLO_WEIGHTS, img_path, 5, conf, device)
    boxes = list(b5)
    confs = list(c5)

    g, gc = _run_jj_model(JJ_YOLO_WEIGHTS_2CLS, img_path, 2, conf, device)
    if len(g[0]) or len(g[1]):
        # Exclude against the 5-class boxes only (the new model's claims).
        exclude = [b for arr in boxes for b in arr]
        n0, n1 = len(boxes[0]), len(boxes[1])
        boxes[0], confs[0] = _augment_missing(boxes[0], confs[0], g[0], gc[0], exclude)
        boxes[1], confs[1] = _augment_missing(boxes[1], confs[1], g[1], gc[1], exclude)
        added_j, added_p = len(boxes[0]) - n0, len(boxes[1]) - n1
        if added_j or added_p:
            print(
                f"[jj-yolo] 2cls model recovered +{added_j} junction(s), "
                f"+{added_p} jump(s) missed by 5cls"
            )

    # Confidence-based junction vs crossover conflict resolution. Where a
    # junction box overlaps a jump/implicit/diag box, keep the HIGHER-confidence
    # detection (ties favor the junction). This replaces a blanket junction-wins,
    # so a clearly more confident jump at a spot is NOT turned into a junction.
    # Skipped in trust-YOLO mode (every raw detection is kept as-is).
    drop_jct = set()
    drop_cross = set()  # (class_id, index)
    for ji in ([] if trust_yolo else range(len(boxes[0]))):
        for ci in (1, 2, 3, 4):
            for ki in range(len(boxes[ci])):
                if not _bbox_overlaps_any(boxes[ci][ki], [boxes[0][ji]], pad=0.0):
                    continue
                if confs[ci][ki] > confs[0][ji]:
                    drop_jct.add(ji)
                else:
                    drop_cross.add((ci, ki))
    if drop_jct or drop_cross:
        def _filter(arr, drop):
            if not drop:
                return arr
            kept = [i for i in range(len(arr)) if i not in drop]
            return arr[kept] if kept else np.empty((0, 4), dtype=np.float32)

        boxes[0] = _filter(boxes[0], drop_jct)
        for ci in (1, 2, 3, 4):
            boxes[ci] = _filter(boxes[ci], {ki for (cc, ki) in drop_cross if cc == ci})
        print(
            f"[jj-yolo] conf-resolved junction/crossover overlaps: "
            f"dropped {len(drop_jct)} junction(s), {len(drop_cross)} crossover(s)"
        )

    return boxes[0], boxes[1], boxes[2], boxes[3], boxes[4]


def _line_extends_outward_past_edge(line, edge_point, outward_normal, min_len):

    import numpy as np

    p0 = np.asarray(line[0], dtype=np.float32)
    p1 = np.asarray(line[1], dtype=np.float32)
    ep = np.asarray(edge_point, dtype=np.float32)
    on = np.asarray(outward_normal, dtype=np.float32)
    ext0 = float(np.dot(p0 - ep, on))
    ext1 = float(np.dot(p1 - ep, on))
    return max(ext0, ext1) >= float(min_len)


def _implicit_jump_has_two_pairs(
    lines, bbox, min_line_len, comp_bboxes=None, comp_adjacent_px=0.0
):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    gap = float(comp_adjacent_px or 0.0)
    use_comp = gap > 0 and comp_bboxes is not None and len(comp_bboxes) > 0
    # Thin strip just outside each edge, used to test component adjacency.
    outward_strip = {
        "left": (x1 - gap, y1, x1, y2),
        "right": (x2, y1, x2 + gap, y2),
        "top": (x1, y1 - gap, x2, y1),
        "bottom": (x1, y2, x2, y2 + gap),
    }

    for edge_name, e1, e2, inward in _bbox_edges_with_inward_normals(bbox):
        outward = -inward
        wire_ok = any(
            _segments_intersect(ln[0], ln[1], e1, e2)
            and _line_extends_outward_past_edge(ln, e1, outward, min_line_len)
            for ln in lines
        )
        if wire_ok:
            continue
        if use_comp and _bbox_overlaps_any(outward_strip[edge_name], comp_bboxes, pad=0.0):
            continue
        return False
    return True


def _diag_jump_has_two_pairs(lines, bbox):
    has_pos = False
    has_neg = False
    for _edge_name, e1, e2, _inward in _bbox_edges_with_inward_normals(bbox):
        for ln in lines:
            if not _segments_intersect(ln[0], ln[1], e1, e2):
                continue
            dx = float(ln[1][0]) - float(ln[0][0])
            dy = float(ln[1][1]) - float(ln[0][1])
            slope_sign = dx * dy
            if abs(slope_sign) < 1e-6:
                continue
            if slope_sign > 0:
                has_pos = True
            else:
                has_neg = True
        if has_pos and has_neg:
            return True
    return has_pos and has_neg


def _route_jj_detections(
    lines,
    junction_bboxes,
    jump_bboxes,
    implicit_jump_bboxes,
    diag_jump_bboxes,
    diag_implicit_jump_bboxes,
    implicit_min_extend_px,
    explicit_min_extend_px,
    comp_bboxes=None,
    comp_adjacent_px=0.0,
    trust_yolo=False,
):
    import numpy as np

    def _arr(rows):
        if len(rows) == 0:
            return np.empty((0, 4), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32).reshape(-1, 4)

    junction_bboxes = _arr(junction_bboxes)
    jump_bboxes = _arr(jump_bboxes)
    implicit_jump_bboxes = _arr(implicit_jump_bboxes)
    diag_jump_bboxes = _arr(diag_jump_bboxes)
    diag_implicit_jump_bboxes = _arr(diag_implicit_jump_bboxes)

    if trust_yolo:
        # Trust the YOLO model directly: every detection maps straight to its
        # geometric category with NO validation, conf-resolution, junction-wins,
        # or implicit->junction conversion. junction -> junction; jump &
        # implicit_jump -> orthogonal crossover (jump); diag_jump &
        # diag_implicit_jump -> diagonal crossover (x_jump).
        out_jump = list(jump_bboxes) + list(implicit_jump_bboxes)
        out_x_jump = list(diag_jump_bboxes) + list(diag_implicit_jump_bboxes)
        print(
            f"[jj-route] trust-yolo: junction={len(junction_bboxes)} "
            f"jump={len(out_jump)} x_jump={len(out_x_jump)} (no validation)"
        )
        return _arr(junction_bboxes), _arr(out_jump), _arr(out_x_jump)

    has_junction = len(junction_bboxes) > 0

    out_junction = list(junction_bboxes)
    out_x_jump = list(diag_jump_bboxes)  # explicit diagonal, no validation

    # Explicit jumps (class 1): validated like implicit jumps (a wire crossing
    # all four edges) but at a relaxed threshold. A real crossover passes
    # easily; a model false-positive at a plain L-corner (only two arms) fails
    # and is dropped instead of wrongly splitting the wire.
    out_jump = []
    n_exp_drop = 0
    for bbox in jump_bboxes:
        if _implicit_jump_has_two_pairs(
            lines, bbox, explicit_min_extend_px, comp_bboxes, comp_adjacent_px
        ):
            out_jump.append(bbox)
        else:
            n_exp_drop += 1
    n_exp_keep = len(out_jump)

    # has_explicit_jump reflects only the kept (validated) explicit jumps.
    has_explicit_jump = n_exp_keep > 0
    implicit_as_junction = has_explicit_jump and not has_junction

    n_imp_jump = n_imp_junction = n_imp_drop = 0
    for bbox in implicit_jump_bboxes:
        if implicit_as_junction:
            out_junction.append(bbox)
            n_imp_junction += 1
        elif _implicit_jump_has_two_pairs(
            lines, bbox, implicit_min_extend_px, comp_bboxes, comp_adjacent_px
        ):
            out_jump.append(bbox)
            n_imp_jump += 1
        else:
            n_imp_drop += 1

    n_diag_imp_keep = n_diag_imp_drop = 0
    for bbox in diag_implicit_jump_bboxes:
        if _diag_jump_has_two_pairs(lines, bbox):
            out_x_jump.append(bbox)
            n_diag_imp_keep += 1
        else:
            n_diag_imp_drop += 1

    # Junction wins over crossover on overlap: if the model detected a junction
    # (a connection dot) where it also placed a jump / x_jump, trust the
    # junction and drop that crossover box. A spot drawn with a connection dot
    # is not a no-connect hop, so keeping the jump would wrongly override the
    # junction force-merge and split the net. (Manual GUI jumps are unioned in
    # later by _apply_manual_jj_overrides and are intentionally left untouched.)
    n_jump_drop_jct = n_xjump_drop_jct = 0
    if out_junction and (out_jump or out_x_jump):
        jct_arr = _arr(out_junction)
        kept_jump = [b for b in out_jump if not _bbox_overlaps_any(b, jct_arr, pad=0.0)]
        kept_xjump = [b for b in out_x_jump if not _bbox_overlaps_any(b, jct_arr, pad=0.0)]
        n_jump_drop_jct = len(out_jump) - len(kept_jump)
        n_xjump_drop_jct = len(out_x_jump) - len(kept_xjump)
        out_jump, out_x_jump = kept_jump, kept_xjump

    if (
        len(jump_bboxes)
        or len(implicit_jump_bboxes)
        or len(diag_implicit_jump_bboxes)
        or len(diag_jump_bboxes)
    ):
        print(
            f"[jj-route] explicit_jump: {n_exp_keep}/{len(jump_bboxes)} kept, "
            f"{n_exp_drop} dropped; "
            f"implicit_jump: +{n_imp_jump} as-jump, "
            f"+{n_imp_junction} as-junction, {n_imp_drop} dropped; "
            f"diag_jump: +{len(diag_jump_bboxes)}; "
            f"diag_implicit: +{n_diag_imp_keep} kept, {n_diag_imp_drop} dropped; "
            f"dropped-overlapping-junction: jump={n_jump_drop_jct} x_jump={n_xjump_drop_jct}"
        )

    return _arr(out_junction), _arr(out_jump), _arr(out_x_jump)


def _apply_manual_jj_overrides(stem, junction_bboxes, jump_bboxes):
    empty_x = np.empty((0, 4), dtype=np.float32)
    if MANUAL_JJ_OVERRIDES_DIR is None:
        return junction_bboxes, jump_bboxes, empty_x
    override_path = Path(MANUAL_JJ_OVERRIDES_DIR) / f"{stem}.json"
    if not override_path.exists():
        return junction_bboxes, jump_bboxes, empty_x
    try:
        import json as _json
        data = _json.loads(override_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[jj-overrides] {stem}: cannot read {override_path}: {exc}; skipping")
        return junction_bboxes, jump_bboxes, empty_x

    add_jumps = np.asarray(data.get("added_jumps") or [], dtype=np.float32).reshape(-1, 4)
    add_junctions = np.asarray(data.get("added_junctions") or [], dtype=np.float32).reshape(-1, 4)
    add_x_jumps = np.asarray(data.get("added_x_jumps") or [], dtype=np.float32).reshape(-1, 4)
    if len(add_junctions):
        junction_bboxes = np.concatenate([junction_bboxes, add_junctions], axis=0)
    if len(add_jumps):
        jump_bboxes = np.concatenate([jump_bboxes, add_jumps], axis=0)
    if len(add_junctions) or len(add_jumps) or len(add_x_jumps):
        print(
            f"[jj-overrides] {stem}: +{len(add_jumps)} jump(s), "
            f"+{len(add_junctions)} junction(s), +{len(add_x_jumps)} x-jump(s) "
            f"from {override_path.name}"
        )
    return junction_bboxes, jump_bboxes, add_x_jumps


def _group_lines_into_nodes_by_anchor_union_find(
    lines,
    valid_anchor_indices,
    node_union_dist=4.0,
    junction_bboxes=None,
    jump_bboxes=None,
    perp_tol_deg=45.0,
    bbox_pad=2.0,
    crossover_probe_dist=5.0,
    crossover_probe_tol=2.0,
    crossover_min_line_len=3.0,
    jump_parallel_tol_deg=20.0,
    x_jump_bboxes=None,
    im_bin=None,
    comp_bboxes=None,
    pixel_bridge_dist=0.0,
    pixel_bridge_fill_frac=0.6,
    pixel_bridge_radius=1,
    pixel_bridge_parallel_tol_deg=30.0,
    black_threshold=128,
):

    import math

    import numpy as np

    n = len(lines)
    line_node_ids = np.full(n, -1, dtype=np.int32)

    if n == 0 or len(valid_anchor_indices) == 0:
        return line_node_ids

    if junction_bboxes is None:
        junction_bboxes = np.empty((0, 4), dtype=np.float32)
    if jump_bboxes is None:
        jump_bboxes = np.empty((0, 4), dtype=np.float32)
    if x_jump_bboxes is None:
        x_jump_bboxes = np.empty((0, 4), dtype=np.float32)

    has_jump = len(jump_bboxes) > 0

    # ---- Pass 1: naive UF over every close pair, recording geometry. ----
    raw_uf = _UnionFind(n)
    aabbs = [_line_aabb(ln) for ln in lines]
    close_pairs = []  # (i, j, contact_xy, dev) with i < j

    pixel_bridge_on = (
        pixel_bridge_dist
        and pixel_bridge_dist > node_union_dist
        and im_bin is not None
    )
    aabb_pad = max(float(node_union_dist), float(pixel_bridge_dist or 0.0))

    for i in range(n):
        for j in range(i + 1, n):
            if _aabb_disjoint(aabbs[i], aabbs[j], pad=aabb_pad):
                continue

            d, contact = _segment_segment_closest_points(
                lines[i][0],
                lines[i][1],
                lines[j][0],
                lines[j][1],
            )

            if d > node_union_dist:
                # Pixel-evidence bridge: two fragments just past the union
                # distance still belong to one wire if a near-collinear gap
                # between them is filled with black pixels (the wire is
                # visible in the binary image even though line detection
                # produced no segment there). Guards: parallel-ish lines, the
                # gap roughly collinear with both, dense black fill, and the
                # gap does not cross a component bbox.
                if not pixel_bridge_on or d > pixel_bridge_dist:
                    continue
                dev = _segments_perpendicular_deviation(lines[i], lines[j])
                # dev is |angle-90|; parallel lines -> dev near 90.
                if (90.0 - dev) > pixel_bridge_parallel_tol_deg:
                    continue
                gd, p_i, p_j = _segment_segment_closest_pair(
                    lines[i][0], lines[i][1], lines[j][0], lines[j][1]
                )
                gap_vec = p_j - p_i
                gap_len = float(np.hypot(gap_vec[0], gap_vec[1]))
                if gap_len < 1e-3:
                    continue
                # Gap must run roughly along both line directions (collinear
                # break), not perpendicular to them.
                bad_dir = False
                for ln in (lines[i], lines[j]):
                    lv = np.asarray(ln[1], dtype=np.float32) - np.asarray(
                        ln[0], dtype=np.float32
                    )
                    lvn = float(np.hypot(lv[0], lv[1]))
                    if lvn < 1e-6:
                        continue
                    cos_abs = abs(float(gap_vec[0] * lv[0] + gap_vec[1] * lv[1]))
                    cos_abs /= gap_len * lvn
                    cos_abs = max(0.0, min(1.0, cos_abs))
                    if math.degrees(math.acos(cos_abs)) > pixel_bridge_parallel_tol_deg:
                        bad_dir = True
                        break
                if bad_dir:
                    continue
                if comp_bboxes is not None and len(comp_bboxes) > 0:
                    if any(
                        _segment_intersects_bbox(p_i, p_j, b) for b in comp_bboxes
                    ):
                        continue
                frac = _segment_black_fill_fraction(
                    im_bin,
                    p_i,
                    p_j,
                    radius=pixel_bridge_radius,
                    black_threshold=black_threshold,
                )
                if frac < pixel_bridge_fill_frac:
                    continue
                raw_uf.union(i, j)
                close_pairs.append((i, j, contact, dev))
                continue

            raw_uf.union(i, j)
            dev = _segments_perpendicular_deviation(lines[i], lines[j])
            close_pairs.append((i, j, contact, dev))

    # Junction bbox force-merge: any lines touching the same junction
    # bbox are unconditionally part of the same node, except when a jump
    # bbox overlaps it. In that conflict, trust the jump so the junction
    # force-merge cannot undo a crossover split.
    junction_force_groups = []
    for jbbox in junction_bboxes:
        if _bbox_overlaps_any(jbbox, jump_bboxes, pad=0.0) or _bbox_overlaps_any(
            jbbox, x_jump_bboxes, pad=0.0
        ):
            continue
        touching = _find_lines_touching_bbox(lines, jbbox, pad=bbox_pad)
        if len(touching) <= 1:
            continue
        junction_force_groups.append(touching)
        for k in range(1, len(touching)):
            raw_uf.union(touching[0], touching[k])

    # Jump bbox side-aware force-merge: after the angle-based split inside
    # a jump bbox, re-merge wire fragments by which bbox side they touch.
    # Lines crossing the left or right side -> one group (the horizontally
    # crossing wire); lines crossing the top or bottom side -> another
    # group (the vertical wire). Applied in Pass 3 only (not to raw_uf),
    # so it does not pollute the implicit-jump probe check.
    #
    # "Crossing a side" is a strict segment-segment intersection check, not
    # a pad-distance check: a vertical wire 2 px inside the bbox should
    # NOT be classified as touching the left side.
    jump_force_groups = []
    for jbbox in jump_bboxes:
        lr_set = set()
        tb_set = set()
        for edge_name, e1, e2, _ in _bbox_edges_with_inward_normals(jbbox):
            for k, ln in enumerate(lines):
                if not _segments_intersect(ln[0], ln[1], e1, e2):
                    continue
                if edge_name in ("left", "right"):
                    lr_set.add(k)
                else:
                    tb_set.add(k)
        if len(lr_set) > 1:
            jump_force_groups.append(sorted(lr_set))
        if len(tb_set) > 1:
            jump_force_groups.append(sorted(tb_set))

    # X-jump force-grouping: for each manually-added X-jump bbox, classify
    # lines crossing it by slope sign. dx*dy > 0 in image coords means the
    # line runs top-left -> bottom-right ("\" direction); dx*dy < 0 means
    # top-right -> bottom-left ("/" direction). Lines in each direction are
    # force-merged into one node; the two directions stay separate.
    x_jump_groups_by_box = []
    for jbbox in x_jump_bboxes:
        pos_slope_set = set()  # "\" direction
        neg_slope_set = set()  # "/" direction
        for edge_name, e1, e2, _ in _bbox_edges_with_inward_normals(jbbox):
            for k, ln in enumerate(lines):
                if not _segments_intersect(ln[0], ln[1], e1, e2):
                    continue
                dx = float(ln[1][0]) - float(ln[0][0])
                dy = float(ln[1][1]) - float(ln[0][1])
                slope_sign = dx * dy
                if abs(slope_sign) < 1e-6:
                    # Near-horizontal or near-vertical lines aren't part of
                    # the X pattern; let regular logic handle them.
                    continue
                if slope_sign > 0:
                    pos_slope_set.add(k)
                else:
                    neg_slope_set.add(k)
        groups = []
        if len(pos_slope_set) > 1:
            groups.append(sorted(pos_slope_set))
        if len(neg_slope_set) > 1:
            groups.append(sorted(neg_slope_set))
        # Keep the per-box groups together with the lines this box acts on, so we
        # can later check whether the force-merge actually split the crossover
        # into >= 2 distinct nodes.
        x_jump_groups_by_box.append(
            (jbbox, groups, sorted(pos_slope_set | neg_slope_set))
        )

    # Map raw root -> list of line indices in that raw node.
    raw_node_lines = {}
    for k in range(n):
        raw_node_lines.setdefault(raw_uf.find(k), []).append(k)

    # ---- Pass 2 + Pass 3, parameterized by which x_jump boxes are honored. ----
    # `honored_x_jump_bboxes` are the diagonal-crossover boxes allowed to drive
    # the in-box aggressive split AND contribute force-groups. A box that the
    # force-merge can't resolve into >= 2 distinct nodes is dropped from BOTH
    # (re-run as if it were never detected), so its region falls back to the
    # general perp-tol/probe crossover handling instead of being aggressively
    # split apart at every corner.
    def _finalize(honored_x_jump_bboxes, x_jump_groups):
        honored_has_x_jump = len(honored_x_jump_bboxes) > 0

        # ---- Pass 2: decide which pairs become splits. ----
        split_pairs = set()
        for i, j, contact, dev in close_pairs:
            in_jump = (
                (has_jump and _point_in_any_bbox(contact, jump_bboxes, pad=bbox_pad))
                or (
                    honored_has_x_jump
                    and _point_in_any_bbox(contact, honored_x_jump_bboxes, pad=bbox_pad)
                )
            )

            if in_jump:
                # Angle between segments, normalized to [0, 90] degrees.
                angle = 90.0 - dev
                should_split = angle > jump_parallel_tol_deg
            else:
                # Outside every jump / x_jump bbox: always fall through to the
                # generic perp-tol / junction-bbox / geometric-probe logic, even
                # when the image has explicit jumps elsewhere. This lets the probe
                # recover crossings the YOLO model missed entirely.
                if dev > perp_tol_deg:
                    continue
                in_junction = _point_in_any_bbox(contact, junction_bboxes, pad=bbox_pad)
                if in_junction:
                    should_split = False
                else:
                    node_i = raw_node_lines[raw_uf.find(i)]
                    node_j = raw_node_lines[raw_uf.find(j)]
                    extends_i = _node_extends_past_contact(
                        node_i,
                        lines,
                        lines[i],
                        contact,
                        probe_dist=crossover_probe_dist,
                        probe_tol=crossover_probe_tol,
                        min_line_length=crossover_min_line_len,
                    )
                    extends_j = _node_extends_past_contact(
                        node_j,
                        lines,
                        lines[j],
                        contact,
                        probe_dist=crossover_probe_dist,
                        probe_tol=crossover_probe_tol,
                        min_line_length=crossover_min_line_len,
                    )
                    should_split = extends_i and extends_j

            if should_split:
                split_pairs.add((i, j))

        # ---- Pass 3: rebuild UF skipping splits, then re-apply force-merge. ----
        uf = _UnionFind(n)
        for i, j, _, _ in close_pairs:
            if (i, j) in split_pairs:
                continue
            uf.union(i, j)

        for touching in junction_force_groups:
            for k in range(1, len(touching)):
                uf.union(touching[0], touching[k])

        for touching in jump_force_groups:
            for k in range(1, len(touching)):
                uf.union(touching[0], touching[k])

        for touching in x_jump_groups:
            for k in range(1, len(touching)):
                uf.union(touching[0], touching[k])

        anchor_roots = {uf.find(i) for i in valid_anchor_indices}
        root_to_node_id = {}
        next_node_id = 0
        for r in sorted(anchor_roots):
            root_to_node_id[r] = next_node_id
            next_node_id += 1

        ids = np.full(n, -1, dtype=np.int32)
        for i in range(n):
            r = uf.find(i)
            if r in root_to_node_id:
                ids[i] = root_to_node_id[r]
        return ids

    # First pass: honor every x_jump box and apply all its force-groups.
    all_boxes = [b for (b, _g, _bl) in x_jump_groups_by_box]
    all_groups = [g for (_b, gs, _bl) in x_jump_groups_by_box for g in gs]
    line_node_ids = _finalize(all_boxes, all_groups)

    # Diagonal-crossover sanity fallback: a diag (x_jump) should split into >= 2
    # distinct nodes. If a box's force-merge collapses its crossing lines into
    # fewer than 2 nodes (the slope buckets mis-grouped the wires), drop that box
    # entirely and let the general angle/probe crossover handling stand.
    if x_jump_groups_by_box:
        good_boxes = []
        good_groups = []
        dropped_any = False
        for jbbox, groups, box_lines in x_jump_groups_by_box:
            distinct = {
                int(line_node_ids[k]) for k in box_lines if line_node_ids[k] >= 0
            }
            if len(distinct) >= 2:
                good_boxes.append(jbbox)
                good_groups.extend(groups)
            else:
                dropped_any = True
                print(
                    f"[node] x_jump at {[round(float(v), 1) for v in jbbox]}: "
                    f"force-merge gave {len(distinct)} node(s) (<2); "
                    f"falling back to general crossover handling"
                )
        if dropped_any:
            line_node_ids = _finalize(good_boxes, good_groups)

    return line_node_ids


def _node_color(node_id):
    import cv2
    import numpy as np

    hue = int((node_id * 37) % 180)
    hsv = np.uint8([[[hue, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(v) for v in bgr)


def _draw_float_line(img, line, color, thickness=2):
    import cv2
    import numpy as np

    p1 = tuple(np.round(line[0]).astype(int).tolist())
    p2 = tuple(np.round(line[1]).astype(int).tolist())
    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


def _save_node_visualization(
    im_bin,
    lines,
    comp_bboxes,
    line_node_ids,
    anchor_infos,
    out_path,
    junction_bboxes=None,
    jump_bboxes=None,
    x_jump_bboxes=None,
):
    import cv2
    import numpy as np

    vis = cv2.cvtColor(im_bin, cv2.COLOR_GRAY2BGR)

    # Draw component bboxes.
    for bbox in comp_bboxes:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (180, 180, 180), 1)

    # Draw junction (green) and jump (red) bboxes. BGR.
    if junction_bboxes is not None:
        for bbox in junction_bboxes:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 1)
            cv2.putText(
                vis,
                "J",
                (x1 + 2, y1 + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 200, 0),
                1,
                cv2.LINE_AA,
            )
    if jump_bboxes is not None:
        for bbox in jump_bboxes:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 220), 1)
            cv2.putText(
                vis,
                "X",
                (x1 + 2, y1 + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 220),
                1,
                cv2.LINE_AA,
            )
    # Draw diagonal-crossover (x_jump) bboxes in magenta, labelled "D". BGR.
    if x_jump_bboxes is not None:
        for bbox in x_jump_bboxes:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (220, 0, 220), 1)
            cv2.putText(
                vis,
                "D",
                (x1 + 2, y1 + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (220, 0, 220),
                1,
                cv2.LINE_AA,
            )

    # Draw non-node lines in light gray.
    for i, ln in enumerate(lines):
        if line_node_ids[i] < 0:
            _draw_float_line(vis, ln, (160, 160, 160), thickness=1)

    # Draw node lines.
    node_ids = sorted(set(int(v) for v in line_node_ids if int(v) >= 0))
    for node_id in node_ids:
        color = _node_color(node_id)

        idxs = np.where(line_node_ids == node_id)[0]
        for i in idxs:
            _draw_float_line(vis, lines[i], color, thickness=2)

        # Put node label near the first line.
        if len(idxs) > 0:
            first_ln = lines[idxs[0]]
            p = np.mean(first_ln, axis=0)
            x = int(round(float(p[0])))
            y = int(round(float(p[1])))
            cv2.putText(
                vis,
                f"N{node_id}",
                (x + 3, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )

    # Draw valid anchors and their extension/contact points.
    for info in anchor_infos:
        node_id = int(line_node_ids[int(info["line_idx"])])
        color = _node_color(node_id) if node_id >= 0 else (0, 255, 255)

        extended = np.asarray(info["extended_line"], dtype=np.float32)
        contact = np.asarray(info["contact"], dtype=np.float32)

        _draw_float_line(vis, extended, color, thickness=1)

        cx = int(round(float(contact[0])))
        cy = int(round(float(contact[1])))
        cv2.circle(vis, (cx, cy), 3, (255, 255, 255), -1)
        cv2.circle(vis, (cx, cy), 3, color, 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def _draw_bboxes_to_mask(mask, bboxes, pad=0):
    import cv2
    import numpy as np

    H, W = mask.shape[:2]
    bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)

    for b in bboxes:
        x1, y1, x2, y2 = map(float, b[:4])

        xmin = int(max(0, min(x1, x2) - pad))
        ymin = int(max(0, min(y1, y2) - pad))
        xmax = int(min(W - 1, max(x1, x2) + pad))
        ymax = int(min(H - 1, max(y1, y2) + pad))

        if xmax <= xmin or ymax <= ymin:
            continue

        cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), 255, thickness=-1)

    return mask


def _skeletonize_mask(mask):
    import cv2
    import numpy as np

    mask = (mask > 0).astype(np.uint8) * 255

    # Best option if opencv-contrib-python is installed
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(mask)

    # Fallback morphological skeletonization
    skel = np.zeros(mask.shape, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    work = mask.copy()
    while True:
        eroded = cv2.erode(work, element)
        opened = cv2.dilate(eroded, element)
        temp = cv2.subtract(work, opened)
        skel = cv2.bitwise_or(skel, temp)
        work = eroded.copy()

        if cv2.countNonZero(work) == 0:
            break

    return skel


def _long_hv_line_mask(black_mask, line_len):
    import cv2
    import numpy as np

    L = max(2, int(line_len))
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, L))
    h = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, hk)
    v = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, vk)
    keep = cv2.bitwise_or(h, v)
    # Widen slightly so the full wire thickness survives the carve-out.
    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return keep


def _add_black_pixel_lines(
    lines,
    im_binary,
    comp_bboxes,
    text_bboxes=None,
    min_len=4.0,
    component_pad=1,
    text_pad=1,
    close_kernel_size=3,
    hough_threshold=8,
    hough_max_gap=6,
    keep_line_len=0,
    black_bridge_len=0,
    debug_path=None,
):
    import cv2
    import numpy as np

    lines = np.asarray(lines, dtype=np.float32).reshape(-1, 2, 2)

    if im_binary is None:
        return lines

    H, W = im_binary.shape[:2]

    # Binary convention:
    #   wire / black pixels = 0
    #   background / white pixels = 255
    black_mask = (im_binary < 128).astype(np.uint8) * 255

    # Build ignore mask for component bboxes and OCR text bboxes
    ignore_mask = np.zeros((H, W), dtype=np.uint8)

    if comp_bboxes is not None and len(comp_bboxes) > 0:
        _draw_bboxes_to_mask(ignore_mask, comp_bboxes, pad=component_pad)

    if text_bboxes is not None and len(text_bboxes) > 0:
        text_mask = np.zeros((H, W), dtype=np.uint8)
        _draw_bboxes_to_mask(text_mask, text_bboxes, pad=text_pad)
        if keep_line_len and keep_line_len > 1:
            # Stroke-level text removal: keep long straight wire runs that pass
            # through a text bbox, so a label sitting on a wire erases only the
            # text strokes, not the wire crossing it.
            line_mask = _long_hv_line_mask(black_mask, int(keep_line_len))
            text_mask = cv2.bitwise_and(text_mask, cv2.bitwise_not(line_mask))
        ignore_mask = cv2.bitwise_or(ignore_mask, text_mask)

    allowed_mask = cv2.bitwise_not(ignore_mask)

    # Only look at black pixels outside component/text boxes
    black_outside = cv2.bitwise_and(black_mask, allowed_mask)

    # Optional: connect tiny gaps among black pixels
    if close_kernel_size is not None and close_kernel_size > 1:
        k = int(close_kernel_size)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        black_outside = cv2.morphologyEx(black_outside, cv2.MORPH_CLOSE, kernel)

    # Optional: directional gap bridging. Heals breaks that run ALONG a wire
    # (faint pixels lost by binarization, or a stretch erased with a text bbox)
    # by closing same-row / same-column gaps up to black_bridge_len px. Long
    # thin H/V kernels only bridge collinear pixels, so perpendicular structure
    # and crossings are left intact. Set 0 to disable.
    if black_bridge_len and black_bridge_len > 1:
        L = int(black_bridge_len)
        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))
        vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, L))
        h_bridge = cv2.morphologyEx(black_outside, cv2.MORPH_CLOSE, hk)
        v_bridge = cv2.morphologyEx(black_outside, cv2.MORPH_CLOSE, vk)
        black_outside = cv2.bitwise_or(
            black_outside, cv2.bitwise_or(h_bridge, v_bridge)
        )

    # Optional debug save: this shows exactly where new lines are extracted from
    if debug_path is not None:
        cv2.imwrite(str(debug_path), black_outside)

    # Skeletonize so Hough does not produce too many duplicate thick-line results
    skel = _skeletonize_mask(black_outside)

    hough_lines = cv2.HoughLinesP(
        skel,
        rho=1,
        theta=np.pi / 180,
        threshold=int(hough_threshold),
        minLineLength=max(1, int(round(min_len))),
        maxLineGap=int(hough_max_gap),
    )

    if hough_lines is None:
        return lines

    added = []

    for ln in hough_lines:
        x1, y1, x2, y2 = ln[0]
        length = float(np.hypot(x2 - x1, y2 - y1))

        if length < min_len:
            continue

        added.append(
            [
                [float(x1), float(y1)],
                [float(x2), float(y2)],
            ]
        )

    if not added:
        return lines

    added = np.asarray(added, dtype=np.float32).reshape(-1, 2, 2)

    # Only add. Do not modify old lines.
    return np.concatenate([lines, added], axis=0)


def _save_combined_visualization(
    im, lines, comp_bboxes, text_bboxes, out_path, title=""
):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(im)
    ax.set_axis_off()
    if title:
        ax.set_title(title)

    for b in comp_bboxes:
        x1, y1, x2, y2 = b[:4]
        ax.add_patch(
            plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="gray",
                linewidth=0.7,
                linestyle=":",
                zorder=1,
            )
        )
    for b in text_bboxes:
        x1, y1, x2, y2 = b[:4]
        ax.add_patch(
            plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="goldenrod",
                linewidth=0.7,
                linestyle="--",
                zorder=1,
            )
        )

    for ln in lines:
        ax.plot(
            [ln[0, 0], ln[1, 0]],
            [ln[0, 1], ln[1, 1]],
            color="red",
            linewidth=1.5,
            zorder=3,
        )
        ax.scatter([ln[0, 0], ln[1, 0]], [ln[0, 1], ln[1, 1]], **PLTOPTS)

    fig.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight", dpi=120)
    plt.close(fig)


def _binarize_local_mean(
    im_gray,
    block_size=31,
    C=10,
    blur_ksize=3,
    do_morph=False,
):
    import cv2
    import numpy as np

    if im_gray is None:
        raise ValueError("im_gray is None")

    if im_gray.dtype != np.uint8:
        im_gray = im_gray.astype(np.uint8)

    # block_size must be odd and > 1
    block_size = int(block_size)
    if block_size <= 1:
        block_size = 3
    if block_size % 2 == 0:
        block_size += 1

    # optional blur to suppress small noise before thresholding
    if blur_ksize is not None and blur_ksize > 1:
        blur_ksize = int(blur_ksize)
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        work = cv2.GaussianBlur(im_gray, (blur_ksize, blur_ksize), 0)
    else:
        work = im_gray

    binary = cv2.adaptiveThreshold(
        work,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block_size,
        C,
    )

    # Optional: clean tiny noise / connect tiny gaps.
    # I would keep this False at first, because morphology may also thicken text strokes.
    if do_morph:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary


def run_combined_stage(device, args):
    import cv2

    if not HAWP_WEIGHTS.exists():
        print(f"[combined] HAWP weights missing at {HAWP_WEIGHTS} - skipping")
        return

    # Always ensure the OCR text-bbox cache is populated. run_ocr_text_removal
    # is idempotent: it short-circuits when all images already have cached
    # results, so this is essentially free on re-runs.
    print("[combined] ensuring OCR text-bbox cache is populated...")
    run_ocr_text_removal(
        device,
        pad=args.ocr_pad,
        conf_thresh=args.ocr_conf,
        force=args.force_ocr,
        stems=getattr(args, "stem", None),
    )

    print(f"[combined] HAWP threshold = {args.combined_hawp_threshold}")
    print(
        f"[combined] line-to-text clearance = {args.text_distance}px  "
        f"min-line-length = {args.min_line_length}px"
    )

    model = build_hawp(device)
    images = sorted(
        [
            p
            for p in MASKED_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ]
    )
    images = _filter_by_stems(images, getattr(args, "stem", None))
    print(f"[combined] processing {len(images)} image(s)")

    COMBINED_LINES_DIR.mkdir(parents=True, exist_ok=True)
    COMBINED_VIS_DIR.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        im_gray_raw = cv2.imread(str(img_path), 0)
        if im_gray_raw is None:
            print(f"[combined] {img_path.name}: unreadable, skipping")
            continue

        # 0. Binary preprocessing on the whole masked image
        im_gray = _binarize_local_mean(im_gray_raw)

        # 1. HAWP
        hawp_lines, hawp_scores = run_hawp_on_image(model, im_gray, device)
        keep_score = hawp_scores >= args.combined_hawp_threshold
        lines = hawp_lines[keep_score]
        n_hawp = len(lines)

        # 2. Clip against component bboxes
        comp_bboxes = _read_bbox_meta(img_path.stem)[:, :4]
        clipped = _clip_lines_outside_bboxes(
            lines,
            comp_bboxes,
            min_len=args.min_line_length,
        )
        n_clip = len(clipped)

        # 3. Read OCR text bboxes
        text_bboxes = _read_text_bboxes(img_path.stem)

        # 4. Cut / handle white-background portions
        whitespace_kept = _drop_lines_on_whitespace(
            clipped,
            im_gray,
            white_threshold=args.whitespace_threshold,
            fraction=args.whitespace_fraction,
            min_len=args.min_line_length,
        )
        n_ws = len(whitespace_kept)

        # 5. Upgrade line result using remaining black pixels outside component/text bboxes
        final = _add_black_pixel_lines(
            whitespace_kept,
            im_gray,
            comp_bboxes=comp_bboxes,
            text_bboxes=text_bboxes,
            min_len=args.min_line_length,
            component_pad=1,
            text_pad=args.text_distance,
            close_kernel_size=3,
            hough_threshold=8,
            hough_max_gap=6,
            keep_line_len=getattr(args, "text_keep_line_len", 12.0),
            black_bridge_len=getattr(args, "black_bridge_len", 0),
            debug_path=COMBINED_VIS_DIR
            / f"{img_path.stem}_black_pixels_for_extra_lines.png",
        )

        n_final = len(final)
        n_added = n_final - n_ws

        print(
            f"[combined] {img_path.name}: "
            f"HAWP={n_hawp} -> clip={n_clip} -> whitespace={n_ws} "
            f"-> added={n_added} -> final={n_final}"
        )

        # Save lines
        np.savez(
            COMBINED_LINES_DIR / f"{img_path.stem}.npz",
            lines=final.astype(np.float32),
        )

        with open(
            COMBINED_LINES_DIR / f"{img_path.stem}.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write("# x1 y1 x2 y2\n")
            for ln in final:
                f.write(
                    f"{ln[0,0]:.2f} {ln[0,1]:.2f} " f"{ln[1,0]:.2f} {ln[1,1]:.2f}\n"
                )

        # Visualize
        im_rgb = cv2.cvtColor(im_gray, cv2.COLOR_GRAY2RGB)
        _save_combined_visualization(
            im_rgb,
            final,
            comp_bboxes,
            text_bboxes,
            COMBINED_VIS_DIR / f"{img_path.stem}.png",
            title=f"{img_path.name}  HAWP->clip->whitespace->pixels",
        )
    # if getattr(args, "generate_node", True):
    #     run_component_anchor_nodes(args)


def run_component_anchor_nodes(args):
    import json
    import cv2
    import numpy as np

    NODE_DIR.mkdir(parents=True, exist_ok=True)
    NODE_VIS_DIR.mkdir(parents=True, exist_ok=True)
    NODE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    line_files = sorted(COMBINED_LINES_DIR.glob("*.npz"))
    line_files = _filter_by_stems(line_files, getattr(args, "stem", None))
    print(f"[node] processing {len(line_files)} saved final-line file(s)")

    max_anchor_dist = getattr(args, "max_anchor_dist", None)
    max_extend_px = getattr(args, "max_extend_px", 12.0)
    extra_into_bbox_px = getattr(args, "extra_into_bbox_px", 2.0)
    node_union_dist = getattr(args, "node_union_dist", 8.0)
    node_pixel_bridge_dist = getattr(args, "node_pixel_bridge_dist", 0.0)
    node_pixel_bridge_fill = getattr(args, "node_pixel_bridge_fill", 0.6)
    node_pixel_bridge_radius = getattr(args, "node_pixel_bridge_radius", 1)
    black_threshold = getattr(args, "black_threshold", 128)
    inside_depth = getattr(args, "inside_depth", 4)
    patch_radius = getattr(args, "patch_radius", 1)
    perp_tol_deg = getattr(args, "perp_tol", 10.0)
    jj_conf = getattr(args, "jj_conf", 0.25)
    jj_bbox_pad = getattr(args, "jj_bbox_pad", 2.0)
    crossover_probe_dist = getattr(args, "crossover_probe_dist", 5.0)
    crossover_probe_tol = getattr(args, "crossover_probe_tol", 2.0)
    crossover_min_line_len = getattr(args, "crossover_min_line_len", 3.0)
    implicit_jump_min_extend_px = getattr(args, "implicit_jump_min_extend_px", 3.0)
    explicit_jump_min_extend_px = getattr(args, "explicit_jump_min_extend_px", 1.0)
    jump_edge_component_adjacent_px = getattr(args, "jump_edge_component_adjacent_px", 6.0)
    jump_parallel_tol_deg = getattr(args, "jump_parallel_tol", 20.0)
    jj_trust_yolo = bool(getattr(args, "jj_trust_yolo", False))
    jj_device = getattr(args, "device", None)

    for line_path in line_files:
        stem = line_path.stem

        data = np.load(line_path)
        lines = data["lines"].astype(np.float32)

        # Find original image.
        img_path = None
        for p in TEST_IMAGES.iterdir():
            if p.is_file() and p.stem == stem and p.suffix.lower() in IMG_EXTS:
                img_path = p
                break

        if img_path is None:
            print(f"[node] {stem}: original image not found, skipping")
            continue

        im_gray_raw = cv2.imread(str(img_path), 0)
        if im_gray_raw is None:
            print(f"[node] {stem}: unreadable image, skipping")
            continue

        # Binary preprocessing for black-pixel contact checking.
        im_bin = _binarize_local_mean(im_gray_raw)

        bbox_meta = _read_bbox_meta(stem)
        bbox_meta = np.asarray(bbox_meta, dtype=np.float32)

        if bbox_meta.size == 0:
            comp_bboxes = np.empty((0, 4), dtype=np.float32)
        else:
            comp_bboxes = bbox_meta[:, :4].astype(np.float32)

        # Junction/jump detection always runs on the text-intact
        # MASKED_DIR/<stem>, never on the text-removed copy.
        jj_img_path = None
        for ext in IMG_EXTS:
            cand = MASKED_DIR / f"{stem}{ext}"
            if cand.exists():
                jj_img_path = cand
                break
        empty4 = np.empty((0, 4), dtype=np.float32)
        if jj_img_path is None:
            print(
                f"[node] {stem}: masked image not found in "
                f"{MASKED_DIR}; junction/jump detection skipped"
            )
            raw_junction = raw_jump = raw_implicit = empty4
            raw_diag_jump = raw_diag_implicit = empty4
        else:
            (
                raw_junction,
                raw_jump,
                raw_implicit,
                raw_diag_jump,
                raw_diag_implicit,
            ) = _detect_junctions_and_jumps(
                jj_img_path,
                conf=jj_conf,
                device=jj_device,
                trust_yolo=jj_trust_yolo,
            )

        # Resolve the 5 raw YOLO classes into junction / jump / x_jump
        # (diagonal) categories, applying the implicit-jump routing and the
        # implicit-only two-pairs validation. diag_jump from YOLO lands in
        # x_jump_bboxes here.
        junction_bboxes, jump_bboxes, x_jump_bboxes = _route_jj_detections(
            lines,
            raw_junction,
            raw_jump,
            raw_implicit,
            raw_diag_jump,
            raw_diag_implicit,
            implicit_min_extend_px=implicit_jump_min_extend_px,
            explicit_min_extend_px=explicit_jump_min_extend_px,
            comp_bboxes=comp_bboxes,
            comp_adjacent_px=jump_edge_component_adjacent_px,
            trust_yolo=jj_trust_yolo,
        )

        # Phase 2c hook: union YOLO output with manual additions from the
        # GUI's override file, if one exists for this stem. Manual added_x_jumps
        # are merged with the YOLO-derived diagonal crossovers above.
        junction_bboxes, jump_bboxes, manual_x_jump_bboxes = _apply_manual_jj_overrides(
            stem, junction_bboxes, jump_bboxes
        )
        if len(manual_x_jump_bboxes):
            x_jump_bboxes = (
                manual_x_jump_bboxes
                if not len(x_jump_bboxes)
                else np.concatenate([x_jump_bboxes, manual_x_jump_bboxes], axis=0)
            )

        valid_anchor_indices, anchor_infos = _find_valid_component_anchor_lines(
            lines=lines,
            im_bin=im_bin,
            comp_bboxes=comp_bboxes,
            max_anchor_dist=max_anchor_dist,
            max_extend_px=max_extend_px,
            extra_into_bbox_px=extra_into_bbox_px,
            black_threshold=black_threshold,
            inside_depth=inside_depth,
            patch_radius=patch_radius,
        )

        line_node_ids = _group_lines_into_nodes_by_anchor_union_find(
            lines=lines,
            valid_anchor_indices=valid_anchor_indices,
            node_union_dist=node_union_dist,
            junction_bboxes=junction_bboxes,
            jump_bboxes=jump_bboxes,
            perp_tol_deg=perp_tol_deg,
            bbox_pad=jj_bbox_pad,
            crossover_probe_dist=crossover_probe_dist,
            crossover_probe_tol=crossover_probe_tol,
            crossover_min_line_len=crossover_min_line_len,
            jump_parallel_tol_deg=jump_parallel_tol_deg,
            x_jump_bboxes=x_jump_bboxes,
            im_bin=im_bin,
            comp_bboxes=comp_bboxes,
            pixel_bridge_dist=node_pixel_bridge_dist,
            pixel_bridge_fill_frac=node_pixel_bridge_fill,
            pixel_bridge_radius=node_pixel_bridge_radius,
            black_threshold=black_threshold,
        )

        n_nodes = len(set(int(v) for v in line_node_ids if int(v) >= 0))
        n_node_lines = int(np.sum(line_node_ids >= 0))

        case_label = "case2-jump" if len(jump_bboxes) > 0 else "case1-junction"
        print(
            f"[node] {stem}: "
            f"lines={len(lines)} anchors={len(valid_anchor_indices)} "
            f"junctions={len(junction_bboxes)} jumps={len(jump_bboxes)} "
            f"[{case_label}] "
            f"node_lines={n_node_lines} nodes={n_nodes}"
        )

        # Save data.
        extended_anchor_lines = []
        contact_points = []
        anchor_line_indices = []
        anchor_bbox_indices = []
        anchor_edge_names = []

        for info in anchor_infos:
            extended_anchor_lines.append(info["extended_line"])
            contact_points.append(info["contact"])
            anchor_line_indices.append(info["line_idx"])
            anchor_bbox_indices.append(info["bbox_idx"])
            anchor_edge_names.append(info["edge_name"])

        if len(extended_anchor_lines) == 0:
            extended_anchor_lines = np.empty((0, 2, 2), dtype=np.float32)
            contact_points = np.empty((0, 2), dtype=np.float32)
        else:
            extended_anchor_lines = np.asarray(extended_anchor_lines, dtype=np.float32)
            contact_points = np.asarray(contact_points, dtype=np.float32)

        np.savez(
            NODE_DATA_DIR / f"{stem}.npz",
            lines=lines.astype(np.float32),
            line_node_ids=line_node_ids.astype(np.int32),
            valid_anchor_indices=np.asarray(valid_anchor_indices, dtype=np.int32),
            extended_anchor_lines=extended_anchor_lines.astype(np.float32),
            contact_points=contact_points.astype(np.float32),
            anchor_line_indices=np.asarray(anchor_line_indices, dtype=np.int32),
            anchor_bbox_indices=np.asarray(anchor_bbox_indices, dtype=np.int32),
            junction_bboxes=junction_bboxes.astype(np.float32),
            jump_bboxes=jump_bboxes.astype(np.float32),
            x_jump_bboxes=x_jump_bboxes.astype(np.float32),
        )

        with open(NODE_DATA_DIR / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "image": stem,
                    "num_lines": int(len(lines)),
                    "num_valid_anchors": int(len(valid_anchor_indices)),
                    "num_node_lines": int(n_node_lines),
                    "num_nodes": int(n_nodes),
                    "case": case_label,
                    "num_junctions": int(len(junction_bboxes)),
                    "num_jumps": int(len(jump_bboxes)),
                    "num_x_jumps": int(len(x_jump_bboxes)),
                    "junction_bboxes": junction_bboxes.astype(float).tolist(),
                    "jump_bboxes": jump_bboxes.astype(float).tolist(),
                    "x_jump_bboxes": x_jump_bboxes.astype(float).tolist(),
                    "params": {
                        "max_anchor_dist": max_anchor_dist,
                        "max_extend_px": max_extend_px,
                        "extra_into_bbox_px": extra_into_bbox_px,
                        "node_union_dist": node_union_dist,
                        "black_threshold": black_threshold,
                        "inside_depth": inside_depth,
                        "patch_radius": patch_radius,
                        "perp_tol_deg": perp_tol_deg,
                        "jj_conf": jj_conf,
                        "jj_bbox_pad": jj_bbox_pad,
                        "crossover_probe_dist": crossover_probe_dist,
                        "crossover_probe_tol": crossover_probe_tol,
                        "crossover_min_line_len": crossover_min_line_len,
                        "implicit_jump_min_extend_px": implicit_jump_min_extend_px,
                        "explicit_jump_min_extend_px": explicit_jump_min_extend_px,
                        "jump_parallel_tol_deg": jump_parallel_tol_deg,
                    },
                    "anchors": anchor_infos,
                    "line_node_ids": line_node_ids.astype(int).tolist(),
                },
                f,
                indent=2,
            )

        _save_node_visualization(
            im_bin=im_bin,
            lines=lines,
            comp_bboxes=comp_bboxes,
            line_node_ids=line_node_ids,
            anchor_infos=anchor_infos,
            out_path=NODE_VIS_DIR / f"{stem}.png",
            junction_bboxes=junction_bboxes,
            jump_bboxes=jump_bboxes,
            x_jump_bboxes=x_jump_bboxes,
        )


