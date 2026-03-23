from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_PROMPT_TEMPLATE = """<image>
Please construct the road map from ({start_x},{start_y}) to ({end_x},{end_y}) in the satellite image.
Only predict road segments inside the target box [{box_x_min},{box_y_min},{box_x_max},{box_y_max}].
Keep all coordinates in the patch-local coordinate system."""

DEFAULT_STATE_PROMPT_TEMPLATE = """<image>
Please construct the road map from ({start_x},{start_y}) to ({end_x},{end_y}) in the satellite image.
Only predict road segments inside the target box [{box_x_min},{box_y_min},{box_x_max},{box_y_max}].
Keep all coordinates in the patch-local coordinate system.
Previous state:
{state_json}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fixed-grid target-box dataset from an existing full-patch Stage A or Stage B dataset."
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Existing full-patch dataset root, e.g. stage_a/dataset or stage_b/dataset.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output dataset root.")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"], help="Splits to process.")
    parser.add_argument("--grid-size", type=int, default=4, help="Grid size per side. 4 means 16 fixed target boxes.")
    parser.add_argument("--target-empty-ratio", type=float, default=0.10, help="Maximum empty-sample ratio after filtering.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for empty-sample downsampling.")
    parser.add_argument("--max-source-samples-per-split", type=int, default=0, help="Optional cap for smoke tests.")
    parser.add_argument("--resample-step-px", type=float, default=12.0, help="Resample clipped target lines. Use <=0 to keep original point spacing.")
    parser.add_argument("--boundary-tol-px", type=float, default=2.5, help="Boundary tolerance for cut endpoint detection.")
    parser.add_argument("--use-system-prompt-from-source", action="store_true", help="Reuse the source system prompt if present.")
    parser.add_argument("--image-root-mode", type=str, default="symlink", choices=["symlink", "copy", "none"], help="How to expose images under the output root.")
    parser.add_argument("--export-visualizations", action="store_true", help="Export QA visualizations.")
    parser.add_argument("--max-visualizations-per-split", type=int, default=0, help="Optional cap for visualization count per split.")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    count = 0
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def dedup_points(points: Sequence[np.ndarray], eps: float = 1e-3) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    out = [arr[0]]
    for idx in range(1, arr.shape[0]):
        if float(np.linalg.norm(arr[idx] - out[-1])) > float(eps):
            out.append(arr[idx])
    return np.asarray(out, dtype=np.float32)


def clamp_points(points_xy: np.ndarray, patch_size: int) -> np.ndarray:
    arr = np.asarray(points_xy, dtype=np.float32).copy()
    if arr.ndim != 2 or arr.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    arr[:, 0] = np.clip(arr[:, 0], 0.0, float(patch_size - 1))
    arr[:, 1] = np.clip(arr[:, 1], 0.0, float(patch_size - 1))
    return arr


def simplify_for_json(points_xy: np.ndarray, patch_size: int) -> List[List[int]]:
    arr = clamp_points(points_xy, patch_size=patch_size)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return []
    arr = np.rint(arr).astype(np.int32)
    arr = dedup_points(arr.astype(np.float32)).astype(np.int32)
    return [[int(x), int(y)] for x, y in arr.tolist()]


def resample_polyline(points_xy: np.ndarray, step_px: float, max_points: Optional[int] = None) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return pts
    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    total = float(np.sum(seg))
    if total < 1e-6:
        return pts[:1]
    step = max(float(step_px), 1.0)
    n = max(2, int(math.floor(total / step)) + 1)
    if max_points is not None:
        n = min(n, int(max_points))
    target = np.linspace(0.0, total, n, dtype=np.float32)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    sampled: List[np.ndarray] = []
    for dist in target:
        j = int(np.searchsorted(cum, dist, side="right") - 1)
        j = min(max(j, 0), len(seg) - 1)
        t0 = float(cum[j])
        t1 = float(cum[j + 1])
        ratio = 0.0 if t1 <= t0 else (float(dist) - t0) / (t1 - t0)
        sampled.append(pts[j] * (1.0 - ratio) + pts[j + 1] * ratio)
    return dedup_points(sampled)


def point_in_rect(point_xy: np.ndarray, rect: Tuple[float, float, float, float], eps: float = 1e-6) -> bool:
    x_min, y_min, x_max, y_max = rect
    x = float(point_xy[0])
    y = float(point_xy[1])
    return (x_min - eps) <= x <= (x_max + eps) and (y_min - eps) <= y <= (y_max + eps)


def clip_segment_liang_barsky(
    p0: np.ndarray,
    p1: np.ndarray,
    rect: Tuple[float, float, float, float],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    x_min, y_min, x_max, y_max = rect
    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])
    p = [-dx, dx, -dy, dy]
    q = [float(p0[0] - x_min), float(x_max - p0[0]), float(p0[1] - y_min), float(y_max - p0[1])]
    u1 = 0.0
    u2 = 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-8:
            if qi < 0.0:
                return None
            continue
        t = qi / pi
        if pi < 0.0:
            if t > u2:
                return None
            if t > u1:
                u1 = t
        else:
            if t < u1:
                return None
            if t < u2:
                u2 = t
    c0 = np.asarray([p0[0] + u1 * dx, p0[1] + u1 * dy], dtype=np.float32)
    c1 = np.asarray([p0[0] + u2 * dx, p0[1] + u2 * dy], dtype=np.float32)
    return c0, c1


def clip_polyline_to_rect(points_xy: np.ndarray, rect: Tuple[float, float, float, float]) -> List[np.ndarray]:
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return []
    pieces: List[np.ndarray] = []
    current: List[np.ndarray] = []
    for idx in range(pts.shape[0] - 1):
        clipped = clip_segment_liang_barsky(pts[idx], pts[idx + 1], rect)
        if clipped is None:
            if len(current) >= 2:
                pieces.append(dedup_points(current))
            current = []
            continue
        c0, c1 = clipped
        if not current:
            current = [c0, c1]
        else:
            if float(np.linalg.norm(current[-1] - c0)) <= 1e-3:
                current.append(c1)
            else:
                if len(current) >= 2:
                    pieces.append(dedup_points(current))
                current = [c0, c1]
        if not point_in_rect(pts[idx + 1], rect):
            if len(current) >= 2:
                pieces.append(dedup_points(current))
            current = []
    if len(current) >= 2:
        pieces.append(dedup_points(current))
    return [piece for piece in pieces if piece.shape[0] >= 2]


def point_boundary_side(point_xy: np.ndarray, rect: Tuple[float, float, float, float], tol_px: float) -> Optional[str]:
    x_min, y_min, x_max, y_max = rect
    x = float(point_xy[0])
    y = float(point_xy[1])
    if abs(x - x_min) <= tol_px:
        return "left"
    if abs(y - y_min) <= tol_px:
        return "top"
    if abs(x - x_max) <= tol_px:
        return "right"
    if abs(y - y_max) <= tol_px:
        return "bottom"
    return None


def point_origin_sort_key(point_xy: Sequence[float]) -> Tuple[float, float, float]:
    x = float(point_xy[0])
    y = float(point_xy[1])
    return (x * x + y * y, y, x)


def canonicalize_line_direction(
    points_xy: np.ndarray,
    start_type: str,
    end_type: str,
) -> Tuple[np.ndarray, str, str]:
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return pts, start_type, end_type
    start_is_cut = str(start_type) == "cut"
    end_is_cut = str(end_type) == "cut"
    reverse = False
    if start_is_cut and not end_is_cut:
        reverse = False
    elif end_is_cut and not start_is_cut:
        reverse = True
    elif point_origin_sort_key(pts[-1]) < point_origin_sort_key(pts[0]):
        reverse = True
    if not reverse:
        return pts, start_type, end_type
    return pts[::-1].copy(), end_type, start_type


def sort_lines(lines: List[Dict]) -> List[Dict]:
    return sorted(
        lines,
        key=lambda item: (*point_origin_sort_key(item.get("points", [[1e9, 1e9]])[0]),),
    )


def line_length_xy(points_xy: Sequence[Sequence[float]]) -> float:
    arr = np.asarray(points_xy, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(arr[1:] - arr[:-1], axis=1).sum())


def extract_message_content(row: Dict, role: str) -> str:
    want = str(role).strip().lower()
    for msg in row.get("messages", []):
        if str(msg.get("role", "")).strip().lower() == want:
            return str(msg.get("content", ""))
    return ""


def build_grid_boxes(patch_size: int, grid_size: int) -> List[Dict[str, int]]:
    edges = [int(round(i * patch_size / grid_size)) for i in range(grid_size + 1)]
    boxes: List[Dict[str, int]] = []
    for grid_row in range(grid_size):
        for grid_col in range(grid_size):
            x_min = edges[grid_col]
            x_max = edges[grid_col + 1] - 1
            y_min = edges[grid_row]
            y_max = edges[grid_row + 1] - 1
            boxes.append(
                {
                    "grid_row": int(grid_row),
                    "grid_col": int(grid_col),
                    "x_min": int(x_min),
                    "y_min": int(y_min),
                    "x_max": int(x_max),
                    "y_max": int(y_max),
                }
            )
    return boxes


def longest_piece_in_box(lines: Sequence[Dict], rect: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
    best_piece: Optional[np.ndarray] = None
    best_len = -1.0
    for line in lines:
        pts = np.asarray(line.get("points", []), dtype=np.float32)
        pieces = clip_polyline_to_rect(pts, rect)
        for piece in pieces:
            ln = line_length_xy(piece.tolist())
            if ln > best_len:
                best_len = ln
                best_piece = piece
    return best_piece


def build_prompt_endpoints(target_lines: Sequence[Dict], target_box: Dict[str, int], patch_size: int) -> Dict[str, object]:
    rect = (
        float(target_box["x_min"]),
        float(target_box["y_min"]),
        float(target_box["x_max"]),
        float(target_box["y_max"]),
    )
    piece = longest_piece_in_box(target_lines, rect)
    if piece is None or piece.shape[0] < 2:
        center_x = int(round((int(target_box["x_min"]) + int(target_box["x_max"])) / 2.0))
        center_y = int(round((int(target_box["y_min"]) + int(target_box["y_max"])) / 2.0))
        return {
            "start_x": center_x,
            "start_y": center_y,
            "end_x": center_x,
            "end_y": center_y,
            "anchor_piece_points": [[center_x, center_y]],
            "anchor_source": "box_center",
        }
    piece = clamp_points(piece, patch_size=patch_size)
    piece_json = simplify_for_json(piece, patch_size=patch_size)
    if len(piece_json) < 2:
        center_x = int(round((int(target_box["x_min"]) + int(target_box["x_max"])) / 2.0))
        center_y = int(round((int(target_box["y_min"]) + int(target_box["y_max"])) / 2.0))
        return {
            "start_x": center_x,
            "start_y": center_y,
            "end_x": center_x,
            "end_y": center_y,
            "anchor_piece_points": [[center_x, center_y]],
            "anchor_source": "box_center",
        }
    return {
        "start_x": int(piece_json[0][0]),
        "start_y": int(piece_json[0][1]),
        "end_x": int(piece_json[-1][0]),
        "end_y": int(piece_json[-1][1]),
        "anchor_piece_points": piece_json,
        "anchor_source": "longest_clipped_piece",
    }


def build_target_lines_for_box(
    full_patch_target_lines: Sequence[Dict],
    target_box: Dict[str, int],
    patch_size: int,
    boundary_tol_px: float,
    resample_step_px: float,
) -> List[Dict]:
    local_rect = (
        float(target_box["x_min"]),
        float(target_box["y_min"]),
        float(target_box["x_max"]),
        float(target_box["y_max"]),
    )
    out: List[Dict] = []
    for segment in full_patch_target_lines:
        pts = np.asarray(segment.get("points", []), dtype=np.float32)
        pieces = clip_polyline_to_rect(pts, local_rect)
        for piece in pieces:
            piece = clamp_points(piece, patch_size=patch_size)
            if resample_step_px > 0:
                piece = resample_polyline(piece, step_px=resample_step_px)
            if piece.ndim != 2 or piece.shape[0] < 2:
                continue
            start_side = point_boundary_side(piece[0], local_rect, boundary_tol_px)
            end_side = point_boundary_side(piece[-1], local_rect, boundary_tol_px)
            start_type = "cut" if start_side is not None else "start"
            end_type = "cut" if end_side is not None else "end"
            piece, start_type, end_type = canonicalize_line_direction(piece, start_type=start_type, end_type=end_type)
            points_json = simplify_for_json(piece, patch_size=patch_size)
            if len(points_json) < 2:
                continue
            out.append(
                {
                    "category": str(segment.get("category", "road")),
                    "start_type": str(start_type),
                    "end_type": str(end_type),
                    "points": points_json,
                }
            )
    return sort_lines(out)


def format_prompt_text(prompt_fields: Dict[str, int], state_json: Optional[str] = None) -> str:
    if state_json is None:
        return DEFAULT_PROMPT_TEMPLATE.format(**prompt_fields)
    fields = dict(prompt_fields)
    fields["state_json"] = str(state_json)
    return DEFAULT_STATE_PROMPT_TEMPLATE.format(**fields)


def make_record(sample_id: str, image_rel_path: str, prompt_text: str, target_lines: Sequence[Dict], system_prompt: str) -> Dict:
    target_json = json.dumps({"lines": list(target_lines)}, ensure_ascii=False, separators=(",", ":"))
    messages: List[Dict] = []
    if str(system_prompt).strip():
        messages.append({"role": "system", "content": str(system_prompt).strip()})
    messages.append({"role": "user", "content": str(prompt_text)})
    messages.append({"role": "assistant", "content": target_json})
    return {
        "id": sample_id,
        "messages": messages,
        "images": [image_rel_path],
    }


def make_state_record(
    sample_id: str,
    image_rel_path: str,
    prompt_text: str,
    target_lines: Sequence[Dict],
    state_lines: Sequence[Dict],
    system_prompt: str,
) -> Dict:
    state_json = json.dumps({"lines": list(state_lines)}, ensure_ascii=False, separators=(",", ":"))
    target_json = json.dumps({"lines": list(target_lines)}, ensure_ascii=False, separators=(",", ":"))
    messages: List[Dict] = []
    if str(system_prompt).strip():
        messages.append({"role": "system", "content": str(system_prompt).strip()})
    messages.append({"role": "user", "content": str(prompt_text).format(state_json=state_json) if "{state_json}" in str(prompt_text) else str(prompt_text)})
    messages.append({"role": "assistant", "content": target_json})
    return {
        "id": sample_id,
        "messages": messages,
        "images": [image_rel_path],
    }


def sanitize_name(name: str) -> str:
    out = []
    for ch in str(name):
        out.append(ch if ch.isalnum() or ch in ("_", "-") else "_")
    return "".join(out).strip("_") or "dataset"


def link_or_copy_images(input_root: Path, output_root: Path, mode: str) -> str:
    src = input_root / "images"
    dst = output_root / "images"
    if not src.exists() or str(mode) == "none":
        return "none"
    if dst.exists() or dst.is_symlink():
        return "existing"
    if str(mode) == "symlink":
        try:
            dst.symlink_to(src, target_is_directory=True)
            return "symlink"
        except OSError:
            shutil.copytree(src, dst)
            return "copy_fallback"
    shutil.copytree(src, dst)
    return "copy"


def filter_pairs_to_empty_ratio(
    pairs: Sequence[Tuple[Dict, Dict]],
    target_empty_ratio: float,
    rng: random.Random,
) -> Tuple[List[Tuple[Dict, Dict]], Dict[str, float]]:
    non_empty_pairs = [pair for pair in pairs if int(pair[1].get("num_target_lines", 0)) > 0]
    empty_pairs = [pair for pair in pairs if int(pair[1].get("num_target_lines", 0)) <= 0]

    if not non_empty_pairs:
        raise ValueError("Split contains no non-empty samples; cannot enforce empty-ratio target.")

    if float(target_empty_ratio) >= 1.0:
        kept_pairs = list(pairs)
        kept_empty = len(empty_pairs)
        kept_total = len(kept_pairs)
        return kept_pairs, {
            "generated_total": len(pairs),
            "generated_non_empty": len(non_empty_pairs),
            "generated_empty": len(empty_pairs),
            "kept_total": kept_total,
            "kept_non_empty": kept_total - kept_empty,
            "kept_empty": kept_empty,
            "kept_empty_ratio": (kept_empty / kept_total if kept_total else 0.0),
        }

    max_empty = math.floor(len(non_empty_pairs) * target_empty_ratio / (1.0 - target_empty_ratio))
    keep_empty = min(len(empty_pairs), max_empty)
    kept_empty_ids = set()
    if keep_empty > 0:
        chosen = rng.sample(empty_pairs, keep_empty) if keep_empty < len(empty_pairs) else list(empty_pairs)
        kept_empty_ids = {str(meta["id"]) for _, meta in chosen}

    kept_pairs: List[Tuple[Dict, Dict]] = []
    for row, meta in pairs:
        if int(meta.get("num_target_lines", 0)) > 0 or str(meta["id"]) in kept_empty_ids:
            kept_pairs.append((row, meta))

    kept_empty = sum(1 for _, meta in kept_pairs if int(meta.get("num_target_lines", 0)) <= 0)
    kept_total = len(kept_pairs)
    return kept_pairs, {
        "generated_total": len(pairs),
        "generated_non_empty": len(non_empty_pairs),
        "generated_empty": len(empty_pairs),
        "kept_total": kept_total,
        "kept_non_empty": kept_total - kept_empty,
        "kept_empty": kept_empty,
        "kept_empty_ratio": (kept_empty / kept_total if kept_total else 0.0),
    }


def draw_endpoint(draw: ImageDraw.ImageDraw, point: Sequence[int], color: Tuple[int, int, int], radius: int = 3) -> None:
    x = int(point[0])
    y = int(point[1])
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def save_visualization(
    patch_image: Image.Image,
    target_lines: Sequence[Dict],
    target_box: Dict[str, int],
    anchor_piece_points: Sequence[Sequence[int]],
    out_path: Path,
) -> None:
    ensure_dir(out_path.parent)
    image = patch_image.convert("RGB")
    draw = ImageDraw.Draw(image)
    patch_size = int(image.size[0])
    draw.rectangle((0, 0, patch_size - 1, patch_size - 1), outline=(255, 0, 180), width=2)
    draw.rectangle(
        (
            int(target_box["x_min"]),
            int(target_box["y_min"]),
            int(target_box["x_max"]),
            int(target_box["y_max"]),
        ),
        outline=(255, 210, 0),
        width=3,
    )
    anchor_pts = [tuple(int(v) for v in p) for p in anchor_piece_points]
    if len(anchor_pts) >= 2:
        draw.line(anchor_pts, fill=(255, 140, 0), width=4)
        draw_endpoint(draw, anchor_pts[0], (0, 255, 80), 4)
        draw_endpoint(draw, anchor_pts[-1], (255, 40, 40), 4)
    elif len(anchor_pts) == 1:
        draw_endpoint(draw, anchor_pts[0], (255, 140, 0), 4)
    for line in target_lines:
        pts = [tuple(int(v) for v in p) for p in line.get("points", [])]
        if len(pts) >= 2:
            draw.line(pts, fill=(40, 220, 255), width=3)
            draw_endpoint(draw, pts[0], (0, 180, 220), 3)
            draw_endpoint(draw, pts[-1], (0, 180, 220), 3)
    image.save(out_path)


def build_split(
    split: str,
    input_root: Path,
    output_root: Path,
    grid_size: int,
    target_empty_ratio: float,
    rng: random.Random,
    max_source_samples_per_split: int,
    boundary_tol_px: float,
    resample_step_px: float,
    reuse_system_prompt: bool,
    export_visualizations: bool,
    max_visualizations_per_split: int,
) -> Dict[str, object]:
    split_jsonl = input_root / f"{split}.jsonl"
    split_meta_jsonl = input_root / f"meta_{split}.jsonl"
    if not split_jsonl.exists() or not split_meta_jsonl.exists():
        return {
            "missing_split": True,
            "split_jsonl": str(split_jsonl),
            "split_meta_jsonl": str(split_meta_jsonl),
        }

    rows = read_jsonl(split_jsonl)
    meta_rows = read_jsonl(split_meta_jsonl)
    row_by_id = {str(row.get("id")): row for row in rows}

    generated_pairs: List[Tuple[Dict, Dict]] = []
    viz_count = 0
    used_source = 0

    for src_meta in meta_rows:
        source_id = str(src_meta.get("id"))
        src_row = row_by_id.get(source_id)
        if src_row is None:
            continue
        used_source += 1
        if max_source_samples_per_split > 0 and used_source > max_source_samples_per_split:
            break

        crop_box = src_meta.get("crop_box", {})
        patch_size = int(crop_box.get("x_max", 0)) - int(crop_box.get("x_min", 0))
        if patch_size <= 1:
            continue
        target_lines_full = list(src_meta.get("target_lines", []))
        source_has_state = ("state_lines" in src_meta) or ("state_mode" in src_meta)
        state_lines_full = list(src_meta.get("state_lines", [])) if source_has_state else []
        image_rel_path = str(src_meta.get("image") or src_row.get("images", [""])[0])
        system_prompt = extract_message_content(src_row, "system") if reuse_system_prompt else ""

        boxes = build_grid_boxes(patch_size=patch_size, grid_size=grid_size)
        patch_image: Optional[Image.Image] = None
        if export_visualizations:
            patch_image = Image.open(input_root / image_rel_path).convert("RGB")

        for box in boxes:
            prompt_info = build_prompt_endpoints(target_lines=target_lines_full, target_box=box, patch_size=patch_size)
            target_lines = build_target_lines_for_box(
                full_patch_target_lines=target_lines_full,
                target_box=box,
                patch_size=patch_size,
                boundary_tol_px=boundary_tol_px,
                resample_step_px=resample_step_px,
            )
            sample_id = f"{source_id}_g{int(box['grid_row'])}{int(box['grid_col'])}"
            prompt_fields = {
                "start_x": int(prompt_info["start_x"]),
                "start_y": int(prompt_info["start_y"]),
                "end_x": int(prompt_info["end_x"]),
                "end_y": int(prompt_info["end_y"]),
                "box_x_min": int(box["x_min"]),
                "box_y_min": int(box["y_min"]),
                "box_x_max": int(box["x_max"]),
                "box_y_max": int(box["y_max"]),
            }
            if source_has_state:
                state_json = json.dumps({"lines": list(state_lines_full)}, ensure_ascii=False, separators=(",", ":"))
                prompt_text = format_prompt_text(prompt_fields, state_json=state_json)
                row = make_state_record(
                    sample_id=sample_id,
                    image_rel_path=image_rel_path,
                    prompt_text=prompt_text,
                    target_lines=target_lines,
                    state_lines=state_lines_full,
                    system_prompt=system_prompt,
                )
            else:
                prompt_text = format_prompt_text(prompt_fields)
                row = make_record(
                    sample_id=sample_id,
                    image_rel_path=image_rel_path,
                    prompt_text=prompt_text,
                    target_lines=target_lines,
                    system_prompt=system_prompt,
                )
            meta = {
                "id": sample_id,
                "source_id": source_id,
                "split": split,
                "family_id": src_meta.get("family_id"),
                "source_image": src_meta.get("source_image"),
                "patch_id": src_meta.get("patch_id"),
                "row": src_meta.get("row"),
                "col": src_meta.get("col"),
                "scan_index": src_meta.get("scan_index"),
                "image": image_rel_path,
                "crop_box": crop_box,
                "target_mode": "fixed_grid_target_box_map",
                "coord_system": src_meta.get("coord_system", "patch_local_896"),
                "source_dataset_type": "state" if source_has_state else "patch_only",
                "serialization_mode": src_meta.get("serialization_mode", "paper_structured"),
                "line_direction_mode": src_meta.get("line_direction_mode", "canonical_cut_then_origin"),
                "line_sort_mode": src_meta.get("line_sort_mode", "first_point_distance_to_patch_origin"),
                "resample_mode": "equal_distance" if resample_step_px > 0 else "inherit_source_spacing",
                "resample_step_px": float(resample_step_px),
                "has_system_prompt": bool(system_prompt.strip()),
                "grid_size": int(grid_size),
                "grid_row": int(box["grid_row"]),
                "grid_col": int(box["grid_col"]),
                "target_box": {
                    "x_min": int(box["x_min"]),
                    "y_min": int(box["y_min"]),
                    "x_max": int(box["x_max"]),
                    "y_max": int(box["y_max"]),
                },
                "target_box_area": int(
                    (int(box["x_max"]) - int(box["x_min"]) + 1)
                    * (int(box["y_max"]) - int(box["y_min"]) + 1)
                ),
                "anchor_source": str(prompt_info["anchor_source"]),
                "anchor_start_xy": [int(prompt_info["start_x"]), int(prompt_info["start_y"])],
                "anchor_end_xy": [int(prompt_info["end_x"]), int(prompt_info["end_y"])],
                "anchor_piece_points": prompt_info["anchor_piece_points"],
                "num_state_lines": int(len(state_lines_full)),
                "state_lines": state_lines_full,
                "num_target_lines": len(target_lines),
                "num_target_points": int(sum(len(x.get("points", [])) for x in target_lines)),
                "prompt_text": prompt_text,
                "target_lines": target_lines,
            }
            generated_pairs.append((row, meta))

            if export_visualizations and patch_image is not None:
                if max_visualizations_per_split <= 0 or viz_count < max_visualizations_per_split:
                    out_path = (
                        output_root
                        / "visualizations"
                        / split
                        / str(src_meta.get("family_id"))
                        / f"p{int(src_meta.get('patch_id', 0)):02d}_g{int(box['grid_row'])}{int(box['grid_col'])}.png"
                    )
                    save_visualization(
                        patch_image=patch_image,
                        target_lines=target_lines,
                        target_box=box,
                        anchor_piece_points=prompt_info["anchor_piece_points"],
                        out_path=out_path,
                    )
                    viz_count += 1

        if patch_image is not None:
            patch_image.close()

    kept_pairs, split_summary = filter_pairs_to_empty_ratio(
        pairs=generated_pairs,
        target_empty_ratio=target_empty_ratio,
        rng=rng,
    )
    out_rows = [row for row, _ in kept_pairs]
    out_meta = [meta for _, meta in kept_pairs]
    count_rows = write_jsonl(output_root / f"{split}.jsonl", out_rows)
    count_meta = write_jsonl(output_root / f"meta_{split}.jsonl", out_meta)
    split_summary.update(
        {
            "used_source_samples": used_source if max_source_samples_per_split <= 0 else min(used_source, max_source_samples_per_split),
            "written_rows": count_rows,
            "written_meta_rows": count_meta,
            "visualizations": viz_count,
        }
    )
    return split_summary


def build_dataset_info(output_root: Path, splits: Sequence[str]) -> Dict[str, Dict]:
    base = sanitize_name(output_root.name)
    info: Dict[str, Dict] = {}
    for split in splits:
        info[f"unimapgen_{base}_{split}"] = {
            "file_name": str((output_root / f"{split}.jsonl").resolve()),
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "images": "images",
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    return info


def build_fixed_grid_targetbox_dataset(
    input_root: Path,
    output_root: Path,
    splits: Sequence[str],
    grid_size: int,
    target_empty_ratio: float,
    target_empty_ratio_by_split: Optional[Dict[str, float]],
    seed: int,
    max_source_samples_per_split: int,
    boundary_tol_px: float,
    resample_step_px: float,
    reuse_system_prompt: bool,
    image_root_mode: str,
    export_visualizations: bool,
    max_visualizations_per_split: int,
) -> Dict[str, object]:
    if int(grid_size) <= 0:
        raise ValueError("--grid-size must be positive.")
    if not (0.0 <= float(target_empty_ratio) <= 1.0):
        raise ValueError("--target-empty-ratio must be in [0, 1]. Use 1.0 to keep all empty boxes.")

    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()
    ensure_dir(output_root)

    image_mode = link_or_copy_images(input_root=input_root, output_root=output_root, mode=str(image_root_mode))
    rng = random.Random(int(seed))

    summary: Dict[str, object] = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "grid_size": int(grid_size),
        "num_boxes_per_patch": int(grid_size) * int(grid_size),
        "target_empty_ratio": float(target_empty_ratio),
        "seed": int(seed),
        "image_root_mode": image_mode,
        "splits": {},
    }

    split_list = [str(x) for x in splits]
    for split in split_list:
        split_empty_ratio = float(target_empty_ratio)
        if target_empty_ratio_by_split is not None and split in target_empty_ratio_by_split:
            split_empty_ratio = float(target_empty_ratio_by_split[split])
        summary["splits"][split] = build_split(
            split=split,
            input_root=input_root,
            output_root=output_root,
            grid_size=int(grid_size),
            target_empty_ratio=float(split_empty_ratio),
            rng=rng,
            max_source_samples_per_split=int(max_source_samples_per_split),
            boundary_tol_px=float(boundary_tol_px),
            resample_step_px=float(resample_step_px),
            reuse_system_prompt=bool(reuse_system_prompt),
            export_visualizations=bool(export_visualizations),
            max_visualizations_per_split=int(max_visualizations_per_split),
        )

    dataset_info = build_dataset_info(output_root=output_root, splits=split_list)
    (output_root / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = build_fixed_grid_targetbox_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        splits=[str(x) for x in args.splits],
        grid_size=int(args.grid_size),
        target_empty_ratio=float(args.target_empty_ratio),
        target_empty_ratio_by_split=None,
        seed=int(args.seed),
        max_source_samples_per_split=int(args.max_source_samples_per_split),
        boundary_tol_px=float(args.boundary_tol_px),
        resample_step_px=float(args.resample_step_px),
        reuse_system_prompt=bool(args.use_system_prompt_from_source),
        image_root_mode=str(args.image_root_mode),
        export_visualizations=bool(args.export_visualizations),
        max_visualizations_per_split=int(args.max_visualizations_per_split),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
