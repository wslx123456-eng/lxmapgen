import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from pyproj import CRS, Transformer
from rasterio import open as rasterio_open
from rasterio.transform import Affine

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_llamafactory_patch_only_from_raw_family_manifest import (
    canonicalize_line_direction,
    clamp_points,
    clip_polyline_to_rect,
    dedup_points,
    point_boundary_side,
    resample_polyline,
    simplify_for_json,
    sort_lines,
)
from export_llamafactory_state_sft_from_raw_family_manifest import build_state_lines_by_mode


DEFAULT_IMAGE_RELPATH = "patch_tif/0.tif"
DEFAULT_MASK_RELPATH = "patch_tif/0_edit_poly.tif"
DEFAULT_LANE_RELPATH = "label_check_crop/Lane.geojson"
DEFAULT_INTERSECTION_RELPATH = "label_check_crop/Intersection.geojson"
DEFAULT_STAGEA_PROMPT_TEMPLATE = """<image>
Please construct the complete road-structure line map in the current satellite patch."""
DEFAULT_STAGEA_SYSTEM_PROMPT = (
    "You are a road-structure reconstruction assistant for satellite-image patches.\n"
    "Predict the complete patch-local line map from the current image.\n"
    "The output JSON schema is {\"lines\": [...]}.\n"
    "Each line must stay in patch-local UV coordinates.\n"
    "Use patch-local integer UV coordinates where one pixel equals one unit.\n"
    "Use category lane_line for roads and intersection_polygon for intersections.\n"
    "Return only valid JSON and no extra text."
)
DEFAULT_STAGEB_PROMPT_TEMPLATE = """<image>
Please construct the road-structure line map in the current patch.
The previous state contains cut traces passed from already processed neighboring patches.
Continue those traces when appropriate and also predict all owned line segments for the current patch.
Previous state:
{state_json}"""
DEFAULT_STAGEB_SYSTEM_PROMPT = (
    "You are a road-structure reconstruction assistant for satellite-image patches.\n"
    "Use the image and the previous line-map state to predict the current patch.\n"
    "The previous state contains cut traces from already processed neighboring patches.\n"
    "Preserve cross-patch continuity whenever those traces enter the current patch.\n"
    "The output JSON schema is {\"lines\": [...]}.\n"
    "Each line must stay in patch-local UV coordinates.\n"
    "Use patch-local integer UV coordinates where one pixel equals one unit.\n"
    "Use category lane_line for roads and intersection_polygon for intersections.\n"
    "Return only valid JSON and no markdown fences."
)


@dataclass
class RasterMeta:
    path: str
    width: int
    height: int
    crs: str
    transform: List[float]

    @property
    def affine(self) -> Affine:
        return Affine(*self.transform)


@dataclass(frozen=True)
class TileWindow:
    x0: int
    y0: int
    x1: int
    y1: int
    keep_x0: int
    keep_y0: int
    keep_x1: int
    keep_y1: int
    mask_ratio: float = 0.0
    mask_pixels: int = 0

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return int(self.x0), int(self.y0), int(self.x1), int(self.y1)

    @property
    def keep_bbox(self) -> Tuple[int, int, int, int]:
        return int(self.keep_x0), int(self.keep_y0), int(self.keep_x1), int(self.keep_y1)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    ensure_dir(path.parent)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_rgb_geotiff(path: Path, band_indices: Sequence[int]) -> Tuple[np.ndarray, RasterMeta]:
    with rasterio_open(path) as ds:
        arr = ds.read(indexes=[int(x) for x in band_indices])
        image = np.transpose(arr, (1, 2, 0)).astype(np.float32)
        meta = RasterMeta(
            path=str(path),
            width=int(ds.width),
            height=int(ds.height),
            crs=str(ds.crs) if ds.crs is not None else "",
            transform=[float(x) for x in tuple(ds.transform)[:6]],
        )
    return image, meta


def read_raster_meta(path: Path) -> RasterMeta:
    with rasterio_open(path) as ds:
        return RasterMeta(
            path=str(path),
            width=int(ds.width),
            height=int(ds.height),
            crs=str(ds.crs) if ds.crs is not None else "",
            transform=[float(x) for x in tuple(ds.transform)[:6]],
        )


def read_binary_mask(path: Path, threshold: int) -> np.ndarray:
    with rasterio_open(path) as ds:
        arr = ds.read(1)
    return (arr > int(threshold)).astype(np.uint8)


def detect_geojson_crs(geojson_dict: Dict) -> str:
    crs = geojson_dict.get("crs", {})
    props = crs.get("properties", {}) if isinstance(crs, dict) else {}
    name = props.get("name")
    if isinstance(name, str) and name.strip():
        return str(name).strip()
    return "urn:ogc:def:crs:OGC:1.3:CRS84"


def build_transformer(src_crs: str, dst_crs: str) -> Transformer:
    return Transformer.from_crs(CRS.from_user_input(src_crs), CRS.from_user_input(dst_crs), always_xy=True)


def project_coords(coordinates, transformer: Transformer) -> np.ndarray:
    points = []
    for value in coordinates:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            continue
        #使用trans方法转换后记录
        x, y = transformer.transform(float(value[0]), float(value[1]))
        points.append([float(x), float(y)])
    return np.asarray(points, dtype=np.float32)


def world_to_pixel(points_world: np.ndarray, affine: Affine) -> np.ndarray:
    #点数量为0则返回空数组
    if points_world.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    #取 affine 的逆变换，正变换时像素坐标 -> 世界坐标，逆变换相反
    inv = ~affine
    cols = []
    rows = []
    for x, y in points_world:
        col, row = inv * (float(x), float(y))
        cols.append(float(col))
        rows.append(float(row))
    return np.stack([cols, rows], axis=-1).astype(np.float32)


def compute_mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def expand_bbox(
    bbox: Optional[Tuple[int, int, int, int]],
    pad_px: int,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    if bbox is None:
        return 0, 0, int(width), int(height)
    x0, y0, x1, y1 = bbox
    pad = max(0, int(pad_px))
    return (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(int(width), x1 + pad),
        min(int(height), y1 + pad),
    )


def build_axis_centers_for_region(
    region_start: int,
    region_end: int,
    crop_size_px: int,
    base_start_px: int,
    base_stride_px: int,
    axis_count: int,
) -> List[int]:
    crop_size_px = max(1, int(crop_size_px))
    half = crop_size_px // 2
    min_center = int(region_start) + half
    max_center = int(region_end) - half
    if max_center < min_center:
        return []
    stride = max(1, int(base_stride_px))
    anchor = int(region_start) + int(base_start_px)
    min_count = max(1, int(axis_count))

    centers = [int(anchor + stride * i) for i in range(min_count)]
    centers = sorted({int(c) for c in centers if min_center <= int(c) <= max_center})

    if not centers:
        center = int(round(0.5 * float(min_center + max_center)))
        return [int(np.clip(center, min_center, max_center))]

    first = int(centers[0])
    while first - stride >= min_center:
        first -= stride
        centers.insert(0, int(first))

    last = int(centers[-1])
    while last + stride <= max_center:
        last += stride
        centers.append(int(last))

    if centers[0] > min_center:
        centers.insert(0, int(min_center))
    if centers[-1] < max_center:
        centers.append(int(max_center))

    deduped: List[int] = []
    for center in centers:
        clipped = int(np.clip(int(center), min_center, max_center))
        if not deduped or deduped[-1] != clipped:
            deduped.append(clipped)
    return deduped


def build_family_patches_from_centers(
    x_centers: Sequence[int],
    y_centers: Sequence[int],
    crop_size_px: int,
    family_grid_size: int,
) -> List[Dict]:
    crop_size_px = max(1, int(crop_size_px))
    half = crop_size_px // 2
    x_centers = [int(x) for x in x_centers]
    y_centers = [int(y) for y in y_centers]
    if len(x_centers) == 0 or len(y_centers) == 0:
        return []
    grid_size = max(1, min(int(family_grid_size), len(x_centers), len(y_centers)))
    families: List[Dict] = []
    max_row0 = max(0, len(y_centers) - grid_size)
    max_col0 = max(0, len(x_centers) - grid_size)
    for row0 in range(max_row0 + 1):
        for col0 in range(max_col0 + 1):
            patches: List[Dict] = []
            for row in range(grid_size):
                for col in range(grid_size):
                    center_x = int(x_centers[col0 + col])
                    center_y = int(y_centers[row0 + row])
                    patch_id = int(row * grid_size + col)
                    patches.append(
                        {
                            "patch_id": patch_id,
                            "row": int(row),
                            "col": int(col),
                            "center_x": int(center_x),
                            "center_y": int(center_y),
                            "crop_box": {
                                "x_min": int(center_x - half),
                                "y_min": int(center_y - half),
                                "x_max": int(center_x + half),
                                "y_max": int(center_y + half),
                                "center_x": int(center_x),
                                "center_y": int(center_y),
                            },
                        }
                    )
            families.append(
                {
                    "row0": int(row0),
                    "col0": int(col0),
                    "grid_size": int(grid_size),
                    "patches": patches,
                }
            )
    return families


def assign_family_ownership_keep_boxes(patches: Sequence[Dict], grid_size: int) -> List[Dict]:
    patch_map = {(int(patch["row"]), int(patch["col"])): patch for patch in patches}
    out: List[Dict] = []
    for patch in patches:
        row = int(patch["row"])
        col = int(patch["col"])
        crop_box = patch["crop_box"]
        left = float(crop_box["x_min"])
        right = float(crop_box["x_max"])
        top = float(crop_box["y_min"])
        bottom = float(crop_box["y_max"])
        if (row, col - 1) in patch_map:
            left_neighbor = patch_map[(row, col - 1)]
            left = 0.5 * (float(patch["center_x"]) + float(left_neighbor["center_x"]))
        if (row, col + 1) in patch_map:
            right_neighbor = patch_map[(row, col + 1)]
            right = 0.5 * (float(patch["center_x"]) + float(right_neighbor["center_x"]))
        if (row - 1, col) in patch_map:
            top_neighbor = patch_map[(row - 1, col)]
            top = 0.5 * (float(patch["center_y"]) + float(top_neighbor["center_y"]))
        if (row + 1, col) in patch_map:
            bottom_neighbor = patch_map[(row + 1, col)]
            bottom = 0.5 * (float(patch["center_y"]) + float(bottom_neighbor["center_y"]))
        patched = dict(patch)
        patched["keep_box"] = {
            "x_min": int(round(left)),
            "y_min": int(round(top)),
            "x_max": int(round(right)),
            "y_max": int(round(bottom)),
        }
        out.append(patched)
    return out

#滑动窗口裁剪，如果滑动时剩余空间大于窗口大小则直接加，如果小于则滑到边界
def _sliding_positions(start: int, end: int, tile_size: int, limit: int, stride: int) -> List[int]:
    start = max(0, int(start))
    end = min(int(limit), int(end))
    tile_size = max(1, int(tile_size))
    stride = max(1, int(stride))
    if end - start <= tile_size:
        return [int(start)]
    positions = list(range(int(start), max(int(start), int(end - tile_size)) + 1, stride))
    last = max(int(start), int(end - tile_size))
    if not positions or positions[-1] != last:
        positions.append(last)
    return [int(p) for p in positions]


def _compute_keep_bbox(
    bbox: Tuple[int, int, int, int],
    width: int,
    height: int,
    keep_margin_px: int,
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = [int(v) for v in bbox]
    margin = max(0, int(keep_margin_px))
    return (
        max(0, x0 + margin),
        max(0, y0 + margin),
        min(int(width), x1 - margin),
        min(int(height), y1 - margin),
    )


def clamp_points_float(points_xy: np.ndarray, patch_size: int) -> np.ndarray:
    arr = np.asarray(points_xy, dtype=np.float32).copy()
    if arr.ndim != 2 or arr.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    upper = max(0.0, float(patch_size))
    arr[:, 0] = np.clip(arr[:, 0], 0.0, upper)
    arr[:, 1] = np.clip(arr[:, 1], 0.0, upper)
    return arr.astype(np.float32)


def clamp_points_float_rect(points_xy: np.ndarray, patch_width: float, patch_height: float) -> np.ndarray:
    arr = np.asarray(points_xy, dtype=np.float32).copy()
    if arr.ndim != 2 or arr.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    arr[:, 0] = np.clip(arr[:, 0], 0.0, max(0.0, float(patch_width)))
    arr[:, 1] = np.clip(arr[:, 1], 0.0, max(0.0, float(patch_height)))
    return arr.astype(np.float32)


def patch_local_size(patch: Dict) -> Tuple[float, float]:
    crop_box = patch["crop_box"]
    return (
        float(crop_box["x_max"] - crop_box["x_min"]),
        float(crop_box["y_max"] - crop_box["y_min"]),
    )


def local_points_to_uv(points_xy: np.ndarray, patch: Dict) -> np.ndarray:
    arr = np.asarray(points_xy, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    patch_width, patch_height = patch_local_size(patch)
    uv = clamp_points_float_rect(arr, patch_width=patch_width, patch_height=patch_height)
    return np.rint(uv).astype(np.float32)


def uv_points_to_local(points_uv: np.ndarray, patch: Dict) -> np.ndarray:
    arr = np.asarray(points_uv, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    patch_width, patch_height = patch_local_size(patch)
    return clamp_points_float_rect(arr, patch_width=patch_width, patch_height=patch_height).astype(np.float32)


def local_lines_to_uv(lines: Sequence[Dict], patch: Dict) -> List[Dict]:
    out: List[Dict] = []
    for line in lines:
        arr = np.asarray(line.get("points", []), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            continue
        uv = local_points_to_uv(arr, patch=patch)
        if str(line.get("geometry_type", "line")) == "polygon":
            uv = ensure_closed_ring(uv)
            if uv.ndim != 2 or uv.shape[0] < 4:
                continue
        else:
            uv = dedup_points(uv)
            if uv.ndim != 2 or uv.shape[0] < 2:
                continue
        copied = dict(line)
        copied["points"] = [[int(round(x)), int(round(y))] for x, y in uv.tolist()]
        copied["coord_system"] = "uv"
        out.append(copied)
    return sort_lines(out)


def uv_lines_to_local(lines: Sequence[Dict], patch: Dict) -> List[Dict]:
    out: List[Dict] = []
    for line in lines:
        arr = np.asarray(line.get("points", []), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            continue
        local = uv_points_to_local(arr, patch=patch)
        if str(line.get("geometry_type", "line")) == "polygon":
            local = ensure_closed_ring(local)
            if local.ndim != 2 or local.shape[0] < 4:
                continue
        else:
            local = dedup_points(local)
            if local.ndim != 2 or local.shape[0] < 2:
                continue
        copied = dict(line)
        copied["points"] = [[float(x), float(y)] for x, y in local.tolist()]
        copied["coord_system"] = "pixel_local"
        out.append(copied)
    return sort_lines(out)


def sanitize_pred_lines_uv(pred_lines: Sequence[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for line in pred_lines:
        arr = np.asarray(line.get("points", []), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] != 2:
            continue
        geometry_type = str(line.get("geometry_type", "line"))
        if geometry_type == "polygon":
            patch_size = int(max(1.0, float(np.max(arr)) + 1.0))
            points_json = simplify_for_json(ensure_closed_ring(arr), patch_size=patch_size)
            if len(points_json) >= 2 and points_json[0] != points_json[-1]:
                points_json.append(list(points_json[0]))
            arr = ensure_closed_ring(dedup_points(np.asarray(points_json, dtype=np.float32)))
            if arr.ndim != 2 or arr.shape[0] < 4:
                continue
            start_type = "closed"
            end_type = "closed"
        else:
            patch_size = int(max(1.0, float(np.max(arr)) + 1.0))
            arr = dedup_points(np.asarray(simplify_for_json(arr, patch_size=patch_size), dtype=np.float32))
            if arr.ndim != 2 or arr.shape[0] < 2:
                continue
            start_type = str(line.get("start_type", "start"))
            end_type = str(line.get("end_type", "end"))
            if start_type not in {"start", "cut"}:
                start_type = "start"
            if end_type not in {"end", "cut"}:
                end_type = "end"
        out.append(
            {
                "category": str(line.get("category", "lane_line")),
                "start_type": start_type,
                "end_type": end_type,
                "geometry_type": geometry_type,
                "coord_system": "uv",
                "points": [[int(round(x)), int(round(y))] for x, y in arr.tolist()],
            }
        )
    return sort_lines(out)


def resample_polyline_preserve_remainder(
    points_xy: np.ndarray,
    step_px: float,
    max_points: Optional[int] = None,
) -> np.ndarray:
    pts = dedup_points(np.asarray(points_xy, dtype=np.float32))
    if pts.ndim != 2 or pts.shape[0] < 2:
        return pts.astype(np.float32)
    step = float(step_px)
    if step <= 0.0:
        return pts.astype(np.float32)
    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    total = float(np.sum(seg))
    if total < 1e-6:
        return pts[:1].astype(np.float32)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    targets: List[float] = [0.0]
    dist = float(step)
    while dist < total:
        targets.append(float(dist))
        dist += float(step)
    if targets[-1] != float(total):
        targets.append(float(total))
    if max_points is not None and int(max_points) > 0 and len(targets) > int(max_points):
        targets = targets[: max(1, int(max_points) - 1)] + [float(total)]
    sampled: List[np.ndarray] = []
    for target in targets:
        if target >= total:
            sampled.append(pts[-1].astype(np.float32))
            continue
        seg_idx = int(np.searchsorted(cum, target, side="right") - 1)
        seg_idx = min(max(seg_idx, 0), len(seg) - 1)
        t0 = float(cum[seg_idx])
        t1 = float(cum[seg_idx + 1])
        ratio = 0.0 if t1 <= t0 else (float(target) - t0) / (t1 - t0)
        sampled.append((pts[seg_idx] * (1.0 - ratio) + pts[seg_idx + 1] * ratio).astype(np.float32))
    return dedup_points(sampled).astype(np.float32)


def _point_in_polygon(point_xy: np.ndarray, ring_xy: np.ndarray) -> bool:
    point = np.asarray(point_xy, dtype=np.float32)
    ring = ensure_closed_ring(np.asarray(ring_xy, dtype=np.float32))
    if ring.ndim != 2 or ring.shape[0] < 4:
        return False
    x = float(point[0])
    y = float(point[1])
    inside = False
    prev = ring[-1]
    for curr in ring:
        x1 = float(prev[0])
        y1 = float(prev[1])
        x2 = float(curr[0])
        y2 = float(curr[1])
        intersects = ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / max(1e-8, (y2 - y1)) + x1)
        if intersects:
            inside = not inside
        prev = curr
    return bool(inside)


def _nearest_point_on_segment(point_xy: np.ndarray, start_xy: np.ndarray, end_xy: np.ndarray) -> Tuple[np.ndarray, float]:
    point = np.asarray(point_xy, dtype=np.float32)
    start = np.asarray(start_xy, dtype=np.float32)
    end = np.asarray(end_xy, dtype=np.float32)
    seg = end - start
    seg_len_sq = float(np.dot(seg, seg))
    if seg_len_sq <= 1e-8:
        nearest = start.astype(np.float32)
    else:
        t = float(np.dot(point - start, seg) / seg_len_sq)
        t = min(1.0, max(0.0, t))
        nearest = (start + seg * t).astype(np.float32)
    dist = float(np.linalg.norm(point - nearest))
    return nearest, dist


def _nearest_point_on_ring_boundary(point_xy: np.ndarray, ring_xy: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    ring = ensure_closed_ring(np.asarray(ring_xy, dtype=np.float32))
    if ring.ndim != 2 or ring.shape[0] < 4:
        return None, float("inf")
    best_point: Optional[np.ndarray] = None
    best_dist = float("inf")
    for idx in range(ring.shape[0] - 1):
        nearest, dist = _nearest_point_on_segment(point_xy=point_xy, start_xy=ring[idx], end_xy=ring[idx + 1])
        if dist < best_dist:
            best_point = nearest.astype(np.float32)
            best_dist = float(dist)
    return best_point, float(best_dist)


def _polygon_contains_any(point_xy: np.ndarray, polygon_rings_xy: Sequence[np.ndarray]) -> bool:
    point = np.asarray(point_xy, dtype=np.float32)
    for ring in polygon_rings_xy:
        ring_arr = ensure_closed_ring(np.asarray(ring, dtype=np.float32))
        if ring_arr.ndim != 2 or ring_arr.shape[0] < 4:
            continue
        if _point_in_polygon(point, ring_arr):
            return True
    return False


def _nearest_point_on_any_ring(point_xy: np.ndarray, polygon_rings_xy: Sequence[np.ndarray]) -> Tuple[Optional[np.ndarray], float]:
    best_point: Optional[np.ndarray] = None
    best_dist = float("inf")
    for ring in polygon_rings_xy:
        candidate, dist = _nearest_point_on_ring_boundary(point_xy=point_xy, ring_xy=ring)
        if candidate is None:
            continue
        if float(dist) < float(best_dist):
            best_point = candidate.astype(np.float32)
            best_dist = float(dist)
    return best_point, float(best_dist)


def _nearest_point_on_polyline(point_xy: np.ndarray, points_xy: np.ndarray) -> Tuple[Optional[np.ndarray], float, int]:
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return None, float("inf"), -1
    best_point: Optional[np.ndarray] = None
    best_dist = float("inf")
    best_seg_idx = -1
    for seg_idx in range(pts.shape[0] - 1):
        candidate, dist = _nearest_point_on_segment(point_xy=point_xy, start_xy=pts[seg_idx], end_xy=pts[seg_idx + 1])
        if float(dist) < float(best_dist):
            best_point = candidate.astype(np.float32)
            best_dist = float(dist)
            best_seg_idx = int(seg_idx)
    return best_point, float(best_dist), int(best_seg_idx)


def _insert_point_into_polyline(points_xy: np.ndarray, point_xy: np.ndarray, seg_idx: int, eps: float = 1e-3) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32)
    point = np.asarray(point_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return pts.astype(np.float32)
    for idx in range(pts.shape[0]):
        if float(np.linalg.norm(pts[idx] - point)) <= float(eps):
            pts[idx] = point.astype(np.float32)
            return dedup_points(pts).astype(np.float32)
    insert_after = min(max(int(seg_idx), 0), pts.shape[0] - 2)
    merged = np.concatenate(
        [
            pts[: insert_after + 1],
            point.astype(np.float32)[None, :],
            pts[insert_after + 1 :],
        ],
        axis=0,
    )
    return dedup_points(merged).astype(np.float32)


def _cluster_endpoint_refs(
    endpoint_refs: Sequence[Dict],
    tol_px: float,
) -> List[List[Dict]]:
    refs = list(endpoint_refs)
    clusters: List[List[Dict]] = []
    visited = [False] * len(refs)
    for start_idx in range(len(refs)):
        if visited[start_idx]:
            continue
        queue = [start_idx]
        visited[start_idx] = True
        cluster_indices: List[int] = []
        while queue:
            idx = queue.pop()
            cluster_indices.append(idx)
            point_a = np.asarray(refs[idx]["point"], dtype=np.float32)
            for other_idx in range(len(refs)):
                if visited[other_idx]:
                    continue
                point_b = np.asarray(refs[other_idx]["point"], dtype=np.float32)
                if float(np.linalg.norm(point_a - point_b)) <= float(tol_px):
                    visited[other_idx] = True
                    queue.append(other_idx)
        clusters.append([refs[idx] for idx in cluster_indices])
    return clusters


def enforce_line_and_polygon_topology(
    segments: Sequence[Dict],
    polygon_rings_xy: Sequence[np.ndarray],
    snap_tol_px: float,
) -> List[Dict]:
    tol = max(0.0, float(snap_tol_px))
    if tol <= 0.0:
        return [dict(segment) for segment in segments]

    line_segments: List[Dict] = []
    polygon_segments: List[Dict] = []
    for segment in segments:
        copied = dict(segment)
        copied["points_global"] = np.asarray(segment["points_global"], dtype=np.float32).copy()
        if str(segment.get("geometry_type", "line")) == "polygon":
            polygon_segments.append(copied)
        else:
            line_segments.append(copied)
    if not line_segments:
        return [*polygon_segments]

    # Step 1: snap all line endpoints to polygon boundaries first.
    for segment in line_segments:
        snapped = snap_line_endpoints_to_polygon_boundaries(
            line_points_xy=np.asarray(segment["points_global"], dtype=np.float32),
            polygon_rings_xy=polygon_rings_xy,
            snap_tol_px=tol,
        )
        segment["points_global"] = dedup_points(snapped).astype(np.float32)

    # Step 2: cluster nearby line endpoints so touching lines share exactly the same endpoint.
    endpoint_refs: List[Dict] = []
    for line_idx, segment in enumerate(line_segments):
        pts = np.asarray(segment["points_global"], dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue
        endpoint_refs.append({"line_idx": int(line_idx), "endpoint_idx": 0, "point": pts[0].astype(np.float32)})
        endpoint_refs.append({"line_idx": int(line_idx), "endpoint_idx": -1, "point": pts[-1].astype(np.float32)})
    for cluster in _cluster_endpoint_refs(endpoint_refs=endpoint_refs, tol_px=tol):
        if len(cluster) <= 1:
            continue
        unique_line_indices = {int(ref["line_idx"]) for ref in cluster}
        if len(unique_line_indices) <= 1:
            continue
        polygon_anchor: Optional[np.ndarray] = None
        polygon_anchor_dist = float("inf")
        for ref in cluster:
            candidate, dist = _nearest_point_on_any_ring(point_xy=np.asarray(ref["point"], dtype=np.float32), polygon_rings_xy=polygon_rings_xy)
            if candidate is not None and float(dist) <= float(tol) and float(dist) < float(polygon_anchor_dist):
                polygon_anchor = candidate.astype(np.float32)
                polygon_anchor_dist = float(dist)
        if polygon_anchor is not None:
            anchor = polygon_anchor.astype(np.float32)
        else:
            best_ref = min(
                cluster,
                key=lambda ref: sum(
                    float(
                        np.linalg.norm(
                            np.asarray(ref["point"], dtype=np.float32) - np.asarray(other["point"], dtype=np.float32)
                        )
                    )
                    for other in cluster
                ),
            )
            anchor = np.asarray(best_ref["point"], dtype=np.float32)
        for ref in cluster:
            pts = np.asarray(line_segments[int(ref["line_idx"])]["points_global"], dtype=np.float32).copy()
            endpoint_idx = 0 if int(ref["endpoint_idx"]) == 0 else -1
            pts[endpoint_idx] = anchor.astype(np.float32)
            line_segments[int(ref["line_idx"])]["points_global"] = dedup_points(pts).astype(np.float32)

    # Step 3: snap line endpoints to nearby line interiors and insert shared vertices on host lines.
    for line_idx, segment in enumerate(line_segments):
        pts = np.asarray(segment["points_global"], dtype=np.float32).copy()
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue
        for endpoint_idx in (0, -1):
            endpoint = pts[endpoint_idx].astype(np.float32)
            best_line_idx = -1
            best_anchor: Optional[np.ndarray] = None
            best_dist = float("inf")
            best_seg_idx = -1
            for other_idx, other_segment in enumerate(line_segments):
                if int(other_idx) == int(line_idx):
                    continue
                other_pts = np.asarray(other_segment["points_global"], dtype=np.float32)
                if other_pts.ndim != 2 or other_pts.shape[0] < 2:
                    continue
                nearest, dist, seg_idx = _nearest_point_on_polyline(endpoint, other_pts)
                if nearest is None:
                    continue
                if float(dist) <= float(tol) and float(dist) < float(best_dist):
                    best_line_idx = int(other_idx)
                    best_anchor = nearest.astype(np.float32)
                    best_dist = float(dist)
                    best_seg_idx = int(seg_idx)
            if best_anchor is None:
                continue
            pts[endpoint_idx] = best_anchor.astype(np.float32)
            pts = dedup_points(pts).astype(np.float32)
            host_pts = np.asarray(line_segments[int(best_line_idx)]["points_global"], dtype=np.float32)
            host_pts = _insert_point_into_polyline(
                points_xy=host_pts,
                point_xy=best_anchor.astype(np.float32),
                seg_idx=int(best_seg_idx),
                eps=max(1e-3, float(tol) * 0.25),
            )
            line_segments[int(best_line_idx)]["points_global"] = host_pts.astype(np.float32)
        line_segments[int(line_idx)]["points_global"] = pts.astype(np.float32)

    # Step 4: final endpoint snap to polygon boundaries after line-line adjustments.
    finalized: List[Dict] = []
    for segment in line_segments:
        pts = snap_line_endpoints_to_polygon_boundaries(
            line_points_xy=np.asarray(segment["points_global"], dtype=np.float32),
            polygon_rings_xy=polygon_rings_xy,
            snap_tol_px=tol,
        )
        pts = dedup_points(pts).astype(np.float32)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue
        copied = dict(segment)
        copied["points_global"] = pts.astype(np.float32)
        finalized.append(copied)
    return [*polygon_segments, *finalized]


def _refine_polygon_transition_point(
    point_a: np.ndarray,
    point_b: np.ndarray,
    inside_a: bool,
    inside_b: bool,
    polygon_rings_xy: Sequence[np.ndarray],
    iterations: int = 10,
) -> Optional[np.ndarray]:
    if bool(inside_a) == bool(inside_b) or not polygon_rings_xy:
        return None
    low = np.asarray(point_a, dtype=np.float32).copy()
    high = np.asarray(point_b, dtype=np.float32).copy()
    low_inside = bool(inside_a)
    for _ in range(max(1, int(iterations))):
        mid = ((low + high) * 0.5).astype(np.float32)
        mid_inside = _polygon_contains_any(mid, polygon_rings_xy=polygon_rings_xy)
        if mid_inside == low_inside:
            low = mid
        else:
            high = mid
    candidate = ((low + high) * 0.5).astype(np.float32)
    nearest, _ = _nearest_point_on_any_ring(candidate, polygon_rings_xy=polygon_rings_xy)
    if nearest is not None:
        return nearest.astype(np.float32)
    return candidate.astype(np.float32)


def clip_line_outside_polygons(
    points_xy: np.ndarray,
    polygon_rings_xy: Sequence[np.ndarray],
    min_points: int = 2,
) -> List[Dict]:
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < int(min_points):
        return []
    valid_rings = [ensure_closed_ring(np.asarray(ring, dtype=np.float32)) for ring in polygon_rings_xy]
    valid_rings = [ring for ring in valid_rings if ring.ndim == 2 and ring.shape[0] >= 4]
    if not valid_rings:
        return [{"points": pts.astype(np.float32)}]
    dense_pts = _densify_line_for_mask(points_xy=pts, step_px=1.0)
    outside = [not _polygon_contains_any(point_xy=point, polygon_rings_xy=valid_rings) for point in dense_pts]
    out: List[Dict] = []
    start = None
    piece_points: List[np.ndarray] = []
    for idx, flag in enumerate(outside):
        if flag and start is None:
            piece_points = []
            if idx > 0:
                transition = _refine_polygon_transition_point(
                    point_a=dense_pts[idx - 1],
                    point_b=dense_pts[idx],
                    inside_a=not bool(outside[idx - 1]),
                    inside_b=not bool(flag),
                    polygon_rings_xy=valid_rings,
                )
                if transition is not None:
                    piece_points.append(transition.astype(np.float32))
            start = idx
        if flag and start is not None:
            if not piece_points or not np.allclose(piece_points[-1], dense_pts[idx], atol=1e-3):
                piece_points.append(dense_pts[idx].astype(np.float32))
        if (not flag) and start is not None:
            if idx > 0:
                transition = _refine_polygon_transition_point(
                    point_a=dense_pts[idx - 1],
                    point_b=dense_pts[idx],
                    inside_a=not bool(outside[idx - 1]),
                    inside_b=not bool(flag),
                    polygon_rings_xy=valid_rings,
                )
                if transition is not None and (
                    not piece_points or not np.allclose(piece_points[-1], transition, atol=1e-3)
                ):
                    piece_points.append(transition.astype(np.float32))
            piece = np.asarray(piece_points, dtype=np.float32)
            if piece.ndim == 2 and piece.shape[0] >= int(min_points):
                out.append({"points": dedup_points(piece).astype(np.float32)})
            start = None
            piece_points = []
    if start is not None:
        piece = np.asarray(piece_points, dtype=np.float32)
        if piece.ndim == 2 and piece.shape[0] >= int(min_points):
            out.append({"points": dedup_points(piece).astype(np.float32)})
    return out


def snap_line_endpoints_to_polygon_boundaries(
    line_points_xy: np.ndarray,
    polygon_rings_xy: Sequence[np.ndarray],
    snap_tol_px: float = 12.0,
) -> np.ndarray:
    pts = np.asarray(line_points_xy, dtype=np.float32).copy()
    if pts.ndim != 2 or pts.shape[0] < 2 or not polygon_rings_xy:
        return pts.astype(np.float32)
    tolerance = max(0.0, float(snap_tol_px))
    for endpoint_idx in (0, -1):
        point = pts[endpoint_idx].astype(np.float32)
        best_point: Optional[np.ndarray] = None
        best_dist = float("inf")
        point_inside_any = False
        for ring in polygon_rings_xy:
            ring_arr = ensure_closed_ring(np.asarray(ring, dtype=np.float32))
            if ring_arr.ndim != 2 or ring_arr.shape[0] < 4:
                continue
            is_inside = _point_in_polygon(point, ring_arr)
            nearest, dist = _nearest_point_on_ring_boundary(point, ring_arr)
            if nearest is None:
                continue
            if is_inside:
                point_inside_any = True
            if is_inside or dist <= tolerance:
                if dist < best_dist:
                    best_dist = float(dist)
                    best_point = nearest.astype(np.float32)
        if best_point is not None and (point_inside_any or best_dist <= tolerance):
            pts[endpoint_idx] = best_point.astype(np.float32)
    return dedup_points(pts).astype(np.float32)


def generate_tile_windows(
    width: int,
    height: int,
    tile_size_px: int,
    overlap_px: int,
    region_bbox: Optional[Tuple[int, int, int, int]],
    keep_margin_px: int,
) -> List[TileWindow]:
    #滑动步长为patch尺寸减overlap，这里overlap我设为0
    stride = max(1, int(tile_size_px) - int(overlap_px))
    rx0, ry0, rx1, ry1 = (0, 0, int(width), int(height)) if region_bbox is None else tuple(int(v) for v in region_bbox)
    #用numpy数据记录每一个窗口的xy值，交给后续处理
    xs = _sliding_positions(start=rx0, end=rx1, tile_size=int(tile_size_px), limit=int(width), stride=int(stride))
    ys = _sliding_positions(start=ry0, end=ry1, tile_size=int(tile_size_px), limit=int(height), stride=int(stride))
    out: List[TileWindow] = []
    #输出窗口的xy值对，同时有左上角和右下角
    for y0 in ys:
        for x0 in xs:
            x1 = min(int(width), int(x0 + tile_size_px))
            y1 = min(int(height), int(y0 + tile_size_px))
            #_compute_keep_bbox这里的计算没有意义了，我把overlap和margin都设置为0了，计算的keepbox就等于patch尺寸
            keep_bbox = _compute_keep_bbox(
                bbox=(x0, y0, x1, y1),
                width=int(width),
                height=int(height),
                keep_margin_px=int(keep_margin_px),
            )
            out.append(
                TileWindow(
                    x0=int(x0),
                    y0=int(y0),
                    x1=int(x1),
                    y1=int(y1),
                    keep_x0=int(keep_bbox[0]),
                    keep_y0=int(keep_bbox[1]),
                    keep_x1=int(keep_bbox[2]),
                    keep_y1=int(keep_bbox[3]),
                )
            )
    return out


def annotate_tile_windows_with_mask(tile_windows: Sequence[TileWindow], mask: Optional[np.ndarray]) -> List[TileWindow]:
    if mask is None:
        return list(tile_windows)
    out: List[TileWindow] = []
    for window in tile_windows:
        x0, y0, x1, y1 = window.bbox
        crop = mask[y0:y1, x0:x1]
        mask_pixels = int(crop.sum()) if crop.size > 0 else 0
        mask_ratio = float(crop.mean()) if crop.size > 0 else 0.0
        out.append(
            TileWindow(
                x0=window.x0,
                y0=window.y0,
                x1=window.x1,
                y1=window.y1,
                keep_x0=window.keep_x0,
                keep_y0=window.keep_y0,
                keep_x1=window.keep_x1,
                keep_y1=window.keep_y1,
                mask_ratio=mask_ratio,
                mask_pixels=mask_pixels,
            )
        )
    return out


def audit_tile_window_selection(
    tile_windows: Sequence[TileWindow],
    min_mask_ratio: float,
    min_mask_pixels: int,
    max_tiles: Optional[int],
    fallback_to_all_if_empty: bool,
) -> Tuple[List[TileWindow], List[Dict]]:
    all_windows = list(tile_windows)
    filtered = [
        window
        for window in all_windows
        if float(window.mask_ratio) >= float(min_mask_ratio) or int(window.mask_pixels) >= int(min_mask_pixels)
    ]
    used_fallback = len(filtered) == 0 and bool(fallback_to_all_if_empty)
    candidates = filtered if filtered else (list(all_windows) if bool(fallback_to_all_if_empty) else [])
    candidates = sorted(candidates, key=lambda item: (float(item.mask_ratio), int(item.mask_pixels)), reverse=True)
    selected = list(candidates)
    if max_tiles is not None and int(max_tiles) > 0:
        selected = selected[: int(max_tiles)]
    selected_keys = {window.bbox for window in selected}
    candidate_keys = {window.bbox for window in candidates}
    audits: List[Dict] = []
    for index, window in enumerate(all_windows):
        key = window.bbox
        if key in selected_keys:
            reason = "selected"
        elif (not used_fallback) and key not in candidate_keys:
            reason = "below_mask_threshold"
        elif max_tiles is not None and int(max_tiles) > 0 and key in candidate_keys:
            reason = "truncated_by_max_tiles"
        else:
            reason = "discarded"
        audits.append(
            {
                "candidate_index": int(index),
                "selected": bool(key in selected_keys),
                "reason": str(reason),
                "bbox": [int(v) for v in window.bbox],
                "keep_bbox": [int(v) for v in window.keep_bbox],
                "mask_ratio": float(window.mask_ratio),
                "mask_pixels": int(window.mask_pixels),
            }
        )
    return selected, audits


def _densify_line_for_mask(points_xy: np.ndarray, step_px: float = 1.0) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] <= 1:
        return pts.astype(np.float32)
    out: List[np.ndarray] = [pts[0].astype(np.float32)]
    step = max(0.25, float(step_px))
    for start_pt, end_pt in zip(pts[:-1], pts[1:]):
        seg = np.asarray(end_pt - start_pt, dtype=np.float32)
        seg_len = float(np.linalg.norm(seg))
        steps = max(1, int(np.ceil(seg_len / step)))
        for t in np.linspace(0.0, 1.0, steps + 1, dtype=np.float32)[1:]:
            out.append((start_pt + seg * float(t)).astype(np.float32))
    return np.stack(out, axis=0).astype(np.float32)


def _mask_contains_point(review_mask: np.ndarray, point_xy: np.ndarray) -> bool:
    x = int(np.clip(round(float(point_xy[0])), 0, int(review_mask.shape[1]) - 1))
    y = int(np.clip(round(float(point_xy[1])), 0, int(review_mask.shape[0]) - 1))
    return bool(review_mask[y, x] > 0)


def _refine_mask_transition_point(
    review_mask: np.ndarray,
    point_a: np.ndarray,
    point_b: np.ndarray,
    inside_a: bool,
    inside_b: bool,
    iterations: int = 8,
) -> Optional[np.ndarray]:
    if bool(inside_a) == bool(inside_b):
        return None
    low = np.asarray(point_a, dtype=np.float32).copy()
    high = np.asarray(point_b, dtype=np.float32).copy()
    low_inside = bool(inside_a)
    for _ in range(max(1, int(iterations))):
        mid = ((low + high) * 0.5).astype(np.float32)
        mid_inside = _mask_contains_point(review_mask=review_mask, point_xy=mid)
        if mid_inside == low_inside:
            low = mid
        else:
            high = mid
    return ((low + high) * 0.5).astype(np.float32)


def mask_clip_line(points_xy: np.ndarray, review_mask: Optional[np.ndarray], min_points: int = 2) -> List[Dict]:
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < int(min_points):
        return []
    if review_mask is None:
        return [{"points": pts.astype(np.float32), "cut_start": False, "cut_end": False}]
    dense_pts = _densify_line_for_mask(points_xy=pts, step_px=1.0)
    height, width = review_mask.shape[:2]
    cols = np.clip(np.round(dense_pts[:, 0]).astype(np.int64), 0, width - 1)
    rows = np.clip(np.round(dense_pts[:, 1]).astype(np.int64), 0, height - 1)
    inside = (review_mask[rows, cols] > 0).tolist()
    out: List[Dict] = []
    start = None
    piece_points: List[np.ndarray] = []
    for idx, flag in enumerate(inside):
        if flag and start is None:
            piece_points = []
            if idx > 0:
                transition = _refine_mask_transition_point(
                    review_mask=review_mask,
                    point_a=dense_pts[idx - 1],
                    point_b=dense_pts[idx],
                    inside_a=bool(inside[idx - 1]),
                    inside_b=bool(flag),
                )
                if transition is not None:
                    piece_points.append(transition.astype(np.float32))
            start = idx
        if flag and start is not None:
            if not piece_points or not np.allclose(piece_points[-1], dense_pts[idx], atol=1e-3):
                piece_points.append(dense_pts[idx].astype(np.float32))
        if (not flag) and start is not None:
            if idx > 0:
                transition = _refine_mask_transition_point(
                    review_mask=review_mask,
                    point_a=dense_pts[idx - 1],
                    point_b=dense_pts[idx],
                    inside_a=bool(inside[idx - 1]),
                    inside_b=bool(flag),
                )
                if transition is not None and (
                    not piece_points or not np.allclose(piece_points[-1], transition, atol=1e-3)
                ):
                    piece_points.append(transition.astype(np.float32))
            piece = np.asarray(piece_points, dtype=np.float32)
            if piece.ndim == 2 and piece.shape[0] >= int(min_points):
                out.append(
                    {
                        "points": piece.astype(np.float32),
                        "cut_start": bool(start > 0) or not np.allclose(piece[0], pts[0], atol=1e-3),
                        "cut_end": True,
                    }
                )
            start = None
            piece_points = []
    if start is not None:
        piece = np.asarray(piece_points, dtype=np.float32)
        if piece.ndim == 2 and piece.shape[0] >= int(min_points):
            out.append(
                {
                    "points": piece.astype(np.float32),
                    "cut_start": bool(start > 0) or not np.allclose(piece[0], pts[0], atol=1e-3),
                    "cut_end": not np.allclose(piece[-1], pts[-1], atol=1e-3),
                }
            )
    return out


def _line_piece_cut_flags_after_clip(
    source_points: np.ndarray,
    clipped_points: np.ndarray,
    cut_start: bool,
    cut_end: bool,
) -> Tuple[bool, bool]:
    src = np.asarray(source_points, dtype=np.float32)
    dst = np.asarray(clipped_points, dtype=np.float32)
    if src.ndim != 2 or dst.ndim != 2 or src.shape[0] == 0 or dst.shape[0] == 0:
        return bool(cut_start), bool(cut_end)
    tol = 1e-3
    return bool(cut_start) or not np.allclose(dst[0], src[0], atol=tol), bool(cut_end) or not np.allclose(
        dst[-1], src[-1], atol=tol
    )


def geojson_lines_to_pixel_lines(geojson_dict: Dict, raster_meta: RasterMeta, category: str) -> List[Dict]:
    #获得geojson中的CRS 是什么，得到坐标系是什么才能做坐标系转换
    src_crs = detect_geojson_crs(geojson_dict)
    #构建GeoJSON 坐标系 -> tif 坐标系的转换器
    transformer = build_transformer(src_crs=src_crs, dst_crs=raster_meta.crs)
    out: List[Dict] = []
    #遍历每个 feature
    for feature in geojson_dict.get("features", []):
        if not isinstance(feature, dict):
            continue
        #取出 geometry
        geometry = feature.get("geometry", {})
        # 只保留 LineString
        if str(geometry.get("type", "")).strip().lower() != "linestring":
            continue
        #把 GeoJSON 坐标投影到 tif 坐标系
        world = project_coords(geometry.get("coordinates", []), transformer=transformer)
        #把世界坐标转成像素坐标
        pixel = dedup_points(world_to_pixel(world, affine=raster_meta.affine))
        #过滤掉无效线,如果最后不是合法二维点列或者只剩不到 2 个点
        if pixel.ndim != 2 or pixel.shape[0] < 2:
            continue
        #写成统一内部格式
        out.append({"category": str(category), "geometry_type": "line", "points_global": pixel.astype(np.float32)})
    return out


def ensure_closed_ring(points_xy: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if pts.shape[0] == 1:
        return np.concatenate([pts, pts], axis=0).astype(np.float32)
    if float(np.linalg.norm(pts[0] - pts[-1])) <= float(eps):
        return pts.astype(np.float32)
    return np.concatenate([pts, pts[:1]], axis=0).astype(np.float32)


def _rdp_recursive(points_xy: np.ndarray, epsilon: float) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.shape[0] <= 2:
        return pts
    start = pts[0]
    end = pts[-1]
    seg = end - start
    seg_len = float(np.linalg.norm(seg))
    if seg_len <= 1e-6:
        distances = np.linalg.norm(pts[1:-1] - start[None, :], axis=1)
    else:
        rel = pts[1:-1] - start[None, :]
        cross = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0])
        distances = cross / seg_len
    if distances.size == 0:
        return np.stack([start, end], axis=0).astype(np.float32)
    index = int(np.argmax(distances))
    max_dist = float(distances[index])
    if max_dist <= float(epsilon):
        return np.stack([start, end], axis=0).astype(np.float32)
    left = _rdp_recursive(pts[: index + 2], epsilon=float(epsilon))
    right = _rdp_recursive(pts[index + 1 :], epsilon=float(epsilon))
    return np.concatenate([left[:-1], right], axis=0).astype(np.float32)


def simplify_polyline_straight(points_xy: np.ndarray, tolerance_px: float = 1.5) -> np.ndarray:
    pts = dedup_points(np.asarray(points_xy, dtype=np.float32))
    if pts.ndim != 2 or pts.shape[0] < 3:
        return pts.astype(np.float32)
    return dedup_points(_rdp_recursive(pts, epsilon=max(0.25, float(tolerance_px)))).astype(np.float32)


def geojson_polygons_to_pixel_rings(geojson_dict: Dict, raster_meta: RasterMeta, category: str) -> List[Dict]:
    src_crs = detect_geojson_crs(geojson_dict)
    transformer = build_transformer(src_crs=src_crs, dst_crs=raster_meta.crs)
    out: List[Dict] = []
    for feature in geojson_dict.get("features", []):
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry", {})
        if str(geometry.get("type", "")).strip().lower() != "polygon":
            continue
        polygon_coords = geometry.get("coordinates", [])
        if not isinstance(polygon_coords, list) or len(polygon_coords) == 0:
            continue
        ring_coords = polygon_coords[0]
        world = project_coords(ring_coords, transformer=transformer)
        pixel = world_to_pixel(world, affine=raster_meta.affine)
        pixel = ensure_closed_ring(dedup_points(pixel))
        if pixel.ndim != 2 or pixel.shape[0] < 4:
            continue
        out.append({"category": str(category), "geometry_type": "polygon", "points_global": pixel.astype(np.float32)})
    return out


def load_sample_global_lines(
    sample_dir: Path,
    raster_meta: RasterMeta,
    lane_relpath: str,
    intersection_relpath: str,
    include_lane: bool = True,
    include_intersection: bool = True,
) -> List[Dict]:
    out: List[Dict] = []
    if include_lane:
        #拼出 lane 文件的真实路径
        lane_path = sample_dir / lane_relpath
        #检查文件是否存在
        if lane_path.is_file():
            #读取并转换 lane
            out.extend(geojson_lines_to_pixel_lines(load_json(lane_path), raster_meta=raster_meta, category="lane_line"))
    if include_intersection:
        intersection_path = sample_dir / intersection_relpath
        if intersection_path.is_file():
            out.extend(
                #转换路口，和转换lane类似，只是数据类型从线变成polygon
                geojson_polygons_to_pixel_rings(
                    load_json(intersection_path),
                    raster_meta=raster_meta,
                    category="intersection_polygon",
                )
            )
    return out


def clip_polygon_ring_to_rect(points_xy: np.ndarray, rect: Tuple[float, float, float, float]) -> List[np.ndarray]:
    pts = ensure_closed_ring(np.asarray(points_xy, dtype=np.float32))
    if pts.ndim != 2 or pts.shape[0] < 4:
        return []
    poly = pts[:-1].astype(np.float32)
    x_min, y_min, x_max, y_max = [float(v) for v in rect]

    def inside_left(p): return float(p[0]) >= x_min
    def inside_right(p): return float(p[0]) <= x_max
    def inside_top(p): return float(p[1]) >= y_min
    def inside_bottom(p): return float(p[1]) <= y_max

    def intersect_vertical(s, e, x_edge):
        s = np.asarray(s, dtype=np.float32)
        e = np.asarray(e, dtype=np.float32)
        dx = float(e[0] - s[0])
        if abs(dx) <= 1e-6:
            return np.asarray([float(x_edge), float(s[1])], dtype=np.float32)
        t = (float(x_edge) - float(s[0])) / dx
        return np.asarray([float(x_edge), float(s[1] + t * (e[1] - s[1]))], dtype=np.float32)

    def intersect_horizontal(s, e, y_edge):
        s = np.asarray(s, dtype=np.float32)
        e = np.asarray(e, dtype=np.float32)
        dy = float(e[1] - s[1])
        if abs(dy) <= 1e-6:
            return np.asarray([float(s[0]), float(y_edge)], dtype=np.float32)
        t = (float(y_edge) - float(s[1])) / dy
        return np.asarray([float(s[0] + t * (e[0] - s[0])), float(y_edge)], dtype=np.float32)

    def clip_against(subject: List[np.ndarray], inside_fn, intersect_fn) -> List[np.ndarray]:
        if not subject:
            return []
        output: List[np.ndarray] = []
        prev = subject[-1]
        prev_inside = bool(inside_fn(prev))
        for curr in subject:
            curr_inside = bool(inside_fn(curr))
            if curr_inside:
                if not prev_inside:
                    output.append(np.asarray(intersect_fn(prev, curr), dtype=np.float32))
                output.append(np.asarray(curr, dtype=np.float32))
            elif prev_inside:
                output.append(np.asarray(intersect_fn(prev, curr), dtype=np.float32))
            prev = curr
            prev_inside = curr_inside
        return output

    subject = [np.asarray(p, dtype=np.float32) for p in poly]
    subject = clip_against(subject, inside_left, lambda s, e: intersect_vertical(s, e, x_min))
    subject = clip_against(subject, inside_right, lambda s, e: intersect_vertical(s, e, x_max))
    subject = clip_against(subject, inside_top, lambda s, e: intersect_horizontal(s, e, y_min))
    subject = clip_against(subject, inside_bottom, lambda s, e: intersect_horizontal(s, e, y_max))
    if len(subject) < 3:
        return []
    ring = ensure_closed_ring(dedup_points(np.asarray(subject, dtype=np.float32)))
    if ring.shape[0] < 4:
        return []
    return [ring.astype(np.float32)]

#把整图里的全局真值 global_lines，按一个全局矩形 rect_global 裁成当前 patch 负责的几何片段
# 并做重采样、cut 判定、去重和排序。
def build_patch_segments_global(
    global_lines: Sequence[Dict],
    review_mask: Optional[np.ndarray],
    rect_global: Tuple[float, float, float, float],
    resample_step_px: float,
    boundary_tol_px: float,
) -> List[Dict]:
    out: List[Dict] = []
    clipped_polygon_rings: List[np.ndarray] = []
    line_features: List[Dict] = []
    #第一轮循环：先把 polygon 和 line 分流
    for line in global_lines:
        #如果它是 polygon，先把 polygon 变成闭环 ring，如果它不是 polygon，就先收起来
        if str(line.get("geometry_type", "line")) == "polygon":
            #检查 ring 是否合法
            source_points = ensure_closed_ring(np.asarray(line["points_global"], dtype=np.float32))
            if source_points.ndim != 2 or source_points.shape[0] < 4:
                continue
            #把 polygon 裁到当前 patch 矩形里
            for clipped_ring in clip_polygon_ring_to_rect(source_points, rect_global):
                #对每个裁出来的 ring 再清洗一次
                # 这里转float32是因为还没有彻底转成int来作为数据集喂给大模型，在做成数据集之前都要用浮点数保证精度
                ring = ensure_closed_ring(np.asarray(clipped_ring, dtype=np.float32))
                if ring.ndim != 2 or ring.shape[0] < 4:
                    continue
                #记录 polygon 结果
                clipped_polygon_rings.append(ring.astype(np.float32))
                out.append(
                    {
                        "category": str(line["category"]),
                        "geometry_type": "polygon",
                        "points_global": ring.astype(np.float32),
                        "start_type": "closed",
                        "end_type": "closed",
                    }
                )
        else:
            #如果它不是 polygon，就先收起来
            line_features.append(dict(line))
    #第二轮循环：处理所有 line
    for line in line_features:
        #取出原始全局点列，这条 line 还是整图坐标
        source_points = np.asarray(line["points_global"], dtype=np.float32)
        #按 rect_global 裁 line，一条 line 裁到矩形里后，可能变成多段 piece，比如一条折线穿进矩形、出去、又进来，可能产生多个段，所以这里也要逐段处理
        for clipped_piece in clip_polyline_to_rect(source_points, rect_global):
            clipped_arr = np.asarray(clipped_piece, dtype=np.float32)
            # 把 piece 转成数组并检查合法性
            if clipped_arr.ndim != 2 or clipped_arr.shape[0] < 2:
                continue
            #判断这个 piece 的首尾是不是“裁切产生的”
            patch_cut_start, patch_cut_end = _line_piece_cut_flags_after_clip(
                source_points=source_points,
                clipped_points=clipped_arr,
                cut_start=False,
                cut_end=False,
            )
            #把当前 piece 暂存到 piece
            piece = clipped_arr.astype(np.float32)
            if piece.ndim != 2 or piece.shape[0] < 2:
                continue
            #如果启用了重采样，就重采样
            if float(resample_step_px) > 0.0:
                piece = resample_polyline_preserve_remainder(piece, step_px=resample_step_px)
            if piece.ndim != 2 or piece.shape[0] < 2:
                continue
            #判断重采样后的首尾点是否贴边界，因为有时候即便 _line_piece_cut_flags_after_clip(...) 没明确判成 cut，但端点贴着边界，本质上仍然应该被看成 patch cut
            start_side = point_boundary_side(piece[0], rect_global, boundary_tol_px)
            end_side = point_boundary_side(piece[-1], rect_global, boundary_tol_px)
            #生成 start_type / end_type
            start_type = "cut" if bool(patch_cut_start) or start_side is not None else "start"
            end_type = "cut" if bool(patch_cut_end) or end_side is not None else "end"
            #统一线方向
            piece, start_type, end_type = canonicalize_line_direction(piece, start_type=start_type, end_type=end_type)
            out.append(
                {
                    "category": str(line["category"]),
                    "points_global": piece.astype(np.float32),
                    "start_type": str(start_type),
                    "end_type": str(end_type),
                }
            )
    normalized: List[Dict] = []
    #第三轮：对所有结果再次做统一清洗
    for segment in out:
        #先复制当前 segment
        copied = dict(segment)
        pts = np.asarray(segment["points_global"], dtype=np.float32)
        #如果它是 polygon
        if str(segment.get("geometry_type", "line")) == "polygon":
            #去掉相邻重复点，再次闭环，检查点数是否合法
            ring = ensure_closed_ring(dedup_points(pts))
            if ring.ndim != 2 or ring.shape[0] < 4:
                continue
            copied["points_global"] = ring.astype(np.float32)
        else:
            #去掉相邻重复点，检查点数是否合法
            pts = dedup_points(pts).astype(np.float32)
            if pts.ndim != 2 or pts.shape[0] < 2:
                continue
            copied["points_global"] = pts.astype(np.float32)
        normalized.append(copied)
        #最后排序并返回
    return sort_lines(normalized)


def build_patch_target_lines_quantized(patch_segments_global: Sequence[Dict], patch: Dict) -> List[Dict]:
    crop_box = patch["crop_box"]
    offset = np.asarray([crop_box["x_min"], crop_box["y_min"]], dtype=np.float32)[None, :]
    patch_size = int(crop_box["x_max"] - crop_box["x_min"])
    out: List[Dict] = []
    for segment in patch_segments_global:
        local = np.asarray(segment["points_global"], dtype=np.float32) - offset
        if str(segment.get("geometry_type", "line")) == "polygon":
            points_json = simplify_for_json(ensure_closed_ring(local), patch_size=patch_size)
            if len(points_json) >= 2 and points_json[0] != points_json[-1]:
                points_json.append(list(points_json[0]))
            if len(points_json) < 4:
                continue
        else:
            points_json = simplify_for_json(local, patch_size=patch_size)
            if len(points_json) < 2:
                continue
        if len(points_json) < 2:
            continue
        out.append(
            {
                "category": str(segment["category"]),
                "start_type": str(segment["start_type"]),
                "end_type": str(segment["end_type"]),
                "geometry_type": str(segment.get("geometry_type", "line")),
                "points": points_json,
            }
        )
    return sort_lines(out)


def build_patch_target_lines_float(patch_segments_global: Sequence[Dict], patch: Dict) -> List[Dict]:
    crop_box = patch["crop_box"]
    offset = np.asarray([crop_box["x_min"], crop_box["y_min"]], dtype=np.float32)[None, :]
    patch_width, patch_height = patch_local_size(patch)
    out: List[Dict] = []
    for segment in patch_segments_global:
        local = np.asarray(segment["points_global"], dtype=np.float32) - offset
        local = clamp_points_float_rect(local, patch_width=patch_width, patch_height=patch_height)
        if str(segment.get("geometry_type", "line")) == "polygon":
            local = ensure_closed_ring(local)
            if local.ndim != 2 or local.shape[0] < 4:
                continue
        else:
            if local.ndim != 2 or local.shape[0] < 2:
                continue
        out.append(
            {
                "category": str(segment["category"]),
                "start_type": str(segment["start_type"]),
                "end_type": str(segment["end_type"]),
                "geometry_type": str(segment.get("geometry_type", "line")),
                "points": [[float(x), float(y)] for x, y in local.tolist()],
                "coord_system": "pixel_local",
            }
        )
    return sort_lines(out)


def build_patch_target_lines(patch_segments_global: Sequence[Dict], patch: Dict) -> List[Dict]:
    return build_patch_target_lines_quantized(patch_segments_global=patch_segments_global, patch=patch)


def build_manifest_for_dataset(
    dataset_root: Path,
    splits: Sequence[str],
    image_relpath: str,
    mask_relpath: str,
    lane_relpath: str,
    intersection_relpath: str,
    mask_threshold: int,
    tile_size_px: int,
    overlap_px: int,
    keep_margin_px: int,
    review_crop_pad_px: int,
    tile_min_mask_ratio: float,
    tile_min_mask_pixels: int,
    tile_max_per_sample: int,
    search_within_review_bbox: bool,
    fallback_to_all_if_empty: bool,
    max_samples_per_split: int,
    shard_index: int = 0,
    num_shards: int = 1,
    split_roots: Optional[Dict[str, Path]] = None,
) -> List[Dict]:
    families: List[Dict] = []
    shard_index = max(0, int(shard_index))
    num_shards = max(1, int(num_shards))
    #如果分片索引大于分片数量则报错，这里的分片是多台电脑分工做数据集，但是弃用了
    if shard_index >= num_shards:
        raise ValueError(f"Invalid shard config: shard_index={shard_index} num_shards={num_shards}")
    #遍历每个数据集
    for split in splits:
        #数据集路径赋值
        explicit_root = None if split_roots is None else split_roots.get(str(split))
        split_root = Path(explicit_root).resolve() if explicit_root is not None else dataset_root / str(split)
        #找不到数据集报错
        if not split_root.is_dir():
            print(f"[Manifest] skip split={split} reason=missing_dir path={split_root}", flush=True)
            continue
        sample_dirs = [path for path in sorted(split_root.iterdir()) if path.is_dir()]
        if int(max_samples_per_split) > 0:
            sample_dirs = sample_dirs[: int(max_samples_per_split)]
        if num_shards > 1:
            sample_dirs = [path for idx, path in enumerate(sample_dirs) if idx % num_shards == shard_index]
        print(
            f"[Manifest] split={split} sample_count={len(sample_dirs)} root={split_root} "
            f"shard={shard_index + 1}/{num_shards}",
            flush=True,
        )
        split_family_count = 0
        split_patch_count = 0
        #遍历每个样本目录
        for sample_index, sample_dir in enumerate(sample_dirs, start=1):
            sample_id = str(sample_dir.name)
            print(f"[Manifest] split={split} sample={sample_index}/{len(sample_dirs)} sample_id={sample_id} stage=scan", flush=True)
            image_path = sample_dir / image_relpath
            mask_path = sample_dir / mask_relpath
            lane_path = sample_dir / lane_relpath
            intersection_path = sample_dir / intersection_relpath
            if not image_path.is_file():
                print(f"[Manifest] split={split} sample_id={sample_id} stage=skip reason=missing_image path={image_path}", flush=True)
                continue
            print(f"[Manifest] split={split} sample_id={sample_id} stage=read_meta", flush=True)
            #读取图像元信息和 mask
            raster_meta = read_raster_meta(image_path)
            review_mask = read_binary_mask(mask_path, threshold=mask_threshold) if mask_path.is_file() else None
            print(
                f"[Manifest] split={split} sample_id={sample_id} stage=mask "
                f"mask_present={bool(mask_path.is_file())} image_size=({raster_meta.width},{raster_meta.height})",
                flush=True,
            )
            #计算 mask 的 bounding box，也就是mask的最小外接框
            #这里传入的review_mask原本是三通道的tif
            #但是预处理中它只被读取了第一个通道的波段赋值给了review_mask，作为二维numpy数组
            review_bbox = compute_mask_bbox(review_mask) if review_mask is not None else None
            region_bbox = None
            #如果开启了只搜索审核区域则只在审核区域内裁切
            if bool(search_within_review_bbox) and review_bbox is not None:
                region_bbox = expand_bbox(review_bbox, pad_px=int(review_crop_pad_px), width=int(raster_meta.width), height=int(raster_meta.height))
            region = (
                (0, 0, int(raster_meta.width), int(raster_meta.height))
                if region_bbox is None
                else tuple(int(v) for v in region_bbox)
            )
            print(f"[Manifest] split={split} sample_id={sample_id} stage=tile_windows region_bbox={region}", flush=True)
            #计算每个patch的左上角与右下角坐标并返回
            tile_windows = generate_tile_windows(
                width=int(raster_meta.width),
                height=int(raster_meta.height),
                tile_size_px=int(tile_size_px),
                overlap_px=int(overlap_px),
                region_bbox=tuple(int(v) for v in region),
                keep_margin_px=int(keep_margin_px),
            )
            #给每个 patch 标上 mask 信息,用mask覆盖并计算mask覆盖率和像素数
            tile_windows = annotate_tile_windows_with_mask(tile_windows=tile_windows, mask=review_mask)
            print(
                f"[Manifest] split={split} sample_id={sample_id} stage=tile_windows "
                f"candidates={len(tile_windows)} tile_size={int(tile_size_px)} overlap={int(overlap_px)}",
                flush=True,
            )
            #如果生成的patch数为0则直接全图生成
            if len(tile_windows) == 0 and bool(fallback_to_all_if_empty):
                fallback_region = (0, 0, int(raster_meta.width), int(raster_meta.height))
                tile_windows = generate_tile_windows(
                    width=int(raster_meta.width),
                    height=int(raster_meta.height),
                    tile_size_px=int(tile_size_px),
                    overlap_px=int(overlap_px),
                    region_bbox=fallback_region,
                    keep_margin_px=int(keep_margin_px),
                )
                tile_windows = annotate_tile_windows_with_mask(tile_windows=tile_windows, mask=review_mask)
                region = fallback_region

            selected_windows = list(tile_windows)
            #统计唯一的 x 和 y，用来构造网格行列号，让patch知道自己在第几行几列
            unique_xs = sorted({int(window.x0) for window in selected_windows})
            unique_ys = sorted({int(window.y0) for window in selected_windows})
            col_map = {int(x0): idx for idx, x0 in enumerate(unique_xs)}
            row_map = {int(y0): idx for idx, y0 in enumerate(unique_ys)}
            patches: List[Dict] = []
            #逐个 window 构造 patch 字典
            for patch_id, window in enumerate(selected_windows):
                x0, y0, x1, y1 = window.bbox
                keep_x0, keep_y0, keep_x1, keep_y1 = window.keep_bbox
                patch = {
                    "patch_id": int(patch_id),
                    "row": int(row_map[int(y0)]),
                    "col": int(col_map[int(x0)]),
                    "center_x": int(round(0.5 * float(x0 + x1))),
                    "center_y": int(round(0.5 * float(y0 + y1))),
                    "crop_box": {
                        "x_min": int(x0),
                        "y_min": int(y0),
                        "x_max": int(x1),
                        "y_max": int(y1),
                        "center_x": int(round(0.5 * float(x0 + x1))),
                        "center_y": int(round(0.5 * float(y0 + y1))),
                    },
                    "keep_box": {
                        "x_min": int(keep_x0),
                        "y_min": int(keep_y0),
                        "x_max": int(keep_x1),
                        "y_max": int(keep_y1),
                    },
                    "mask_ratio": float(window.mask_ratio),
                    "mask_pixels": int(window.mask_pixels),
                }
                patches.append(patch)
             #给 patch 分配 ownership keep boxes，但是因为我设置了overlap和margimn为0，这里的keepbox就是patch本身，grid就是patch网格   
            patches = assign_family_ownership_keep_boxes(patches=patches, grid_size=max(1, len(unique_xs)))
            #生成审计信息
            tile_audits: List[Dict] = []
            for patch in patches:
                x0 = int(patch["crop_box"]["x_min"])
                y0 = int(patch["crop_box"]["y_min"])
                x1 = int(patch["crop_box"]["x_max"])
                y1 = int(patch["crop_box"]["y_max"])
                keep_x0 = int(patch["keep_box"]["x_min"])
                keep_y0 = int(patch["keep_box"]["y_min"])
                keep_x1 = int(patch["keep_box"]["x_max"])
                keep_y1 = int(patch["keep_box"]["y_max"])
                tile_audits.append(
                    {
                        "candidate_index": int(patch["patch_id"]),#该patch的编号
                        "selected": True,#是否被丢弃
                        "reason": "selected",
                        "bbox": [int(x0), int(y0), int(x1), int(y1)],
                        "keep_bbox": [int(keep_x0), int(keep_y0), int(keep_x1), int(keep_y1)],
                        "mask_ratio": float(window.mask_ratio),
                        "mask_pixels": int(window.mask_pixels),
                    }
                )

            #为该样本生成family，即为“这一整张图怎么被切、切成了哪些 patch”的总描述对象
            split_family_count += 1
            families.append(
                {
                    "family_id": f"{sample_id}__geo_current_sw",
                    "split": str(split),
                    "source_sample_id": sample_id,
                    "source_image": image_path.name,
                    "source_image_path": str(image_path),
                    "source_mask_path": str(mask_path) if mask_path.is_file() else "",
                    "source_lane_path": str(lane_path) if lane_path.is_file() else "",
                    "source_intersection_path": str(intersection_path) if intersection_path.is_file() else "",
                    "image_size": [int(raster_meta.width), int(raster_meta.height)],
                    "crop_size": int(tile_size_px),
                    "paper_grid": {},
                    "tiling": {
                        "tile_size_px": int(tile_size_px),
                        "overlap_px": int(overlap_px),
                        "keep_margin_px": int(keep_margin_px),
                        "review_crop_pad_px": int(review_crop_pad_px),
                        "search_within_review_bbox": bool(search_within_review_bbox),
                        "tile_min_mask_ratio": float(tile_min_mask_ratio),
                        "tile_min_mask_pixels": int(tile_min_mask_pixels),
                        "tile_max_per_sample": int(tile_max_per_sample),
                        "row_count": int(len(unique_ys)),
                        "col_count": int(len(unique_xs)),
                    },
                    "crop_bbox": [int(v) for v in region],
                    "patches": patches,
                    "tile_audits": tile_audits,
                }
            )
            split_patch_count += len(patches)
            print(
                f"[Manifest] split={split} sample_id={sample_id} stage=done "
                f"family_index={split_family_count} patch_count={len(patches)} total_split_patches={split_patch_count}",
                flush=True,
            )
        print(
            f"[Manifest] split={split} completed families={split_family_count} patches={split_patch_count}",
            flush=True,
        )
        #返回family
    return families


def load_family_raster_and_mask(family: Dict, band_indices: Sequence[int], mask_threshold: int) -> Tuple[np.ndarray, RasterMeta, Optional[np.ndarray]]:
    #读取原始图像
    image_hwc, raster_meta = read_rgb_geotiff(Path(family["source_image_path"]).resolve(), band_indices=band_indices)
    mask_path = str(family.get("source_mask_path", "")).strip()
    review_mask = read_binary_mask(Path(mask_path), threshold=mask_threshold) if mask_path else None
    #用 mask 把图像外部置黑
    if review_mask is not None and image_hwc.ndim == 3:
        image_hwc = image_hwc.copy()
        #review_mask <= 0先生成一个布尔 mask
        #用这个二维的布尔数组去选图像中的像素，最后置0
        image_hwc[review_mask <= 0] = 0.0
    return image_hwc, raster_meta, review_mask

def family_global_lines(
    family: Dict,
    raster_meta: RasterMeta,
    include_lane: bool = True,
    include_intersection: bool = True,
) -> List[Dict]:
    #根据一个 family，把这张原始大图对应的 Lane.geojson 和 Intersection.geojson 读出来
    image_path = Path(family["source_image_path"]).resolve()
    sample_dir = image_path.parents[1]
    lane_path = Path(str(family.get("source_lane_path", "")).strip()) if str(family.get("source_lane_path", "")).strip() else sample_dir / DEFAULT_LANE_RELPATH
    intersection_path = Path(str(family.get("source_intersection_path", "")).strip()) if str(family.get("source_intersection_path", "")).strip() else sample_dir / DEFAULT_INTERSECTION_RELPATH
    lane_rel = str(lane_path.resolve().relative_to(sample_dir))
    intersection_rel = str(intersection_path.resolve().relative_to(sample_dir))
    #这里调用的函数才是核心
    return load_sample_global_lines(
        sample_dir=sample_dir,
        raster_meta=raster_meta,
        lane_relpath=lane_rel,
        intersection_relpath=intersection_rel,
        include_lane=bool(include_lane),
        include_intersection=bool(include_intersection),
    )


def build_patch_image(raw_image_hwc: np.ndarray, patch: Dict) -> Image.Image:
    crop_box = patch["crop_box"]
    crop = raw_image_hwc[
        int(crop_box["y_min"]) : int(crop_box["y_max"]),
        int(crop_box["x_min"]) : int(crop_box["x_max"]),
    ]
    return Image.fromarray(np.asarray(np.clip(crop, 0.0, 255.0), dtype=np.uint8))


def build_owned_segments_by_patch(
    family: Dict,
    global_lines: Sequence[Dict],
    review_mask: Optional[np.ndarray],
    resample_step_px: float,
    boundary_tol_px: float,
) -> Dict[int, List[Dict]]:
    #初始化输出字典
    out: Dict[int, List[Dict]] = {}
    #按 patch_id 排序遍历 patch
    for patch in sorted(list(family["patches"]), key=lambda item: int(item["patch_id"])):
        #取出当前 patch 的 keep_box
        keep_box = patch["keep_box"]
        #组装成矩形元组
        rect_global = (
            float(keep_box["x_min"]),
            float(keep_box["y_min"]),
            float(keep_box["x_max"]),
            float(keep_box["y_max"]),
        )
        #生成这个 patch 拥有的真值片段
        out[int(patch["patch_id"])] = build_patch_segments_global(
            global_lines=global_lines,
            review_mask=review_mask,
            rect_global=rect_global,
            resample_step_px=float(resample_step_px),
            boundary_tol_px=float(boundary_tol_px),
        )
    return out


def build_full_segments_for_patch(
    patch: Dict,
    global_lines: Sequence[Dict],
    review_mask: Optional[np.ndarray],
    resample_step_px: float,
    boundary_tol_px: float,
) -> List[Dict]:
    crop_box = patch["crop_box"]
    rect_global = (
        float(crop_box["x_min"]),
        float(crop_box["y_min"]),
        float(crop_box["x_max"]),
        float(crop_box["y_max"]),
    )
    return build_patch_segments_global(
        global_lines=global_lines,
        review_mask=review_mask,
        rect_global=rect_global,
        resample_step_px=float(resample_step_px),
        boundary_tol_px=float(boundary_tol_px),
    )


def extract_state_lines(
    patch: Dict,
    family: Dict,
    owned_segments_by_patch: Dict[int, List[Dict]],
    trace_points: int,
    boundary_tol_px: float,
) -> List[Dict]:
    patches = sorted(list(family["patches"]), key=lambda item: int(item["patch_id"]))
    patch_map = {(int(item["row"]), int(item["col"])): item for item in patches}
    row = int(patch["row"])
    col = int(patch["col"])
    crop_box = patch["crop_box"]
    keep_box = patch["keep_box"]
    crop_rect_global = (
        float(crop_box["x_min"]),
        float(crop_box["y_min"]),
        float(crop_box["x_max"]),
        float(crop_box["y_max"]),
    )
    local_rect = (
        float(keep_box["x_min"] - crop_box["x_min"]),
        float(keep_box["y_min"] - crop_box["y_min"]),
        float(keep_box["x_max"] - crop_box["x_min"]),
        float(keep_box["y_max"] - crop_box["y_min"]),
    )
    patch_size = int(crop_box["x_max"] - crop_box["x_min"])
    offset = np.asarray([crop_box["x_min"], crop_box["y_min"]], dtype=np.float32)[None, :]
    neighbors = []
    if (row, col - 1) in patch_map:
        neighbors.append((patch_map[(row, col - 1)], "left"))
    if (row - 1, col) in patch_map:
        neighbors.append((patch_map[(row - 1, col)], "top"))
    out: List[Dict] = []
    for neighbor_patch, handoff_side in neighbors:
        for segment in owned_segments_by_patch.get(int(neighbor_patch["patch_id"]), []):
            if str(segment.get("geometry_type", "line")) != "line":
                continue
            for piece in clip_polyline_to_rect(np.asarray(segment["points_global"], dtype=np.float32), crop_rect_global):
                local = np.asarray(piece, dtype=np.float32) - offset
                if local.ndim != 2 or local.shape[0] < 2:
                    continue
                local = clamp_points_float(local, patch_size=patch_size)
                boundary_idx = None
                if point_boundary_side(local[0], local_rect, boundary_tol_px) == handoff_side:
                    boundary_idx = 0
                elif point_boundary_side(local[-1], local_rect, boundary_tol_px) == handoff_side:
                    boundary_idx = -1
                if boundary_idx is None:
                    continue
                if boundary_idx == -1:
                    local = local[::-1].copy()
                trace = local[: max(2, int(trace_points))]
                if trace.ndim != 2 or trace.shape[0] < 2:
                    continue
                out.append(
                    {
                        "source_patch": int(neighbor_patch["patch_id"]),
                        "category": str(segment["category"]),
                        "start_type": "cut",
                        "end_type": "cut",
                        "points": [[float(x), float(y)] for x, y in trace.tolist()],
                    }
                )
    seen = set()
    deduped: List[Dict] = []
    for line in sort_lines(out):
        key = (int(line["source_patch"]), tuple((round(float(p[0]), 3), round(float(p[1]), 3)) for p in line["points"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(line)
    return deduped


def build_patch_only_record(
    image_rel_path: str,
    target_lines: Sequence[Dict],
    sample_id: str,
    system_prompt: str,
    prompt_template: str,
) -> Dict:
    target_json = json.dumps({"lines": list(target_lines)}, ensure_ascii=False, separators=(",", ":"))
    messages: List[Dict] = []
    if str(system_prompt).strip():
        messages.append({"role": "system", "content": str(system_prompt).strip()})
    messages.append({"role": "user", "content": str(prompt_template)})
    messages.append({"role": "assistant", "content": target_json})
    return {"id": str(sample_id), "messages": messages, "images": [str(image_rel_path).replace("\\", "/")]}


def build_state_record(
    image_rel_path: str,
    state_lines: Sequence[Dict],
    target_lines: Sequence[Dict],
    sample_id: str,
    system_prompt: str,
    prompt_template: str,
) -> Dict:
    state_json = json.dumps({"lines": list(state_lines)}, ensure_ascii=False, separators=(",", ":"))
    target_json = json.dumps({"lines": list(target_lines)}, ensure_ascii=False, separators=(",", ":"))
    messages: List[Dict] = []
    if str(system_prompt).strip():
        messages.append({"role": "system", "content": str(system_prompt).strip()})
    messages.append({"role": "user", "content": str(prompt_template).format(state_json=state_json)})
    messages.append({"role": "assistant", "content": target_json})
    return {"id": str(sample_id), "messages": messages, "images": [str(image_rel_path).replace("\\", "/")]}


def parse_generated_json(text: str) -> Tuple[Optional[Dict], str]:
    raw = str(text or "").strip()
    if not raw:
        return None, ""
    start = raw.find("{")
    if start < 0:
        return None, raw
    depth = 0
    in_string = False
    escape = False
    end = None
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break
    if end is None:
        return None, raw
    cleaned = raw[start:end]
    try:
        return json.loads(cleaned), cleaned
    except Exception:
        return None, cleaned


def sanitize_pred_lines(pred_lines: Sequence[Dict], patch_size: int) -> List[Dict]:
    out: List[Dict] = []
    for line in pred_lines:
        arr = np.asarray(line.get("points", []), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] != 2:
            continue
        points = simplify_for_json(arr, patch_size=patch_size)
        if len(points) < 2:
            continue
        start_type = str(line.get("start_type", "start"))
        end_type = str(line.get("end_type", "end"))
        if start_type not in {"start", "cut"}:
            start_type = "start"
        if end_type not in {"end", "cut"}:
            end_type = "end"
        out.append(
            {
                "category": str(line.get("category", "lane_line")),
                "start_type": start_type,
                "end_type": end_type,
                "geometry_type": str(line.get("geometry_type", "line")),
                "points": points,
            }
        )
    return sort_lines(out)


def local_lines_to_global(pred_lines: Sequence[Dict], patch: Dict) -> List[Dict]:
    crop_box = patch["crop_box"]
    offset = np.asarray([crop_box["x_min"], crop_box["y_min"]], dtype=np.float32)[None, :]
    out: List[Dict] = []
    for line in pred_lines:
        arr = np.asarray(line.get("points", []), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 2:
            continue
        if str(line.get("coord_system", "")).strip().lower() == "uv":
            arr = uv_points_to_local(arr, patch=patch)
        out.append(
            {
                "category": str(line.get("category", "lane_line")),
                "start_type": str(line.get("start_type", "start")),
                "end_type": str(line.get("end_type", "end")),
                "geometry_type": str(line.get("geometry_type", "line")),
                "points_global": dedup_points(arr + offset),
            }
        )
    return out


def apply_state_mode(
    raw_state_lines: Sequence[Dict],
    state_mode: str,
    patch_size: int,
    weak_trace_points: int,
    state_line_dropout: float,
    state_point_jitter_px: float,
    state_truncate_prob: float,
    rng: np.random.Generator,
) -> List[Dict]:
    if state_mode in {"empty", "no_state"}:
        return []
    if state_mode in {"full", "full_state"}:
        return [dict(line) for line in raw_state_lines]

    weak_lines: List[Dict] = []
    keep_prob = 1.0 - max(0.0, min(1.0, float(state_line_dropout)))
    max_trace_points = max(2, int(weak_trace_points))
    truncate_prob = max(0.0, min(1.0, float(state_truncate_prob)))
    for line in raw_state_lines:
        if float(rng.random()) > keep_prob:
            continue
        arr = np.asarray(line.get("points", []), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 2:
            continue
        if float(rng.random()) < truncate_prob and arr.shape[0] > 2:
            new_len = int(rng.integers(2, min(arr.shape[0], max_trace_points) + 1))
        else:
            new_len = min(arr.shape[0], max_trace_points)
        truncated = arr[:new_len].astype(np.float32)
        if float(state_point_jitter_px) > 0.0:
            noise = rng.uniform(
                low=-float(state_point_jitter_px),
                high=float(state_point_jitter_px),
                size=truncated.shape,
            ).astype(np.float32)
            truncated = clamp_points_float(truncated + noise, patch_size=int(patch_size))
        if truncated.ndim != 2 or truncated.shape[0] < 2:
            continue
        weak_lines.append(
            {
                "source_patch": int(line.get("source_patch", -1)),
                "category": str(line.get("category", "road")),
                "start_type": str(line.get("start_type", "cut")),
                "end_type": str(line.get("end_type", "cut")),
                "points": [[float(x), float(y)] for x, y in truncated.tolist()],
            }
        )
    if weak_lines:
        return sort_lines(weak_lines)
    first = next((line for line in raw_state_lines if len(line.get("points", [])) >= 2), None)
    if first is None:
        return []
    arr = np.asarray(first.get("points", []), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return []
    fallback = arr[: max(2, int(weak_trace_points))].astype(np.float32)
    if float(state_point_jitter_px) > 0.0:
        noise = rng.uniform(
            low=-float(state_point_jitter_px),
            high=float(state_point_jitter_px),
            size=fallback.shape,
        ).astype(np.float32)
        fallback = clamp_points_float(fallback + noise, patch_size=int(patch_size))
    if fallback.ndim != 2 or fallback.shape[0] < 2:
        return []
    return [
        {
            "source_patch": int(first.get("source_patch", -1)),
            "category": str(first.get("category", "road")),
            "start_type": str(first.get("start_type", "cut")),
            "end_type": str(first.get("end_type", "cut")),
            "points": [[float(x), float(y)] for x, y in fallback.tolist()],
        }
    ]
