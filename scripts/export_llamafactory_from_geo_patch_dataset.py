import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from pyproj import CRS, Transformer
from rasterio import open as rasterio_open
from rasterio.transform import Affine


DEFAULT_PROMPT_TEMPLATE = """<image>
Please construct the complete road-line map in the current satellite patch."""

DEFAULT_SYSTEM_PROMPT = (
    "You are a road-map reconstruction assistant for satellite image patches.\n"
    "Predict the complete road-line map in the current patch from the satellite image.\n"
    "Return only valid JSON in the required schema.\n"
    "Do not output markdown fences or extra explanation.\n"
    "Keep all coordinates in the patch-local pixel coordinate system."
)

DEFAULT_DATASET_ROOT = "/dataset/zsy/dataset-extracted"
DEFAULT_IMAGE_RELPATH = "patch_tif/0.tif"
DEFAULT_LANE_RELPATH = "label_check_crop/Lane.geojson"
DEFAULT_OUTPUT_CATEGORY = "road"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export current tif+GeoJSON patch dataset into v1.0 LLaMAFactory ShareGPT format."
    )
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"])
    parser.add_argument("--image-relpath", type=str, default=DEFAULT_IMAGE_RELPATH)
    parser.add_argument("--lane-relpath", type=str, default=DEFAULT_LANE_RELPATH)
    parser.add_argument("--band-indices", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--output-category", type=str, default=DEFAULT_OUTPUT_CATEGORY)
    parser.add_argument("--boundary-tol-px", type=float, default=2.5)
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument("--use-system-prompt", action="store_true")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--dataset-prefix", type=str, default="geomapgen_geo_patch_v1")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    ensure_dir(path.parent)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def detect_geojson_crs(geojson_dict: Dict) -> str:
    if not isinstance(geojson_dict, dict):
        return "OGC:CRS84"
    crs = geojson_dict.get("crs")
    if isinstance(crs, dict):
        props = crs.get("properties", {})
        name = props.get("name")
        if isinstance(name, str) and name.strip():
            return str(name).strip()
    return "OGC:CRS84"


def build_transformer(src_crs: str, dst_crs: str) -> Transformer:
    return Transformer.from_crs(CRS.from_user_input(src_crs), CRS.from_user_input(dst_crs), always_xy=True)


def project_coords(coordinates, transformer: Transformer) -> np.ndarray:
    points = []
    for value in coordinates:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            continue
        x, y = transformer.transform(float(value[0]), float(value[1]))
        points.append([float(x), float(y)])
    return np.asarray(points, dtype=np.float32)


def world_to_pixel(points_world: np.ndarray, affine: Affine) -> np.ndarray:
    if points_world.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    inv = ~affine
    cols = []
    rows = []
    for x, y in points_world:
        col, row = inv * (float(x), float(y))
        cols.append(float(col))
        rows.append(float(row))
    return np.stack([cols, rows], axis=-1).astype(np.float32)


def read_rgb_geotiff(path: Path, band_indices: Sequence[int]) -> Tuple[np.ndarray, RasterMeta]:
    with rasterio_open(path) as ds:
        arr = ds.read(indexes=[int(x) for x in band_indices])
        image = np.transpose(arr, (1, 2, 0)).astype(np.float32)
        meta = RasterMeta(
            path=str(path),
            width=int(ds.width),
            height=int(ds.height),
            crs=str(ds.crs) if ds.crs is not None else "",
            transform=[float(x) for x in ds.transform],
        )
    return image, meta


def save_png(image_hwc: np.ndarray, path: Path) -> None:
    ensure_dir(path.parent)
    image_u8 = np.asarray(np.clip(image_hwc, 0.0, 255.0), dtype=np.uint8)
    Image.fromarray(image_u8).save(path)


def clamp_points(points_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32).copy()
    if pts.ndim != 2 or pts.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0.0, float(max(0, width - 1)))
    pts[:, 1] = np.clip(pts[:, 1], 0.0, float(max(0, height - 1)))
    return pts


def dedup_points(points_xy: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    out = [pts[0]]
    for idx in range(1, pts.shape[0]):
        if float(np.linalg.norm(pts[idx] - out[-1])) > float(eps):
            out.append(pts[idx])
    return np.asarray(out, dtype=np.float32)


def point_boundary_side(point_xy: np.ndarray, width: int, height: int, tol_px: float) -> str:
    x = float(point_xy[0])
    y = float(point_xy[1])
    if abs(x - 0.0) <= tol_px:
        return "cut"
    if abs(y - 0.0) <= tol_px:
        return "cut"
    if abs(x - float(width - 1)) <= tol_px:
        return "cut"
    if abs(y - float(height - 1)) <= tol_px:
        return "cut"
    return "start"


def simplify_for_json(points_xy: np.ndarray, width: int, height: int) -> List[List[int]]:
    pts = clamp_points(points_xy, width=width, height=height)
    pts = np.rint(pts).astype(np.int32)
    pts = dedup_points(pts.astype(np.float32)).astype(np.int32)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return []
    return [[int(x), int(y)] for x, y in pts.tolist()]


def lane_geojson_to_v1_lines(geojson_dict: Dict, raster_meta: RasterMeta, output_category: str, boundary_tol_px: float) -> List[Dict]:
    src_crs = detect_geojson_crs(geojson_dict)
    transformer = build_transformer(src_crs=src_crs, dst_crs=raster_meta.crs)
    lines: List[Dict] = []
    for feature in geojson_dict.get("features", []):
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry", {})
        if str(geometry.get("type", "")).strip().lower() != "linestring":
            continue
        world = project_coords(geometry.get("coordinates", []), transformer=transformer)
        pixel = world_to_pixel(world, affine=raster_meta.affine)
        points_json = simplify_for_json(pixel, width=raster_meta.width, height=raster_meta.height)
        if len(points_json) < 2:
            continue
        pts_np = np.asarray(points_json, dtype=np.float32)
        start_type = point_boundary_side(pts_np[0], raster_meta.width, raster_meta.height, boundary_tol_px)
        end_type = point_boundary_side(pts_np[-1], raster_meta.width, raster_meta.height, boundary_tol_px)
        lines.append(
            {
                "category": str(output_category),
                "start_type": str(start_type),
                "end_type": str("cut" if end_type == "cut" else "end"),
                "points": points_json,
            }
        )
    lines.sort(key=lambda item: (item.get("points", [[10**9, 10**9]])[0][1], item.get("points", [[10**9, 10**9]])[0][0]))
    return lines


def build_messages(target_lines: Sequence[Dict], sample_id: str, image_rel_path: str, use_system_prompt: bool, system_prompt: str) -> Dict:
    target_json = json.dumps({"lines": list(target_lines)}, ensure_ascii=False, separators=(",", ":"))
    messages: List[Dict] = []
    if bool(use_system_prompt) and str(system_prompt).strip():
        messages.append({"role": "system", "content": str(system_prompt).strip()})
    messages.append({"role": "user", "content": DEFAULT_PROMPT_TEMPLATE})
    messages.append({"role": "assistant", "content": target_json})
    return {
        "id": str(sample_id),
        "messages": messages,
        "images": [str(image_rel_path).replace("\\", "/")],
    }


def export_split(args: argparse.Namespace, split: str, output_root: Path) -> Dict[str, int]:
    split_root = Path(args.dataset_root) / split
    image_out_root = output_root / "images" / split
    rows: List[Dict] = []
    meta_rows: List[Dict] = []
    sample_dirs = [path for path in sorted(split_root.iterdir()) if path.is_dir()] if split_root.is_dir() else []
    if int(args.max_samples_per_split) > 0:
        sample_dirs = sample_dirs[: int(args.max_samples_per_split)]

    for sample_dir in sample_dirs:
        sample_id = str(sample_dir.name)
        image_path = sample_dir / str(args.image_relpath)
        lane_path = sample_dir / str(args.lane_relpath)
        if not image_path.is_file() or not lane_path.is_file():
            continue
        image_hwc, raster_meta = read_rgb_geotiff(image_path, band_indices=args.band_indices)
        lane_geojson = read_json(lane_path)
        target_lines = lane_geojson_to_v1_lines(
            geojson_dict=lane_geojson,
            raster_meta=raster_meta,
            output_category=str(args.output_category),
            boundary_tol_px=float(args.boundary_tol_px),
        )
        image_rel = Path("images") / split / f"{sample_id}.png"
        save_png(image_hwc=image_hwc, path=output_root / image_rel)
        rows.append(
            build_messages(
                target_lines=target_lines,
                sample_id=sample_id,
                image_rel_path=str(image_rel),
                use_system_prompt=bool(args.use_system_prompt),
                system_prompt=str(args.system_prompt),
            )
        )
        meta_rows.append(
            {
                "id": sample_id,
                "split": str(split),
                "image_path": str(image_path),
                "lane_path": str(lane_path),
                "width": int(raster_meta.width),
                "height": int(raster_meta.height),
                "line_count": int(len(target_lines)),
            }
        )

    count_rows = write_jsonl(output_root / f"{split}.jsonl", rows)
    count_meta = write_jsonl(output_root / f"meta_{split}.jsonl", meta_rows)
    return {
        "row_count": int(count_rows),
        "meta_count": int(count_meta),
    }


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    ensure_dir(output_root)

    summary = {
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "output_root": str(output_root),
        "splits": {},
        "notes": [
            "Adapter currently exports only Lane.geojson as v1.0 road-line targets.",
            "Intersection.geojson is not used because the v1.0 patch-only baseline is line-map oriented.",
        ],
    }
    dataset_info = {}
    prefix = str(args.dataset_prefix).strip() or "geomapgen_geo_patch_v1"

    for split in args.splits:
        export_info = export_split(args=args, split=str(split), output_root=output_root)
        summary["splits"][str(split)] = export_info
        dataset_name = f"{prefix}_{split}"
        dataset_info[dataset_name] = {
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

    with (output_root / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    with (output_root / "export_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
