import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rasterio import open as rasterio_open

from geo_current_dataset_v1_common import load_jsonl, uv_lines_to_local


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_manifest_map(path: Path) -> Dict[str, Dict]:
    return {str(row["family_id"]): row for row in load_jsonl(path)}


def infer_source_sample_id(family_id: str, row: Optional[Dict] = None, family: Optional[Dict] = None) -> str:
    if isinstance(row, dict):
        value = str(row.get("source_sample_id", "")).strip()
        if value:
            return value
    if isinstance(family, dict):
        value = str(family.get("source_sample_id", "")).strip()
        if value:
            return value
    fid = str(family_id).strip()
    for marker in ("__geo_current_", "__paper", "__family", "__"):
        if marker in fid:
            return fid.split(marker, 1)[0]
    return fid


def load_dataset_image_map(path: Path) -> Dict[str, str]:
    rows = load_jsonl(path)
    out: Dict[str, str] = {}
    for row in rows:
        image_list = row.get("images", [])
        if isinstance(image_list, list) and len(image_list) > 0:
            out[str(row.get("id"))] = str(image_list[0])
    return out


def pixel_to_world(points_xy: Iterable[Iterable[float]], transform) -> List[List[float]]:
    out: List[List[float]] = []
    for point in points_xy:
        x = float(point[0])
        y = float(point[1])
        world_x, world_y = transform * (x, y)
        out.append([float(world_x), float(world_y)])
    return out


def build_feature_collection(features: List[Dict], crs_name: str, name: str) -> Dict:
    return {
        "type": "FeatureCollection",
        "name": str(name),
        "crs": {
            "type": "name",
            "properties": {"name": str(crs_name)},
        },
        "features": features,
    }


def resolve_sample_raster_context(rows: List[Dict], family: Optional[Dict]) -> Dict:
    source_image_path = ""
    if isinstance(family, dict):
        source_image_path = str(family.get("source_image_path", "")).strip()
    if not source_image_path:
        for row in rows:
            candidate = str(row.get("source_image_path", "")).strip()
            if candidate:
                source_image_path = candidate
                break

    if source_image_path:
        source_path = Path(source_image_path).resolve()
        if source_path.is_file():
            with rasterio_open(source_path) as ds:
                return {
                    "width": int(ds.width),
                    "height": int(ds.height),
                    "profile": ds.profile.copy(),
                    "transform": ds.transform,
                    "crs_name": str(ds.crs) if ds.crs is not None else "urn:ogc:def:crs:OGC:1.3:CRS84",
                    "source_image_path": str(source_path),
                    "georeferenced": True,
                }

    width = 0
    height = 0
    for row in rows:
        crop_box = row.get("crop_box", {})
        width = max(width, int(crop_box.get("x_max", 0)))
        height = max(height, int(crop_box.get("y_max", 0)))
    return {
        "width": max(1, int(width)),
        "height": max(1, int(height)),
        "profile": None,
        "transform": None,
        "crs_name": "LOCAL_PIXEL",
        "source_image_path": str(source_image_path).strip(),
        "georeferenced": False,
    }


def _origin_sort_key(point_xy: Iterable[float]) -> Tuple[float, float, float]:
    x = float(point_xy[0])
    y = float(point_xy[1])
    return (x * x + y * y, y, x)


def _endpoint_label_prefix(endpoint_type: str) -> str:
    endpoint_type = str(endpoint_type).strip().lower()
    if endpoint_type == "start":
        return "s"
    if endpoint_type == "end":
        return "e"
    if endpoint_type == "cut":
        return "c"
    if endpoint_type:
        return endpoint_type[:1]
    return "p"


def annotate_patch_endpoint_order_labels(lines: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    line_refs: List[Dict] = []
    for line_idx, row in enumerate(lines):
        copied = dict(row)
        out.append(copied)
        if str(copied.get("geometry_type", "line")) == "polygon":
            continue
        local_points = copied.get("local_points", copied.get("points", []))
        if not isinstance(local_points, list) or len(local_points) < 2:
            continue
        start_type = str(copied.get("start_type", "")).strip() or "start"
        end_type = str(copied.get("end_type", "")).strip() or "end"
        start_point = local_points[0]
        end_point = local_points[-1]
        line_refs.append(
            {
                "line_idx": int(line_idx),
                "start_type": start_type,
                "end_type": end_type,
                "sort_point": min(
                    [start_point, end_point],
                    key=_origin_sort_key,
                ),
            }
        )
    ordered = sorted(line_refs, key=lambda item: _origin_sort_key(item["sort_point"]))
    for rank, ref in enumerate(ordered, start=1):
        start_prefix = _endpoint_label_prefix(str(ref["start_type"]))
        end_prefix = _endpoint_label_prefix(str(ref["end_type"]))
        out[int(ref["line_idx"])]["start_label"] = f"{start_prefix}{rank}"
        out[int(ref["line_idx"])]["end_label"] = f"{end_prefix}{rank}"
    return out


def _extract_line_identifier(row: Dict, fallback_rank: int) -> str:
    for key in ("line_id", "segment_id", "source_line_id", "source_segment_id", "id"):
        value = row.get(key)
        if value is None:
            continue
        value_str = str(value).strip()
        if value_str:
            return value_str
    return str(int(fallback_rank))


def annotate_patch_line_labels(lines: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    next_rank = 1
    for row in lines:
        copied = dict(row)
        out.append(copied)
        if str(copied.get("geometry_type", "line")) == "polygon":
            continue
        local_points = copied.get("local_points", copied.get("points", []))
        if not isinstance(local_points, list) or len(local_points) < 2:
            continue
        line_id = _extract_line_identifier(copied, fallback_rank=next_rank)
        copied["line_id"] = line_id
        copied["line_label"] = line_id
        next_rank += 1
    return out


def _line_color(category: str) -> Tuple[int, int, int]:
    if str(category) == "intersection_polygon":
        return (255, 180, 0)
    return (0, 255, 255)


def _line_color_compare(kind: str) -> Tuple[int, int, int]:
    if str(kind) == "float":
        return (80, 255, 80)
    return (255, 80, 255)


def _endpoint_color(endpoint_type: str) -> Tuple[int, int, int]:
    endpoint_type = str(endpoint_type).strip().lower()
    if endpoint_type == "cut":
        return (255, 60, 60)
    if endpoint_type == "end":
        return (255, 220, 0)
    return (0, 255, 0)


def _arrow_color() -> Tuple[int, int, int]:
    return (255, 220, 0)


def _distance_xy(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return math.hypot(dx, dy)


def _spread_endpoint_positions(
    visual_lines: List[Dict],
    cluster_dist_px: float = 14.0,
    base_offset_px: float = 14.0,
) -> Dict[Tuple[int, str], Tuple[float, float]]:
    endpoints: List[Dict] = []
    for line_idx, row in enumerate(visual_lines):
        if str(row.get("geometry_type", "line")) == "polygon":
            continue
        points = row.get("points", [])
        if not isinstance(points, list) or len(points) < 2:
            continue
        endpoints.append(
            {
                "line_idx": int(line_idx),
                "endpoint_key": "start",
                "point": (float(points[0][0]), float(points[0][1])),
            }
        )
        endpoints.append(
            {
                "line_idx": int(line_idx),
                "endpoint_key": "end",
                "point": (float(points[-1][0]), float(points[-1][1])),
            }
        )

    clusters: List[List[Dict]] = []
    for endpoint in endpoints:
        assigned = False
        for cluster in clusters:
            anchor = cluster[0]["point"]
            if _distance_xy(endpoint["point"], anchor) <= float(cluster_dist_px):
                cluster.append(endpoint)
                assigned = True
                break
        if not assigned:
            clusters.append([endpoint])

    out: Dict[Tuple[int, str], Tuple[float, float]] = {}
    for cluster in clusters:
        if len(cluster) == 1:
            item = cluster[0]
            out[(int(item["line_idx"]), str(item["endpoint_key"]))] = item["point"]
            continue
        ordered = sorted(
            cluster,
            key=lambda item: (
                float(item["point"][0]) * float(item["point"][0]) + float(item["point"][1]) * float(item["point"][1]),
                float(item["point"][1]),
                float(item["point"][0]),
            ),
        )
        count = len(ordered)
        angle_step = (2.0 * math.pi) / float(max(1, count))
        radius = float(base_offset_px) + max(0.0, float(count - 2) * 2.0)
        for idx, item in enumerate(ordered):
            angle = (math.pi / 2.0) + (float(idx) * angle_step)
            px = float(item["point"][0]) + radius * math.cos(angle)
            py = float(item["point"][1]) + radius * math.sin(angle)
            out[(int(item["line_idx"]), str(item["endpoint_key"]))] = (px, py)
    return out


def _select_arrow_segment(xy: List[Tuple[int, int]]) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    if len(xy) < 2:
        return None
    seg_lengths: List[float] = []
    total = 0.0
    for idx in range(len(xy) - 1):
        length = _distance_xy(xy[idx], xy[idx + 1])
        seg_lengths.append(length)
        total += length
    if total <= 1e-6:
        return None
    target = total * 0.6
    accum = 0.0
    for idx, length in enumerate(seg_lengths):
        next_accum = accum + length
        if length > 1.0 and target <= next_accum:
            return (
                (float(xy[idx][0]), float(xy[idx][1])),
                (float(xy[idx + 1][0]), float(xy[idx + 1][1])),
            )
        accum = next_accum
    for idx in range(len(seg_lengths) - 1, -1, -1):
        if seg_lengths[idx] > 1.0:
            return (
                (float(xy[idx][0]), float(xy[idx][1])),
                (float(xy[idx + 1][0]), float(xy[idx + 1][1])),
            )
    return None


def _draw_direction_arrow(draw: ImageDraw.ImageDraw, xy: List[Tuple[int, int]], color: Tuple[int, int, int]) -> None:
    segment = _select_arrow_segment(xy)
    if segment is None:
        return
    start_xy, end_xy = segment
    dx = float(end_xy[0]) - float(start_xy[0])
    dy = float(end_xy[1]) - float(start_xy[1])
    seg_len = math.hypot(dx, dy)
    if seg_len <= 1e-6:
        return
    ux = dx / seg_len
    uy = dy / seg_len
    arrow_tip = (
        float(start_xy[0]) + 0.72 * dx,
        float(start_xy[1]) + 0.72 * dy,
    )
    arrow_base = (
        float(arrow_tip[0]) - 10.0 * ux,
        float(arrow_tip[1]) - 10.0 * uy,
    )
    perp = (-uy, ux)
    wing1 = (
        float(arrow_base[0]) + 4.0 * perp[0],
        float(arrow_base[1]) + 4.0 * perp[1],
    )
    wing2 = (
        float(arrow_base[0]) - 4.0 * perp[0],
        float(arrow_base[1]) - 4.0 * perp[1],
    )
    draw.line((arrow_base, arrow_tip), fill=color, width=3)
    draw.polygon([arrow_tip, wing1, wing2], fill=color)


def _polyline_label_anchor(xy: List[Tuple[int, int]]) -> Tuple[float, float]:
    if not xy:
        return (0.0, 0.0)
    if len(xy) == 1:
        return (float(xy[0][0]), float(xy[0][1]))
    seg_lengths: List[float] = []
    total = 0.0
    for idx in range(len(xy) - 1):
        length = _distance_xy(xy[idx], xy[idx + 1])
        seg_lengths.append(length)
        total += length
    if total <= 1e-6:
        mid = xy[len(xy) // 2]
        return (float(mid[0]), float(mid[1]))
    target = total * 0.5
    accum = 0.0
    for idx, length in enumerate(seg_lengths):
        next_accum = accum + length
        if target <= next_accum and length > 1e-6:
            ratio = (target - accum) / length
            x = float(xy[idx][0]) + ratio * (float(xy[idx + 1][0]) - float(xy[idx][0]))
            y = float(xy[idx][1]) + ratio * (float(xy[idx + 1][1]) - float(xy[idx][1]))
            return (x, y)
        accum = next_accum
    last = xy[-1]
    return (float(last[0]), float(last[1]))


def build_overlay_image(
    canvas: np.ndarray,
    visual_lines: List[Dict],
    color_mode: str = "category",
    keep_boxes: List[Dict] | None = None,
    label_mode: str = "endpoint",
) -> Image.Image:
    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    endpoint_positions = _spread_endpoint_positions(visual_lines)

    for keep_box in list(keep_boxes or []):
        x_min = int(round(float(keep_box.get("x_min", 0))))
        y_min = int(round(float(keep_box.get("y_min", 0))))
        x_max = int(round(float(keep_box.get("x_max", 0))))
        y_max = int(round(float(keep_box.get("y_max", 0))))
        label = str(keep_box.get("label", "keep_box"))
        draw.rectangle((x_min, y_min, x_max, y_max), outline=(255, 255, 255), width=2)
        draw.text((x_min + 4, max(0, y_min - 12)), label, fill=(255, 255, 255), font=font)

    for line_idx, row in enumerate(visual_lines):
        points = row.get("points", [])
        if not isinstance(points, list) or len(points) < 2:
            continue
        color = _line_color(str(row.get("category", ""))) if color_mode == "category" else _line_color_compare(str(row.get("viz_kind", "quantized")))
        xy = [(int(round(pt[0])), int(round(pt[1]))) for pt in points]
        if str(row.get("geometry_type", "line")) == "polygon" and len(xy) >= 3:
            draw.polygon(xy, outline=color, width=3)
            first_xy = xy[0]
            radius = 5
            draw.ellipse(
                (first_xy[0] - radius, first_xy[1] - radius, first_xy[0] + radius, first_xy[1] + radius),
                fill=(255, 180, 0),
                outline=(0, 0, 0),
            )
            label = "polygon" if color_mode == "category" else str(row.get("viz_kind", "polygon"))
            draw.text((first_xy[0] + 6, first_xy[1] - 10), label, fill=color, font=font)
            continue
        draw.line(xy, fill=color, width=3)
        _draw_direction_arrow(draw, xy, _arrow_color())

        start_xy = xy[0]
        end_xy = xy[-1]
        start_type = str(row.get("start_type", "")).strip() or "start"
        end_type = str(row.get("end_type", "")).strip() or "end"
        start_display = endpoint_positions.get((int(line_idx), "start"), (float(start_xy[0]), float(start_xy[1])))
        end_display = endpoint_positions.get((int(line_idx), "end"), (float(end_xy[0]), float(end_xy[1])))
        if _distance_xy(start_display, start_xy) > 1.0:
            draw.line((start_xy, start_display), fill=_endpoint_color(start_type), width=2)
        if _distance_xy(end_display, end_xy) > 1.0:
            draw.line((end_xy, end_display), fill=_endpoint_color(end_type), width=2)

        start_r = 5
        end_r = 5
        draw.ellipse(
            (
                float(start_display[0]) - start_r,
                float(start_display[1]) - start_r,
                float(start_display[0]) + start_r,
                float(start_display[1]) + start_r,
            ),
            fill=_endpoint_color(start_type),
            outline=(0, 0, 0),
        )
        draw.ellipse(
            (
                float(end_display[0]) - end_r,
                float(end_display[1]) - end_r,
                float(end_display[0]) + end_r,
                float(end_display[1]) + end_r,
            ),
            fill=_endpoint_color(end_type),
            outline=(0, 0, 0),
        )
        if str(label_mode).strip().lower() == "line":
            line_label = str(row.get("line_label", row.get("line_id", ""))).strip()
            if line_label:
                label_anchor = _polyline_label_anchor(xy)
                draw.text(
                    (float(label_anchor[0]) + 6, float(label_anchor[1]) - 10),
                    line_label,
                    fill=color,
                    font=font,
                )
        else:
            start_label = str(row.get("start_label", start_type))
            end_label = str(row.get("end_label", end_type))
            draw.text(
                (float(start_display[0]) + 6, float(start_display[1]) - 10),
                start_label,
                fill=_endpoint_color(start_type),
                font=font,
            )
            draw.text(
                (float(end_display[0]) + 6, float(end_display[1]) - 10),
                end_label,
                fill=_endpoint_color(end_type),
                font=font,
            )
    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct full-size masked GeoTIFF and GeoJSON from processed geo-current training dataset."
    )
    parser.add_argument("--processed-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--stage", type=str, default="stage_a", choices=["stage_a", "stage_b"])
    parser.add_argument("--family-manifest", type=str, default="")
    parser.add_argument("--source-sample-id", type=str, default="")
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_root = Path(args.processed_root).resolve()
    output_root = Path(args.output_root).resolve()
    ensure_dir(output_root)

    dataset_root = processed_root / str(args.stage) / "dataset"
    meta_path = dataset_root / f"meta_{args.split}.jsonl"
    rows_path = dataset_root / f"{args.split}.jsonl"

    family_manifest = Path(args.family_manifest).resolve() if str(args.family_manifest).strip() else processed_root / "family_manifest.jsonl"
    manifest_map: Dict[str, Dict] = {}
    manifest_mode = "missing"
    if family_manifest.is_file():
        manifest_map = load_manifest_map(family_manifest)
        manifest_mode = str(family_manifest)
    meta_rows = load_jsonl(meta_path)
    image_map = load_dataset_image_map(rows_path)

    grouped: Dict[str, List[Dict]] = defaultdict(list)
    source_meta: Dict[str, Dict] = {}
    for row in meta_rows:
        family = manifest_map.get(str(row["family_id"]))
        sample_id = infer_source_sample_id(family_id=str(row.get("family_id", "")), row=row, family=family)
        if str(args.source_sample_id).strip() and sample_id != str(args.source_sample_id).strip():
            continue
        grouped[sample_id].append(row)
        if family is not None:
            source_meta[sample_id] = family

    sample_ids = sorted(grouped.keys())
    if int(args.max_samples) > 0:
        sample_ids = sample_ids[: int(args.max_samples)]

    summary: List[Dict] = []
    for sample_id in sample_ids:
        family = source_meta.get(sample_id)
        raster_context = resolve_sample_raster_context(rows=grouped[sample_id], family=family)
        width = int(raster_context["width"])
        height = int(raster_context["height"])
        profile = raster_context["profile"]
        transform = raster_context["transform"]
        crs_name = str(raster_context["crs_name"])
        source_image_path = Path(str(raster_context.get("source_image_path", "")).strip()).resolve() if str(raster_context.get("source_image_path", "")).strip() else None

        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        lane_features: List[Dict] = []
        inter_features: List[Dict] = []
        lane_features_pixel: List[Dict] = []
        inter_features_pixel: List[Dict] = []
        visual_lines: List[Dict] = []
        visual_lines_float: List[Dict] = []
        seen_lane = set()
        seen_inter = set()

        sample_out = output_root / sample_id
        ensure_dir(sample_out)

        for row in grouped[sample_id]:
            row_id = str(row["id"])
            image_rel = image_map.get(row_id, "")
            if not image_rel:
                continue
            patch_path = dataset_root / Path(image_rel)
            if not patch_path.is_file():
                continue
            crop_box = row["crop_box"]
            x0 = int(crop_box["x_min"])
            y0 = int(crop_box["y_min"])
            x1 = int(crop_box["x_max"])
            y1 = int(crop_box["y_max"])
            patch = rasterio_open(patch_path).read() if patch_path.suffix.lower() in {".tif", ".tiff"} else None
            if patch is None:
                from PIL import Image

                patch_img = np.asarray(Image.open(patch_path).convert("RGB"), dtype=np.uint8)
            else:
                patch_img = np.transpose(patch, (1, 2, 0)).astype(np.uint8)
            canvas[y0:y1, x0:x1] = patch_img[: y1 - y0, : x1 - x0]

            export_lines = row.get("target_lines_float", [])
            if not export_lines:
                export_lines = uv_lines_to_local(row.get("target_lines", []), patch=row)
            for line in export_lines:
                points = line.get("points", [])
                if not isinstance(points, list) or len(points) < 2:
                    continue
                global_points = [[float(p[0]) + x0, float(p[1]) + y0] for p in points]
                key = (
                    str(line.get("category", "")),
                    str(line.get("geometry_type", "line")),
                    str(line.get("start_type", "")),
                    str(line.get("end_type", "")),
                    tuple((round(float(p[0]), 3), round(float(p[1]), 3)) for p in global_points),
                )
                geometry_type = str(line.get("geometry_type", "line"))
                if str(line.get("category", "")) == "intersection_polygon":
                    ring_world = global_points if transform is None else pixel_to_world(global_points, transform)
                    if len(ring_world) >= 3 and ring_world[0] != ring_world[-1]:
                        ring_world.append(list(ring_world[0]))
                    ring_pixel = [[float(p[0]), float(p[1])] for p in global_points]
                    if len(ring_pixel) >= 3 and ring_pixel[0] != ring_pixel[-1]:
                        ring_pixel.append(list(ring_pixel[0]))
                    feature = {
                        "type": "Feature",
                        "properties": {
                            "category": str(line.get("category", "")),
                            "geometry_type": geometry_type,
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [ring_world],
                        },
                    }
                    feature_pixel = {
                        "type": "Feature",
                        "properties": {
                            "category": str(line.get("category", "")),
                            "geometry_type": geometry_type,
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [ring_pixel],
                        },
                    }
                else:
                    line_world = global_points if transform is None else pixel_to_world(global_points, transform)
                    feature = {
                        "type": "Feature",
                        "properties": {
                            "category": str(line.get("category", "")),
                            "start_type": str(line.get("start_type", "")),
                            "end_type": str(line.get("end_type", "")),
                            "geometry_type": geometry_type,
                        },
                        "geometry": {
                            "type": "LineString",
                            "coordinates": line_world,
                        },
                    }
                    feature_pixel = {
                        "type": "Feature",
                        "properties": {
                            "category": str(line.get("category", "")),
                            "start_type": str(line.get("start_type", "")),
                            "end_type": str(line.get("end_type", "")),
                            "geometry_type": geometry_type,
                        },
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[float(p[0]), float(p[1])] for p in global_points],
                        },
                    }
                if str(line.get("category", "")) == "intersection_polygon":
                    if key not in seen_inter:
                        seen_inter.add(key)
                        inter_features.append(feature)
                        inter_features_pixel.append(feature_pixel)
                else:
                    if key not in seen_lane:
                        seen_lane.add(key)
                        lane_features.append(feature)
                        lane_features_pixel.append(feature_pixel)

            patch_visual_lines: List[Dict] = []
            visual_quantized_source = row.get("target_lines_quantized", [])
            if not visual_quantized_source:
                visual_quantized_source = uv_lines_to_local(row.get("target_lines", []), patch=row)
            for line in visual_quantized_source:
                points = line.get("points", [])
                if not isinstance(points, list) or len(points) < 2:
                    continue
                global_points = [[int(p[0]) + x0, int(p[1]) + y0] for p in points]
                patch_visual_lines.append(
                    {
                        "category": str(line.get("category", "")),
                        "geometry_type": str(line.get("geometry_type", "line")),
                        "start_type": str(line.get("start_type", "")),
                        "end_type": str(line.get("end_type", "")),
                        "points": global_points,
                        "local_points": [[float(p[0]), float(p[1])] for p in points],
                        "viz_kind": "quantized",
                    }
                )
            visual_lines.extend(annotate_patch_endpoint_order_labels(patch_visual_lines))

            patch_visual_lines_float: List[Dict] = []
            float_source = row.get("target_lines_float", [])
            if not float_source:
                float_source = uv_lines_to_local(row.get("target_lines", []), patch=row)
            for line in float_source:
                points = line.get("points", [])
                if not isinstance(points, list) or len(points) < 2:
                    continue
                global_points = [[float(p[0]) + x0, float(p[1]) + y0] for p in points]
                patch_visual_lines_float.append(
                    {
                        "category": str(line.get("category", "")),
                        "geometry_type": str(line.get("geometry_type", "line")),
                        "start_type": str(line.get("start_type", "")),
                        "end_type": str(line.get("end_type", "")),
                        "points": global_points,
                        "local_points": [[float(p[0]), float(p[1])] for p in points],
                        "viz_kind": "float",
                    }
                )
            visual_lines_float.extend(annotate_patch_endpoint_order_labels(patch_visual_lines_float))

        tif_path = sample_out / "masked_reconstructed.tif"
        png_path = sample_out / "masked_reconstructed.png"
        overlay_path = sample_out / "masked_reconstructed_overlay.png"
        overlay_float_path = sample_out / "masked_reconstructed_overlay_float.png"
        overlay_compare_path = sample_out / "masked_reconstructed_overlay_compare.png"
        Image.fromarray(canvas, mode="RGB").save(png_path)
        if isinstance(profile, dict):
            out_profile = profile.copy()
            out_profile.update(count=3, dtype="uint8")
            with rasterio_open(tif_path, "w", **out_profile) as dst:
                dst.write(np.transpose(canvas, (2, 0, 1)))
        else:
            Image.fromarray(canvas, mode="RGB").save(tif_path)
        build_overlay_image(canvas=canvas, visual_lines=visual_lines).save(overlay_path)
        build_overlay_image(canvas=canvas, visual_lines=visual_lines_float, color_mode="compare").save(overlay_float_path)
        build_overlay_image(canvas=canvas, visual_lines=[*visual_lines_float, *visual_lines], color_mode="compare").save(overlay_compare_path)

        lane_geojson = build_feature_collection(lane_features, crs_name=crs_name, name="Lane")
        inter_geojson = build_feature_collection(inter_features, crs_name=crs_name, name="Intersection")
        lane_pixel_geojson = build_feature_collection(lane_features_pixel, crs_name="LOCAL_PIXEL", name="LanePixel")
        inter_pixel_geojson = build_feature_collection(inter_features_pixel, crs_name="LOCAL_PIXEL", name="IntersectionPixel")
        lane_path = sample_out / "Lane.geojson"
        inter_path = sample_out / "Intersection.geojson"
        lane_pixel_path = sample_out / "Lane.pixel.geojson"
        inter_pixel_path = sample_out / "Intersection.pixel.geojson"
        with lane_path.open("w", encoding="utf-8") as f:
            json.dump(lane_geojson, f, ensure_ascii=False, indent=2)
        with inter_path.open("w", encoding="utf-8") as f:
            json.dump(inter_geojson, f, ensure_ascii=False, indent=2)
        with lane_pixel_path.open("w", encoding="utf-8") as f:
            json.dump(lane_pixel_geojson, f, ensure_ascii=False, indent=2)
        with inter_pixel_path.open("w", encoding="utf-8") as f:
            json.dump(inter_pixel_geojson, f, ensure_ascii=False, indent=2)

        summary.append(
            {
                "source_sample_id": sample_id,
                "manifest_mode": manifest_mode,
                "source_image_path": str(source_image_path) if source_image_path is not None else "",
                "georeferenced": bool(raster_context.get("georeferenced", False)),
                "patch_count": len(grouped[sample_id]),
                "lane_feature_count": len(lane_features),
                "intersection_polygon_count": len(inter_features),
                "png_path": str(png_path),
                "tif_path": str(tif_path),
                "overlay_path": str(overlay_path),
                "overlay_float_path": str(overlay_float_path),
                "overlay_compare_path": str(overlay_compare_path),
                "lane_geojson_path": str(lane_path),
                "intersection_geojson_path": str(inter_path),
                "lane_pixel_geojson_path": str(lane_pixel_path),
                "intersection_pixel_geojson_path": str(inter_pixel_path),
            }
        )
        print(
            f"[Verify] sample={sample_id} patches={len(grouped[sample_id])} "
            f"lane={len(lane_features)} inter={len(inter_features)} "
            f"mode={manifest_mode} georef={bool(raster_context.get('georeferenced', False))} -> {sample_out}",
            flush=True,
        )

    summary_path = output_root / "reconstruct_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[Verify] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
