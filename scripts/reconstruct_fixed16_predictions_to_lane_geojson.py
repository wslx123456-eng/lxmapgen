import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from rasterio import open as rasterio_open

from export_llamafactory_patch_only_from_raw_family_manifest import clip_polyline_to_rect, dedup_points, point_boundary_side
from geo_current_dataset_v1_common import load_jsonl, parse_generated_json, sanitize_pred_lines_uv


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stitch fixed16 prediction outputs back into full Lane.geojson files."
    )
    parser.add_argument("--fixed16-root", type=str, required=True, help="fixed16_stage_a or fixed16_stage_b root.")
    parser.add_argument("--predictions-path", type=str, required=True, help="Prediction json/jsonl file aligned with the fixed16 split.")
    parser.add_argument("--output-root", type=str, required=True, help="Where to write stitched GeoJSON outputs.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--family-manifest", type=str, default="", help="Optional family_manifest.jsonl path. Defaults to fixed16_root/../family_manifest.jsonl")
    parser.add_argument("--merge-endpoint-tol-px", type=float, default=2.0, help="Tolerance for merging cut-to-cut box fragments.")
    parser.add_argument("--source-sample-id", type=str, default="", help="Optional source sample filter.")
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def load_prediction_rows(path: Path) -> List[Dict]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl(path)
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return [row for row in obj if isinstance(row, dict)]
    if isinstance(obj, dict):
        for key in ("predictions", "rows", "data", "items"):
            value = obj.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        return [obj]
    return []


def _parse_prediction_text(text: str) -> List[Dict]:
    pred_obj, _ = parse_generated_json(str(text or ""))
    if isinstance(pred_obj, dict):
        return sanitize_pred_lines_uv(list(pred_obj.get("lines", [])))
    return []


def extract_prediction_lines(row: Dict) -> List[Dict]:
    direct_keys = ("pred_lines", "prediction_lines", "lines")
    for key in direct_keys:
        value = row.get(key)
        if isinstance(value, list):
            return sanitize_pred_lines_uv(value)
        if isinstance(value, dict) and isinstance(value.get("lines"), list):
            return sanitize_pred_lines_uv(value.get("lines", []))

    text_keys = (
        "predict",
        "prediction",
        "pred_text",
        "generated_text",
        "response",
        "output",
        "assistant",
        "text",
    )
    for key in text_keys:
        value = row.get(key)
        if isinstance(value, dict):
            if isinstance(value.get("lines"), list):
                return sanitize_pred_lines_uv(value.get("lines", []))
            if isinstance(value.get("text"), str):
                parsed = _parse_prediction_text(value["text"])
                if parsed:
                    return parsed
        if isinstance(value, str) and value.strip():
            parsed = _parse_prediction_text(value)
            if parsed:
                return parsed

    messages = row.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            if str(msg.get("role", "")).strip().lower() != "assistant":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                parsed = _parse_prediction_text(content)
                if parsed:
                    return parsed
    return []


def build_prediction_index(prediction_rows: Sequence[Dict], meta_rows: Sequence[Dict]) -> Tuple[Dict[str, List[Dict]], Dict[str, int], int]:
    by_id: Dict[str, List[Dict]] = {}
    parse_count = 0
    for row in prediction_rows:
        pred_lines = extract_prediction_lines(row)
        parse_count += 1 if pred_lines else 0
        sample_id = str(
            row.get("id")
            or row.get("sample_id")
            or row.get("source_id")
            or row.get("instance_id")
            or row.get("custom_id")
            or ""
        ).strip()
        if sample_id:
            by_id[sample_id] = pred_lines

    fallback_index: Dict[str, int] = {}
    if len(by_id) < len(meta_rows) and len(prediction_rows) == len(meta_rows):
        for idx, meta in enumerate(meta_rows):
            fallback_index[str(meta.get("id"))] = idx
    return by_id, fallback_index, parse_count


def clip_pred_lines_to_target_box(pred_lines: Sequence[Dict], target_box: Dict[str, int], boundary_tol_px: float) -> List[Dict]:
    rect = (
        float(target_box["x_min"]),
        float(target_box["y_min"]),
        float(target_box["x_max"]),
        float(target_box["y_max"]),
    )
    out: List[Dict] = []
    for line in pred_lines:
        if str(line.get("category", "lane_line")) != "lane_line":
            continue
        arr = np.asarray(line.get("points", []), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] != 2:
            continue
        pieces = clip_polyline_to_rect(arr, rect)
        for piece in pieces:
            piece = dedup_points(piece)
            if piece.ndim != 2 or piece.shape[0] < 2:
                continue
            start_type = str(line.get("start_type", "start"))
            end_type = str(line.get("end_type", "end"))
            if point_boundary_side(piece[0], rect, float(boundary_tol_px)) is not None:
                start_type = "cut"
            elif start_type not in {"start", "cut"}:
                start_type = "start"
            if point_boundary_side(piece[-1], rect, float(boundary_tol_px)) is not None:
                end_type = "cut"
            elif end_type not in {"end", "cut"}:
                end_type = "end"
            out.append(
                {
                    "category": "lane_line",
                    "start_type": start_type,
                    "end_type": end_type,
                    "points": [[float(x), float(y)] for x, y in piece.tolist()],
                }
            )
    return out


def to_global_lines(pred_lines: Sequence[Dict], crop_box: Dict[str, int]) -> List[Dict]:
    x0 = float(crop_box["x_min"])
    y0 = float(crop_box["y_min"])
    out: List[Dict] = []
    for line in pred_lines:
        arr = np.asarray(line.get("points", []), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 2:
            continue
        global_arr = dedup_points(arr + np.asarray([[x0, y0]], dtype=np.float32))
        if global_arr.ndim != 2 or global_arr.shape[0] < 2:
            continue
        out.append(
            {
                "category": "lane_line",
                "start_type": str(line.get("start_type", "start")),
                "end_type": str(line.get("end_type", "end")),
                "points_global": global_arr,
            }
        )
    return out


def _merge_key(line: Dict) -> Tuple[str]:
    return (str(line.get("category", "lane_line")),)


def _endpoint_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _merge_two_lines(a: Dict, b: Dict, tol_px: float) -> Optional[Dict]:
    if _merge_key(a) != _merge_key(b):
        return None
    a_pts = np.asarray(a["points_global"], dtype=np.float32)
    b_pts = np.asarray(b["points_global"], dtype=np.float32)
    if a_pts.ndim != 2 or b_pts.ndim != 2 or a_pts.shape[0] < 2 or b_pts.shape[0] < 2:
        return None

    candidates = []
    if str(a.get("end_type")) == "cut" and str(b.get("start_type")) == "cut":
        candidates.append(("a_end_b_start", a_pts[-1], b_pts[0]))
    if str(a.get("end_type")) == "cut" and str(b.get("end_type")) == "cut":
        candidates.append(("a_end_b_end", a_pts[-1], b_pts[-1]))
    if str(a.get("start_type")) == "cut" and str(b.get("start_type")) == "cut":
        candidates.append(("a_start_b_start", a_pts[0], b_pts[0]))
    if str(a.get("start_type")) == "cut" and str(b.get("end_type")) == "cut":
        candidates.append(("a_start_b_end", a_pts[0], b_pts[-1]))

    best_mode = None
    best_dist = float("inf")
    for mode, p0, p1 in candidates:
        dist = _endpoint_distance(np.asarray(p0), np.asarray(p1))
        if dist <= float(tol_px) and dist < best_dist:
            best_mode = mode
            best_dist = dist

    if best_mode is None:
        return None

    if best_mode == "a_end_b_start":
        join = 0.5 * (a_pts[-1] + b_pts[0])
        pts = np.vstack([a_pts[:-1], join[None, :], b_pts[1:]])
        start_type = str(a.get("start_type", "start"))
        end_type = str(b.get("end_type", "end"))
    elif best_mode == "a_end_b_end":
        join = 0.5 * (a_pts[-1] + b_pts[-1])
        pts = np.vstack([a_pts[:-1], join[None, :], b_pts[-2::-1]])
        start_type = str(a.get("start_type", "start"))
        end_type = str(b.get("start_type", "start"))
    elif best_mode == "a_start_b_start":
        join = 0.5 * (a_pts[0] + b_pts[0])
        pts = np.vstack([a_pts[:0:-1], join[None, :], b_pts[1:]])
        start_type = str(a.get("end_type", "end"))
        end_type = str(b.get("end_type", "end"))
    else:
        join = 0.5 * (a_pts[0] + b_pts[-1])
        pts = np.vstack([b_pts[:-1], join[None, :], a_pts[1:]])
        start_type = str(b.get("start_type", "start"))
        end_type = str(a.get("end_type", "end"))

    merged_pts = dedup_points(pts)
    if merged_pts.ndim != 2 or merged_pts.shape[0] < 2:
        return None
    return {
        "category": "lane_line",
        "start_type": start_type,
        "end_type": end_type,
        "points_global": merged_pts,
    }


def merge_cut_connected_lines(lines: Sequence[Dict], tol_px: float) -> List[Dict]:
    working = [dict(line) for line in lines]
    changed = True
    while changed:
        changed = False
        for i in range(len(working)):
            if changed:
                break
            for j in range(i + 1, len(working)):
                merged = _merge_two_lines(working[i], working[j], tol_px=float(tol_px))
                if merged is None:
                    continue
                working[i] = merged
                del working[j]
                changed = True
                break
    return working


def build_lane_features(lines: Sequence[Dict], transform) -> List[Dict]:
    features: List[Dict] = []
    for line in lines:
        arr = np.asarray(line.get("points_global", []), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 2:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "category": "lane_line",
                    "start_type": str(line.get("start_type", "start")),
                    "end_type": str(line.get("end_type", "end")),
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": pixel_to_world(arr.tolist(), transform),
                },
            }
        )
    return features


def main() -> None:
    args = parse_args()
    fixed16_root = Path(args.fixed16_root).resolve()
    predictions_path = Path(args.predictions_path).resolve()
    output_root = Path(args.output_root).resolve()
    ensure_dir(output_root)

    family_manifest = Path(args.family_manifest).resolve() if str(args.family_manifest).strip() else (fixed16_root.parent / "family_manifest.jsonl")
    meta_path = fixed16_root / f"meta_{args.split}.jsonl"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing fixed16 meta file: {meta_path}")
    if not family_manifest.is_file():
        raise FileNotFoundError(f"Missing family manifest: {family_manifest}")
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Missing predictions file: {predictions_path}")

    manifest_rows = load_jsonl(family_manifest)
    manifest_map = {str(row["family_id"]): row for row in manifest_rows}
    meta_rows = load_jsonl(meta_path)
    prediction_rows = load_prediction_rows(predictions_path)
    pred_by_id, fallback_index, parse_count = build_prediction_index(prediction_rows, meta_rows)

    grouped_global_lines: Dict[str, List[Dict]] = defaultdict(list)
    grouped_manifest: Dict[str, Dict] = {}
    matched_predictions = 0
    missing_predictions = 0
    raw_piece_count = 0

    for idx, meta in enumerate(meta_rows):
        family = manifest_map.get(str(meta.get("family_id")))
        if family is None:
            continue
        source_sample_id = str(family.get("source_sample_id", family.get("source_image", meta.get("family_id"))))
        if str(args.source_sample_id).strip() and source_sample_id != str(args.source_sample_id).strip():
            continue
        grouped_manifest[source_sample_id] = family

        pred_lines = pred_by_id.get(str(meta.get("id")))
        if pred_lines is None and str(meta.get("id")) in fallback_index:
            pred_lines = extract_prediction_lines(prediction_rows[fallback_index[str(meta.get("id"))]])
        if pred_lines is None:
            missing_predictions += 1
            continue
        matched_predictions += 1
        clipped_local_lines = clip_pred_lines_to_target_box(
            pred_lines=pred_lines,
            target_box=dict(meta.get("target_box", {})),
            boundary_tol_px=float(args.merge_endpoint_tol_px),
        )
        raw_piece_count += len(clipped_local_lines)
        grouped_global_lines[source_sample_id].extend(
            to_global_lines(
                pred_lines=clipped_local_lines,
                crop_box=dict(meta.get("crop_box", {})),
            )
        )

    sample_ids = sorted(grouped_global_lines.keys())
    if int(args.max_samples) > 0:
        sample_ids = sample_ids[: int(args.max_samples)]

    summary: List[Dict] = []
    for sample_id in sample_ids:
        family = grouped_manifest[sample_id]
        source_image_path = Path(str(family["source_image_path"])).resolve()
        with rasterio_open(source_image_path) as ds:
            transform = ds.transform
            crs_name = str(ds.crs) if ds.crs is not None else "urn:ogc:def:crs:OGC:1.3:CRS84"

        raw_lines = grouped_global_lines[sample_id]
        merged_lines = merge_cut_connected_lines(raw_lines, tol_px=float(args.merge_endpoint_tol_px))

        sample_out = output_root / sample_id
        ensure_dir(sample_out)
        raw_lane_geojson = build_feature_collection(build_lane_features(raw_lines, transform), crs_name=crs_name, name="Lane_raw")
        merged_lane_geojson = build_feature_collection(build_lane_features(merged_lines, transform), crs_name=crs_name, name="Lane")

        raw_lane_path = sample_out / "Lane.raw.geojson"
        lane_path = sample_out / "Lane.geojson"
        with raw_lane_path.open("w", encoding="utf-8") as f:
            json.dump(raw_lane_geojson, f, ensure_ascii=False, indent=2)
        with lane_path.open("w", encoding="utf-8") as f:
            json.dump(merged_lane_geojson, f, ensure_ascii=False, indent=2)

        summary.append(
            {
                "source_sample_id": sample_id,
                "source_image_path": str(source_image_path),
                "raw_piece_count": len(raw_lines),
                "merged_lane_count": len(merged_lines),
                "lane_geojson_path": str(lane_path),
                "raw_lane_geojson_path": str(raw_lane_path),
            }
        )

    summary_path = output_root / "reconstruct_fixed16_predictions.summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "fixed16_root": str(fixed16_root),
                "predictions_path": str(predictions_path),
                "family_manifest": str(family_manifest),
                "split": str(args.split),
                "matched_predictions": int(matched_predictions),
                "missing_predictions": int(missing_predictions),
                "parsed_prediction_rows": int(parse_count),
                "raw_piece_count": int(raw_piece_count),
                "samples": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[Fixed16 Stitch] wrote summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
