import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from reconstruct_geo_current_from_processed_dataset import (
    annotate_patch_endpoint_order_labels,
    build_overlay_image,
    ensure_dir,
    infer_source_sample_id,
    load_dataset_image_map,
)
from geo_current_dataset_v1_common import load_jsonl, uv_lines_to_local


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize processed geo-current dataset patch by patch."
    )
    parser.add_argument("--processed-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--stage", type=str, default="stage_a", choices=["stage_a", "stage_b"])
    parser.add_argument("--source-sample-id", type=str, default="")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-patches-per-sample", type=int, default=0)
    return parser.parse_args()


def _build_visual_lines(lines: List[Dict], kind: str) -> List[Dict]:
    out: List[Dict] = []
    for line in lines:
        points = line.get("points", [])
        if not isinstance(points, list) or len(points) < 2:
            continue
        local_points = [[float(p[0]), float(p[1])] for p in points]
        out.append(
            {
                "category": str(line.get("category", "")),
                "geometry_type": str(line.get("geometry_type", "line")),
                "start_type": str(line.get("start_type", "")),
                "end_type": str(line.get("end_type", "")),
                "points": local_points,
                "local_points": local_points,
                "viz_kind": str(kind),
            }
        )
    return annotate_patch_endpoint_order_labels(out)


def main() -> None:
    args = parse_args()
    processed_root = Path(args.processed_root).resolve()
    output_root = Path(args.output_root).resolve()
    ensure_dir(output_root)

    dataset_root = processed_root / str(args.stage) / "dataset"
    meta_path = dataset_root / f"meta_{args.split}.jsonl"
    rows_path = dataset_root / f"{args.split}.jsonl"

    meta_rows = load_jsonl(meta_path)
    image_map = load_dataset_image_map(rows_path)

    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in meta_rows:
        sample_id = infer_source_sample_id(family_id=str(row.get("family_id", "")), row=row, family=None)
        if str(args.source_sample_id).strip() and sample_id != str(args.source_sample_id).strip():
            continue
        grouped[sample_id].append(row)

    sample_ids = sorted(grouped.keys())
    if int(args.max_samples) > 0:
        sample_ids = sample_ids[: int(args.max_samples)]

    summary: List[Dict] = []
    for sample_id in sample_ids:
        sample_out = output_root / sample_id
        ensure_dir(sample_out)
        rows = sorted(grouped[sample_id], key=lambda item: (int(item.get("row", 0)), int(item.get("col", 0)), int(item.get("patch_id", 0))))
        if int(args.max_patches_per_sample) > 0:
            rows = rows[: int(args.max_patches_per_sample)]

        patch_summaries: List[Dict] = []
        for row in rows:
            row_id = str(row["id"])
            image_rel = image_map.get(row_id, "")
            if not image_rel:
                continue
            patch_path = dataset_root / Path(image_rel)
            if not patch_path.is_file():
                continue

            patch_image = Image.open(patch_path).convert("RGB")
            patch_np = np.asarray(patch_image, dtype=np.uint8)
            patch_id = int(row.get("patch_id", -1))
            patch_dir = sample_out / f"p{patch_id:04d}"
            ensure_dir(patch_dir)

            quantized_source = row.get("target_lines_quantized", [])
            if not quantized_source:
                quantized_source = uv_lines_to_local(row.get("target_lines", []), patch=row)
            quantized_lines = _build_visual_lines(quantized_source, kind="quantized")
            float_source = row.get("target_lines_float", [])
            if not float_source:
                float_source = uv_lines_to_local(row.get("target_lines", []), patch=row)
            float_lines = _build_visual_lines(float_source, kind="float")
            crop_box = row.get("crop_box", {})
            keep_box = row.get("keep_box", {})
            keep_box_local = {
                "x_min": float(keep_box.get("x_min", 0)) - float(crop_box.get("x_min", 0)),
                "y_min": float(keep_box.get("y_min", 0)) - float(crop_box.get("y_min", 0)),
                "x_max": float(keep_box.get("x_max", 0)) - float(crop_box.get("x_min", 0)),
                "y_max": float(keep_box.get("y_max", 0)) - float(crop_box.get("y_min", 0)),
                "label": "keep_box",
            }

            patch_image.save(patch_dir / "patch.png")
            build_overlay_image(canvas=patch_np, visual_lines=quantized_lines, keep_boxes=[keep_box_local]).save(
                patch_dir / "overlay_quantized.png"
            )
            build_overlay_image(canvas=patch_np, visual_lines=float_lines, color_mode="compare", keep_boxes=[keep_box_local]).save(
                patch_dir / "overlay_float.png"
            )
            build_overlay_image(
                canvas=patch_np,
                visual_lines=[*float_lines, *quantized_lines],
                color_mode="compare",
                keep_boxes=[keep_box_local],
            ).save(patch_dir / "overlay_compare.png")

            patch_meta = {
                "id": row_id,
                "family_id": row.get("family_id"),
                "patch_id": patch_id,
                "row": int(row.get("row", 0)),
                "col": int(row.get("col", 0)),
                "crop_box": crop_box,
                "keep_box": keep_box,
                "keep_box_local": keep_box_local,
                "num_target_lines": int(len(row.get("target_lines", []))),
                "num_target_lines_float": int(len(row.get("target_lines_float", []))),
            }
            with (patch_dir / "patch_meta.json").open("w", encoding="utf-8") as f:
                json.dump(patch_meta, f, ensure_ascii=False, indent=2)

            patch_summaries.append(
                {
                    "patch_id": patch_id,
                    "patch_dir": str(patch_dir),
                    "patch_image": str(patch_dir / "patch.png"),
                    "overlay_quantized": str(patch_dir / "overlay_quantized.png"),
                    "overlay_float": str(patch_dir / "overlay_float.png"),
                    "overlay_compare": str(patch_dir / "overlay_compare.png"),
                    "patch_meta": str(patch_dir / "patch_meta.json"),
                }
            )

        with (sample_out / "patch_visualize_summary.json").open("w", encoding="utf-8") as f:
            json.dump(patch_summaries, f, ensure_ascii=False, indent=2)

        summary.append(
            {
                "source_sample_id": sample_id,
                "patch_count": len(patch_summaries),
                "sample_output_dir": str(sample_out),
            }
        )
        print(
            f"[PatchViz] sample={sample_id} patches={len(patch_summaries)} -> {sample_out}",
            flush=True,
        )

    with (output_root / "patch_visualize_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[PatchViz] summary={output_root / 'patch_visualize_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
