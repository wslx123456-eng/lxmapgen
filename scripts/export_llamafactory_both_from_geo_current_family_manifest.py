import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from export_llamafactory_state_sft_from_raw_family_manifest import build_sample_rng, choose_state_mode
from geo_current_dataset_v1_common import (
    DEFAULT_STAGEA_PROMPT_TEMPLATE,
    DEFAULT_STAGEA_SYSTEM_PROMPT,
    DEFAULT_STAGEB_PROMPT_TEMPLATE,
    DEFAULT_STAGEB_SYSTEM_PROMPT,
    apply_state_mode,
    build_owned_segments_by_patch,
    build_patch_image,
    build_patch_only_record,
    build_patch_target_lines,
    build_patch_target_lines_float,
    build_patch_target_lines_quantized,
    build_state_record,
    ensure_dir,
    extract_state_lines,
    family_global_lines,
    load_family_raster_and_mask,
    load_jsonl,
    local_lines_to_uv,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Stage A and Stage B datasets together from current-dataset family manifests.")
    parser.add_argument("--family-manifest", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"])
    parser.add_argument("--band-indices", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--resample-step-px", type=float, default=4.0)
    parser.add_argument("--boundary-tol-px", type=float, default=2.5)
    parser.add_argument("--trace-points", type=int, default=8)
    parser.add_argument("--state-mixture-mode", type=str, default="full", choices=["full", "mixed"])
    parser.add_argument("--state-no-state-ratio", type=float, default=0.30)
    parser.add_argument("--state-weak-ratio", type=float, default=0.40)
    parser.add_argument("--state-full-ratio", type=float, default=0.30)
    parser.add_argument("--state-weak-trace-points", type=int, default=3)
    parser.add_argument("--state-line-dropout", type=float, default=0.40)
    parser.add_argument("--state-point-jitter-px", type=float, default=2.0)
    parser.add_argument("--state-truncate-prob", type=float, default=0.30)
    parser.add_argument("--include-lane", action="store_true")
    parser.add_argument("--include-intersection-boundary", action="store_true")
    parser.add_argument("--max-families-per-split", type=int, default=0)
    parser.add_argument("--empty-patch-drop-ratio", type=float, default=0.95)
    parser.add_argument("--empty-patch-seed", type=int, default=42)
    parser.add_argument("--use-system-prompt", action="store_true")
    parser.add_argument("--stagea-system-prompt", type=str, default=DEFAULT_STAGEA_SYSTEM_PROMPT)
    parser.add_argument("--stagea-prompt-template", type=str, default=DEFAULT_STAGEA_PROMPT_TEMPLATE)
    parser.add_argument("--stageb-system-prompt", type=str, default=DEFAULT_STAGEB_SYSTEM_PROMPT)
    parser.add_argument("--stageb-prompt-template", type=str, default=DEFAULT_STAGEB_PROMPT_TEMPLATE)
    return parser.parse_args()


def build_dataset_registry(output_root: Path, prefix: str) -> Dict[str, Dict]:
    registry: Dict[str, Dict] = {}
    for split in ("train", "val"):
        dataset_file = output_root / f"{split}.jsonl"
        if not dataset_file.is_file():
            continue
        registry[f"{prefix}_{split}"] = {
            "file_name": str(dataset_file.resolve()),
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    return registry


def downsample_empty_patch_records(
    records: Sequence[Dict],
    drop_ratio: float,
    seed: int,
    split: str,
) -> Tuple[List[Dict], Dict[str, float]]:
    safe_ratio = max(0.0, min(1.0, float(drop_ratio)))
    generated_total = len(records)
    empty_records = [record for record in records if int(record["stagea_meta"].get("num_target_lines", 0)) <= 0]
    non_empty_records = [record for record in records if int(record["stagea_meta"].get("num_target_lines", 0)) > 0]
    keep_empty_count = int(round(float(len(empty_records)) * (1.0 - safe_ratio)))
    keep_empty_count = max(0, min(len(empty_records), keep_empty_count))
    shuffled_empty = list(empty_records)
    random.Random(f"{int(seed)}::{split}").shuffle(shuffled_empty)
    kept_empty_records = shuffled_empty[:keep_empty_count]
    keep_ids = {id(record) for record in non_empty_records}
    keep_ids.update(id(record) for record in kept_empty_records)
    kept_records = [record for record in records if id(record) in keep_ids]
    summary = {
        "generated_total": int(generated_total),
        "generated_non_empty": int(len(non_empty_records)),
        "generated_empty": int(len(empty_records)),
        "kept_total": int(len(kept_records)),
        "kept_non_empty": int(len(non_empty_records)),
        "kept_empty": int(len(kept_empty_records)),
        "dropped_empty": int(len(empty_records) - len(kept_empty_records)),
        "drop_ratio": float(safe_ratio),
    }
    return kept_records, summary


def export_families_to_stage_datasets(
    families: Sequence[Dict],
    output_root: Path,
    splits: Sequence[str],
    band_indices: Sequence[int],
    mask_threshold: int,
    resample_step_px: float,
    boundary_tol_px: float,
    trace_points: int,
    state_mixture_mode: str,
    state_no_state_ratio: float,
    state_weak_ratio: float,
    state_full_ratio: float,
    state_weak_trace_points: int,
    state_line_dropout: float,
    state_point_jitter_px: float,
    state_truncate_prob: float,
    include_lane: bool,
    include_intersection_boundary: bool,
    max_families_per_split: int,
    empty_patch_drop_ratio: float,
    empty_patch_seed: int,
    empty_patch_drop_ratio_by_split: Dict[str, float] | None,
    stagea_system_prompt: str,
    stagea_prompt_template: str,
    stageb_system_prompt: str,
    stageb_prompt_template: str,
) -> Dict[str, object]:
    #创建输出目录
    output_root = Path(output_root).resolve()
    stage_a_root = output_root / "stage_a" / "dataset"
    stage_b_root = output_root / "stage_b" / "dataset"
    ensure_dir(stage_a_root)
    ensure_dir(stage_b_root)
    #初始化 split 相关容器
    split_set = {str(split) for split in splits}
    split_records: Dict[str, List[Dict]] = {str(split): [] for split in splits}
    family_seen: Dict[str, int] = {str(split): 0 for split in splits}
    family_exported: Dict[str, int] = {str(split): 0 for split in splits}
    #遍历每个 family
    for family in families:
        split = str(family.get("split"))
        #过滤 split 和数量限制
        if split not in split_set:
            continue
        #统计看到了多少 family
        #如果限制了每个 split 最大导出 family 数量，就超过后跳过
        #真正导出的才记到 family_exported
        family_seen[split] += 1
        if int(max_families_per_split) > 0 and family_seen[split] > int(max_families_per_split):
            continue
        family_exported[split] += 1
        #读取 family 的大图元数据和 mask，并把 mask 外的图像区域置黑
        raw_image_hwc, raster_meta, review_mask = load_family_raster_and_mask(
            family=family,
            band_indices=[int(x) for x in band_indices],
            mask_threshold=int(mask_threshold),
        )
        #根据一个 family，把这张原始大图对应的 Lane.geojson 和 Intersection.geojson 读出来
        #转换成“整图像素坐标系下的全局几何线/面”列表
        global_lines = family_global_lines(
            family=family,
            raster_meta=raster_meta,
            include_lane=bool(include_lane),
            include_intersection=bool(include_intersection_boundary),
        )
        #把整图真值切成每个patch真正拥有的那部分线/路口
        owned_segments_by_patch = build_owned_segments_by_patch(
            family=family,
            global_lines=global_lines,
            review_mask=review_mask,
            resample_step_px=float(resample_step_px),
            boundary_tol_px=float(boundary_tol_px),
        )

        for patch in sorted(list(family["patches"]), key=lambda item: int(item["patch_id"])):
            patch_id = int(patch["patch_id"])
            patch_image = build_patch_image(raw_image_hwc=raw_image_hwc, patch=patch)
            target_lines_float = build_patch_target_lines_float(owned_segments_by_patch.get(patch_id, []), patch=patch)
            target_lines = build_patch_target_lines(owned_segments_by_patch.get(patch_id, []), patch=patch)
            target_lines_quantized = build_patch_target_lines_quantized(owned_segments_by_patch.get(patch_id, []), patch=patch)
            image_rel = Path("images") / split / str(family["family_id"]) / f"p{patch_id:04d}.png"

            out_stagea_image = stage_a_root / image_rel
            ensure_dir(out_stagea_image.parent)
            patch_image.save(out_stagea_image)

            out_stageb_image = stage_b_root / image_rel
            ensure_dir(out_stageb_image.parent)
            patch_image.save(out_stageb_image)

            sample_id = f"{family['family_id']}_p{patch_id:04d}"
            stagea_row = build_patch_only_record(
                image_rel_path=image_rel.as_posix(),
                target_lines=target_lines,
                sample_id=sample_id,
                system_prompt=stagea_system_prompt,
                prompt_template=str(stagea_prompt_template),
            )
            stagea_meta = {
                "id": sample_id,
                "split": split,
                "family_id": family["family_id"],
                "source_sample_id": family.get("source_sample_id", ""),
                "source_image": family["source_image"],
                "source_image_path": family.get("source_image_path", ""),
                "source_mask_path": family.get("source_mask_path", ""),
                "source_lane_path": family.get("source_lane_path", ""),
                "source_intersection_path": family.get("source_intersection_path", ""),
                "image_size": family.get("image_size", []),
                "patch_id": patch_id,
                "row": int(patch["row"]),
                "col": int(patch["col"]),
                "crop_box": patch["crop_box"],
                "keep_box": patch["keep_box"],
                "mask_ratio": float(patch.get("mask_ratio", 0.0)),
                "mask_pixels": int(patch.get("mask_pixels", 0)),
                "num_target_lines": len(target_lines),
                "target_lines": target_lines,
                "target_lines_quantized": target_lines_quantized,
                "target_lines_float": target_lines_float,
            }

            raw_state_lines = extract_state_lines(
                patch=patch,
                family=family,
                owned_segments_by_patch=owned_segments_by_patch,
                trace_points=int(trace_points),
                boundary_tol_px=float(boundary_tol_px),
            )
            sample_rng = build_sample_rng(sample_id)
            state_mode = choose_state_mode(
                rng=sample_rng,
                mixture_mode=str(state_mixture_mode),
                raw_state_lines=raw_state_lines,
                no_state_ratio=float(state_no_state_ratio),
                weak_ratio=float(state_weak_ratio),
                full_ratio=float(state_full_ratio),
            )
            patch_size = int(patch["crop_box"]["x_max"] - patch["crop_box"]["x_min"])
            state_lines = apply_state_mode(
                raw_state_lines=raw_state_lines,
                state_mode=state_mode,
                patch_size=patch_size,
                weak_trace_points=int(state_weak_trace_points),
                state_line_dropout=float(state_line_dropout),
                state_point_jitter_px=float(state_point_jitter_px),
                state_truncate_prob=float(state_truncate_prob),
                rng=sample_rng,
            )
            state_lines_uv = local_lines_to_uv(state_lines, patch=patch)
            stageb_row = build_state_record(
                image_rel_path=image_rel.as_posix(),
                state_lines=state_lines_uv,
                target_lines=target_lines,
                sample_id=sample_id,
                system_prompt=stageb_system_prompt,
                prompt_template=str(stageb_prompt_template),
            )
            stageb_meta = {
                "id": sample_id,
                "split": split,
                "family_id": family["family_id"],
                "source_sample_id": family.get("source_sample_id", ""),
                "source_image": family["source_image"],
                "source_image_path": family.get("source_image_path", ""),
                "source_mask_path": family.get("source_mask_path", ""),
                "source_lane_path": family.get("source_lane_path", ""),
                "source_intersection_path": family.get("source_intersection_path", ""),
                "image_size": family.get("image_size", []),
                "patch_id": patch_id,
                "row": int(patch["row"]),
                "col": int(patch["col"]),
                "crop_box": patch["crop_box"],
                "keep_box": patch["keep_box"],
                "state_mode": str(state_mode),
                "num_state_lines": len(state_lines),
                "num_target_lines": len(target_lines),
                "state_lines": state_lines_uv,
                "state_lines_float": state_lines,
                "target_lines": target_lines,
                "target_lines_quantized": target_lines_quantized,
                "target_lines_float": target_lines_float,
            }
            split_records[split].append(
                {
                    "stagea_row": stagea_row,
                    "stagea_meta": stagea_meta,
                    "stageb_row": stageb_row,
                    "stageb_meta": stageb_meta,
                }
            )

    filter_summary: Dict[str, Dict[str, float]] = {}
    stage_a_summary: Dict[str, Dict[str, int]] = {}
    stage_b_summary: Dict[str, Dict[str, int]] = {}

    for split in splits:
        split = str(split)
        split_drop_ratio = float(empty_patch_drop_ratio)
        if empty_patch_drop_ratio_by_split is not None and split in empty_patch_drop_ratio_by_split:
            split_drop_ratio = float(empty_patch_drop_ratio_by_split[split])
        kept_records, filter_summary[split] = downsample_empty_patch_records(
            records=split_records[split],
            drop_ratio=float(split_drop_ratio),
            seed=int(empty_patch_seed),
            split=split,
        )
        stagea_rows = [record["stagea_row"] for record in kept_records]
        stagea_meta_rows = [record["stagea_meta"] for record in kept_records]
        stageb_rows = [record["stageb_row"] for record in kept_records]
        stageb_meta_rows = [record["stageb_meta"] for record in kept_records]

        stage_a_summary[split] = {
            "families": int(family_exported.get(split, 0)),
            "samples": write_jsonl(stage_a_root / f"{split}.jsonl", stagea_rows),
            "meta_samples": write_jsonl(stage_a_root / f"meta_{split}.jsonl", stagea_meta_rows),
        }
        stage_b_summary[split] = {
            "families": int(family_exported.get(split, 0)),
            "samples": write_jsonl(stage_b_root / f"{split}.jsonl", stageb_rows),
            "meta_samples": write_jsonl(stage_b_root / f"meta_{split}.jsonl", stageb_meta_rows),
        }

    with (stage_a_root / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(build_dataset_registry(stage_a_root, "unimapgen_geo_current_patch_only"), f, ensure_ascii=False, indent=2)
    with (stage_b_root / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(build_dataset_registry(stage_b_root, "unimapgen_geo_current_state"), f, ensure_ascii=False, indent=2)

    with (stage_a_root / "export_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_name_prefix": "unimapgen_geo_current_patch_only",
                "source_family_manifest": "",
                "summary": stage_a_summary,
                "empty_patch_filter": filter_summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with (stage_b_root / "export_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_name_prefix": "unimapgen_geo_current_state",
                "source_family_manifest": "",
                "summary": stage_b_summary,
                "empty_patch_filter": filter_summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "stage_a_root": stage_a_root,
        "stage_b_root": stage_b_root,
        "stage_a_summary": stage_a_summary,
        "stage_b_summary": stage_b_summary,
        "empty_patch_filter": filter_summary,
    }


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    ensure_dir(output_root)
    families = load_jsonl(Path(args.family_manifest).resolve())
    include_lane = True if (not bool(args.include_lane) and not bool(args.include_intersection_boundary)) else bool(args.include_lane)
    include_intersection_boundary = True if (not bool(args.include_lane) and not bool(args.include_intersection_boundary)) else bool(args.include_intersection_boundary)
    stagea_system_prompt = str(args.stagea_system_prompt).strip() if bool(args.use_system_prompt) else ""
    stageb_system_prompt = str(args.stageb_system_prompt).strip() if bool(args.use_system_prompt) else ""

    result = export_families_to_stage_datasets(
        families=families,
        output_root=output_root,
        splits=[str(split) for split in args.splits],
        band_indices=[int(x) for x in args.band_indices],
        mask_threshold=int(args.mask_threshold),
        resample_step_px=float(args.resample_step_px),
        boundary_tol_px=float(args.boundary_tol_px),
        trace_points=int(args.trace_points),
        state_mixture_mode=str(args.state_mixture_mode),
        state_no_state_ratio=float(args.state_no_state_ratio),
        state_weak_ratio=float(args.state_weak_ratio),
        state_full_ratio=float(args.state_full_ratio),
        state_weak_trace_points=int(args.state_weak_trace_points),
        state_line_dropout=float(args.state_line_dropout),
        state_point_jitter_px=float(args.state_point_jitter_px),
        state_truncate_prob=float(args.state_truncate_prob),
        include_lane=include_lane,
        include_intersection_boundary=include_intersection_boundary,
        max_families_per_split=int(args.max_families_per_split),
        empty_patch_drop_ratio=float(args.empty_patch_drop_ratio),
        empty_patch_seed=int(args.empty_patch_seed),
        empty_patch_drop_ratio_by_split=None,
        stagea_system_prompt=stagea_system_prompt,
        stagea_prompt_template=str(args.stagea_prompt_template),
        stageb_system_prompt=stageb_system_prompt,
        stageb_prompt_template=str(args.stageb_prompt_template),
    )

    stage_a_root = result["stage_a_root"]
    stage_b_root = result["stage_b_root"]
    stage_a_summary = result["stage_a_summary"]
    stage_b_summary = result["stage_b_summary"]
    filter_summary = result["empty_patch_filter"]

    with (stage_a_root / "export_summary.json").open("r", encoding="utf-8") as f:
        stage_a_export = json.load(f)
    stage_a_export["source_family_manifest"] = str(Path(args.family_manifest).resolve())
    with (stage_a_root / "export_summary.json").open("w", encoding="utf-8") as f:
        json.dump(stage_a_export, f, ensure_ascii=False, indent=2)

    with (stage_b_root / "export_summary.json").open("r", encoding="utf-8") as f:
        stage_b_export = json.load(f)
    stage_b_export["source_family_manifest"] = str(Path(args.family_manifest).resolve())
    with (stage_b_root / "export_summary.json").open("w", encoding="utf-8") as f:
        json.dump(stage_b_export, f, ensure_ascii=False, indent=2)

    for split in args.splits:
        split = str(split)
        print(
            f"[{split}] stage_a_samples={stage_a_summary[split]['samples']} "
            f"stage_b_samples={stage_b_summary[split]['samples']} "
            f"empty_generated={filter_summary[split]['generated_empty']} "
            f"empty_kept={filter_summary[split]['kept_empty']}",
            flush=True,
        )
    print(f"Saved Stage A and Stage B datasets under {output_root}", flush=True)


if __name__ == "__main__":
    main()
