import argparse
import json
from pathlib import Path

from build_geo_current_family_manifest import (
    DEFAULT_IMAGE_RELPATH,
    DEFAULT_INTERSECTION_RELPATH,
    DEFAULT_LANE_RELPATH,
    DEFAULT_MASK_RELPATH,
)
from build_patch_only_fixed_grid_targetbox_dataset import build_fixed_grid_targetbox_dataset
from export_llamafactory_both_from_geo_current_family_manifest import export_families_to_stage_datasets
from geo_current_dataset_v1_common import (
    DEFAULT_STAGEA_PROMPT_TEMPLATE,
    DEFAULT_STAGEA_SYSTEM_PROMPT,
    DEFAULT_STAGEB_PROMPT_TEMPLATE,
    DEFAULT_STAGEB_SYSTEM_PROMPT,
    build_manifest_for_dataset,
    ensure_dir,
    write_jsonl,
)


def _build_split_roots(args: argparse.Namespace) -> dict:
    split_roots = {}
    if str(args.train_root).strip():
        split_roots["train"] = Path(str(args.train_root).strip()).resolve()
    if str(args.val_root).strip():
        split_roots["val"] = Path(str(args.val_root).strip()).resolve()
    return split_roots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-click build manifest + Stage A/Stage B datasets from manual train/val roots.")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--manifest-path", type=str, default="")
    parser.add_argument("--dataset-root", type=str, default="/dataset/zsy/dataset-extracted")
    parser.add_argument("--train-root", type=str, default="")
    parser.add_argument("--val-root", type=str, default="")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"])
    parser.add_argument("--image-relpath", type=str, default=DEFAULT_IMAGE_RELPATH)
    parser.add_argument("--mask-relpath", type=str, default=DEFAULT_MASK_RELPATH)
    parser.add_argument("--lane-relpath", type=str, default=DEFAULT_LANE_RELPATH)
    parser.add_argument("--intersection-relpath", type=str, default=DEFAULT_INTERSECTION_RELPATH)
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--tile-size-px", type=int, default=896)
    parser.add_argument("--overlap-px", type=int, default=232)
    parser.add_argument("--keep-margin-px", type=int, default=116)
    parser.add_argument("--review-crop-pad-px", type=int, default=64)
    parser.add_argument("--tile-min-mask-ratio", type=float, default=0.02)
    parser.add_argument("--tile-min-mask-pixels", type=int, default=256)
    parser.add_argument("--tile-max-per-sample", type=int, default=0)
    parser.add_argument("--search-within-review-bbox", action="store_true")
    parser.add_argument("--fallback-to-all-if-empty", action="store_true")
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--band-indices", type=int, nargs="+", default=[1, 2, 3])
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
    parser.add_argument("--skip-fixed16-build", action="store_true")
    parser.add_argument("--fixed16-output-name", type=str, default="fixed16_stage_a")
    parser.add_argument("--fixed16-stageb-output-name", type=str, default="fixed16_stage_b")
    parser.add_argument("--fixed16-grid-size", type=int, default=4)
    parser.add_argument("--fixed16-target-empty-ratio", type=float, default=0.10)
    parser.add_argument("--fixed16-seed", type=int, default=42)
    parser.add_argument("--fixed16-max-source-samples-per-split", type=int, default=0)
    parser.add_argument("--fixed16-resample-step-px", type=float, default=4.0)
    parser.add_argument("--fixed16-boundary-tol-px", type=float, default=2.5)
    parser.add_argument("--fixed16-image-root-mode", type=str, default="symlink", choices=["symlink", "copy", "none"])
    parser.add_argument("--fixed16-export-visualizations", action="store_true")
    parser.add_argument("--fixed16-max-visualizations-per-split", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    ensure_dir(output_root)
    manifest_path = Path(args.manifest_path).resolve() if str(args.manifest_path).strip() else (output_root / "family_manifest.jsonl")

    split_roots = _build_split_roots(args)
    split_list = [str(x) for x in args.splits]
    families = []
    dataset_root = Path(args.dataset_root).resolve()
    if "train" in split_list:
        families.extend(
            build_manifest_for_dataset(
                dataset_root=dataset_root,
                splits=["train"],
                image_relpath=str(args.image_relpath),
                mask_relpath=str(args.mask_relpath),
                lane_relpath=str(args.lane_relpath),
                intersection_relpath=str(args.intersection_relpath),
                mask_threshold=int(args.mask_threshold),
                tile_size_px=int(args.tile_size_px),
                overlap_px=int(args.overlap_px),
                keep_margin_px=int(args.keep_margin_px),
                review_crop_pad_px=int(args.review_crop_pad_px),
                tile_min_mask_ratio=float(args.tile_min_mask_ratio),
                tile_min_mask_pixels=int(args.tile_min_mask_pixels),
                tile_max_per_sample=int(args.tile_max_per_sample),
                search_within_review_bbox=bool(args.search_within_review_bbox),
                fallback_to_all_if_empty=bool(args.fallback_to_all_if_empty),
                max_samples_per_split=int(args.max_samples_per_split),
                shard_index=int(args.shard_index),
                num_shards=int(args.num_shards),
                split_roots=split_roots or None,
            )
        )
    if "val" in split_list:
        families.extend(
            build_manifest_for_dataset(
                dataset_root=dataset_root,
                splits=["val"],
                image_relpath=str(args.image_relpath),
                mask_relpath=str(args.mask_relpath),
                lane_relpath=str(args.lane_relpath),
                intersection_relpath=str(args.intersection_relpath),
                mask_threshold=int(args.mask_threshold),
                tile_size_px=int(args.tile_size_px),
                overlap_px=int(args.overlap_px),
                keep_margin_px=int(args.keep_margin_px),
                review_crop_pad_px=int(args.review_crop_pad_px),
                tile_min_mask_ratio=0.0,
                tile_min_mask_pixels=0,
                tile_max_per_sample=0,
                search_within_review_bbox=False,
                fallback_to_all_if_empty=True,
                max_samples_per_split=int(args.max_samples_per_split),
                shard_index=int(args.shard_index),
                num_shards=int(args.num_shards),
                split_roots=split_roots or None,
            )
        )
    family_count = write_jsonl(manifest_path, families)

    include_lane = True if (not bool(args.include_lane) and not bool(args.include_intersection_boundary)) else bool(args.include_lane)
    include_intersection_boundary = True if (not bool(args.include_lane) and not bool(args.include_intersection_boundary)) else bool(args.include_intersection_boundary)
    stagea_system_prompt = str(args.stagea_system_prompt).strip() if bool(args.use_system_prompt) else ""
    stageb_system_prompt = str(args.stageb_system_prompt).strip() if bool(args.use_system_prompt) else ""

    export_result = export_families_to_stage_datasets(
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
        empty_patch_drop_ratio_by_split={"val": 0.0},
        stagea_system_prompt=stagea_system_prompt,
        stagea_prompt_template=str(args.stagea_prompt_template),
        stageb_system_prompt=stageb_system_prompt,
        stageb_prompt_template=str(args.stageb_prompt_template),
    )

    for stage_root in (export_result["stage_a_root"], export_result["stage_b_root"]):
        export_summary_path = Path(stage_root) / "export_summary.json"
        if export_summary_path.is_file():
            with export_summary_path.open("r", encoding="utf-8") as f:
                export_summary = json.load(f)
            export_summary["source_family_manifest"] = str(manifest_path)
            with export_summary_path.open("w", encoding="utf-8") as f:
                json.dump(export_summary, f, ensure_ascii=False, indent=2)

    fixed16_stagea_summary = None
    fixed16_stageb_summary = None
    fixed16_output_root = output_root / str(args.fixed16_output_name).strip()
    fixed16_stageb_output_root = output_root / str(args.fixed16_stageb_output_name).strip()
    if not bool(args.skip_fixed16_build):
        fixed16_stagea_summary = build_fixed_grid_targetbox_dataset(
            input_root=Path(export_result["stage_a_root"]),
            output_root=fixed16_output_root,
            splits=[str(x) for x in args.splits],
            grid_size=int(args.fixed16_grid_size),
            target_empty_ratio=float(args.fixed16_target_empty_ratio),
            target_empty_ratio_by_split={"val": 1.0},
            seed=int(args.fixed16_seed),
            max_source_samples_per_split=int(args.fixed16_max_source_samples_per_split),
            boundary_tol_px=float(args.fixed16_boundary_tol_px),
            resample_step_px=float(args.fixed16_resample_step_px),
            reuse_system_prompt=True,
            image_root_mode=str(args.fixed16_image_root_mode),
            export_visualizations=bool(args.fixed16_export_visualizations),
            max_visualizations_per_split=int(args.fixed16_max_visualizations_per_split),
        )
        fixed16_stageb_summary = build_fixed_grid_targetbox_dataset(
            input_root=Path(export_result["stage_b_root"]),
            output_root=fixed16_stageb_output_root,
            splits=[str(x) for x in args.splits],
            grid_size=int(args.fixed16_grid_size),
            target_empty_ratio=float(args.fixed16_target_empty_ratio),
            target_empty_ratio_by_split={"val": 1.0},
            seed=int(args.fixed16_seed),
            max_source_samples_per_split=int(args.fixed16_max_source_samples_per_split),
            boundary_tol_px=float(args.fixed16_boundary_tol_px),
            resample_step_px=float(args.fixed16_resample_step_px),
            reuse_system_prompt=True,
            image_root_mode=str(args.fixed16_image_root_mode),
            export_visualizations=bool(args.fixed16_export_visualizations),
            max_visualizations_per_split=int(args.fixed16_max_visualizations_per_split),
        )

    summary = {
        "output_root": str(output_root),
        "manifest_path": str(manifest_path),
        "family_count": int(family_count),
        "splits": [str(x) for x in args.splits],
        "split_roots": {k: str(v) for k, v in split_roots.items()},
        "manifest_config": {
            "tile_size_px": int(args.tile_size_px),
            "overlap_px": int(args.overlap_px),
            "keep_margin_px": int(args.keep_margin_px),
            "review_crop_pad_px": int(args.review_crop_pad_px),
            "tile_min_mask_ratio": float(args.tile_min_mask_ratio),
            "tile_min_mask_pixels": int(args.tile_min_mask_pixels),
            "tile_max_per_sample": int(args.tile_max_per_sample),
            "search_within_review_bbox": bool(args.search_within_review_bbox),
            "fallback_to_all_if_empty": bool(args.fallback_to_all_if_empty),
            "val_override": {
                "search_within_review_bbox": False,
                "tile_min_mask_ratio": 0.0,
                "tile_min_mask_pixels": 0,
                "tile_max_per_sample": 0,
                "fallback_to_all_if_empty": True,
            },
            "max_samples_per_split": int(args.max_samples_per_split),
            "shard_index": int(args.shard_index),
            "num_shards": int(args.num_shards),
        },
        "export_config": {
            "resample_step_px": float(args.resample_step_px),
            "boundary_tol_px": float(args.boundary_tol_px),
            "trace_points": int(args.trace_points),
            "state_mixture_mode": str(args.state_mixture_mode),
            "empty_patch_drop_ratio": float(args.empty_patch_drop_ratio),
            "empty_patch_seed": int(args.empty_patch_seed),
            "empty_patch_drop_ratio_by_split": {"val": 0.0},
        },
        "fixed16_config": {
            "enabled": not bool(args.skip_fixed16_build),
            "stage_a_output_root": str(fixed16_output_root),
            "stage_b_output_root": str(fixed16_stageb_output_root),
            "grid_size": int(args.fixed16_grid_size),
            "target_empty_ratio": float(args.fixed16_target_empty_ratio),
            "target_empty_ratio_by_split": {"val": 1.0},
            "seed": int(args.fixed16_seed),
            "resample_step_px": float(args.fixed16_resample_step_px),
            "boundary_tol_px": float(args.fixed16_boundary_tol_px),
            "image_root_mode": str(args.fixed16_image_root_mode),
        },
        "stage_a_summary": export_result["stage_a_summary"],
        "stage_b_summary": export_result["stage_b_summary"],
        "empty_patch_filter": export_result["empty_patch_filter"],
        "fixed16_stage_a_summary": fixed16_stagea_summary,
        "fixed16_stage_b_summary": fixed16_stageb_summary,
    }
    with (output_root / "build_manifest_and_datasets.summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Built manifest: {manifest_path}", flush=True)
    print(f"Family count: {family_count}", flush=True)
    for split in args.splits:
        split = str(split)
        filter_info = export_result["empty_patch_filter"][split]
        print(
            f"[{split}] stage_a={export_result['stage_a_summary'][split]['samples']} "
            f"stage_b={export_result['stage_b_summary'][split]['samples']} "
            f"empty_generated={filter_info['generated_empty']} "
            f"empty_kept={filter_info['kept_empty']}",
            flush=True,
        )
        if fixed16_stagea_summary is not None:
            fixed16_split = fixed16_stagea_summary["splits"].get(split, {})
            print(
                f"[{split}] fixed16_stage_a_rows={fixed16_split.get('written_rows', 0)} "
                f"fixed16_stage_a_empty_kept={fixed16_split.get('kept_empty', 0)}",
                flush=True,
            )
        if fixed16_stageb_summary is not None:
            fixed16_stageb_split = fixed16_stageb_summary["splits"].get(split, {})
            print(
                f"[{split}] fixed16_stage_b_rows={fixed16_stageb_split.get('written_rows', 0)} "
                f"fixed16_stage_b_empty_kept={fixed16_stageb_split.get('kept_empty', 0)}",
                flush=True,
            )
    print(f"Saved all outputs under {output_root}", flush=True)


if __name__ == "__main__":
    main()
