import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from geo_current_dataset_v1_common import (
    DEFAULT_STAGEA_PROMPT_TEMPLATE,
    DEFAULT_STAGEA_SYSTEM_PROMPT,
    DEFAULT_STAGEB_PROMPT_TEMPLATE,
    DEFAULT_STAGEB_SYSTEM_PROMPT,
    apply_state_mode,
    build_full_segments_for_patch,
    build_owned_segments_by_patch,
    build_patch_image,
    build_patch_target_lines,
    build_patch_target_lines_float,
    ensure_dir,
    extract_state_lines,
    family_global_lines,
    local_lines_to_uv,
    load_family_raster_and_mask,
    load_jsonl,
    local_lines_to_global,
    parse_generated_json,
    sanitize_pred_lines_uv,
    uv_lines_to_local,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rollout inference on current-dataset family manifests.")
    parser.add_argument("--family-manifest", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--max-families", type=int, default=16)
    parser.add_argument("--family-ids", type=str, default="")
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--processor-path", type=str, default="")
    parser.add_argument("--engine", type=str, default="custom", choices=["custom", "llamafactory"])
    parser.add_argument("--template", type=str, default="qwen2_vl")
    parser.add_argument("--image-max-pixels", type=int, default=802816)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--resample-step-px", type=float, default=4.0)
    parser.add_argument("--boundary-tol-px", type=float, default=2.5)
    parser.add_argument("--trace-points", type=int, default=8)
    parser.add_argument("--state-mode", type=str, default="full", choices=["full", "empty", "no_state", "weak_state", "full_state"])
    parser.add_argument("--state-weak-trace-points", type=int, default=3)
    parser.add_argument("--state-line-dropout", type=float, default=0.40)
    parser.add_argument("--state-point-jitter-px", type=float, default=2.0)
    parser.add_argument("--state-truncate-prob", type=float, default=0.30)
    parser.add_argument("--include-lane", action="store_true")
    parser.add_argument("--include-intersection-boundary", action="store_true")
    parser.add_argument("--use-patch-only-prompt-when-empty", action="store_true")
    parser.add_argument("--export-visualizations", action="store_true")
    return parser.parse_args()


def resolve_torch_dtype(precision: str) -> torch.dtype:
    mode = str(precision).strip().lower()
    if mode == "fp32":
        return torch.float32
    if mode == "fp16":
        return torch.float16
    if mode == "bf16":
        return torch.bfloat16
    if not torch.cuda.is_available():
        return torch.float32
    major = 0
    try:
        major = int(str(torch.__version__).split("+")[0].split(".")[0])
    except Exception:
        major = 0
    if major < 2:
        return torch.float16
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


def build_conversation(prompt_text: str, system_text: str, image_ref: str) -> List[Dict[str, Any]]:
    conversation: List[Dict[str, Any]] = []
    if str(system_text).strip():
        conversation.append({"role": "system", "content": [{"type": "text", "text": system_text}]})
    user_text = prompt_text.replace("<image>\n", "", 1).replace("<image>", "", 1).strip()
    conversation.append(
        {
            "role": "user",
            "content": [{"type": "image", "image": image_ref}, {"type": "text", "text": user_text}],
        }
    )
    return conversation


def build_prompt_and_system(state_lines: Sequence[Dict], use_patch_only_prompt_when_empty: bool) -> Tuple[str, str]:
    if len(state_lines) == 0 and bool(use_patch_only_prompt_when_empty):
        return DEFAULT_STAGEA_PROMPT_TEMPLATE, DEFAULT_STAGEA_SYSTEM_PROMPT
    state_json = json.dumps({"lines": list(state_lines)}, ensure_ascii=False, separators=(",", ":"))
    return DEFAULT_STAGEB_PROMPT_TEMPLATE.format(state_json=state_json), DEFAULT_STAGEB_SYSTEM_PROMPT


def generate_with_custom_engine(
    prompt_text: str,
    system_text: str,
    image: Image.Image,
    image_ref: str,
    processor: AutoProcessor,
    model: torch.nn.Module,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    conversation = build_conversation(prompt_text, system_text, image_ref)
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(do_sample),
            temperature=float(temperature),
            top_p=float(top_p),
            use_cache=True,
        )
    prompt_len = int(inputs["input_ids"].shape[1])
    gen_ids = generated[:, prompt_len:]
    return processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def polyline_hausdorff_distance_px(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] < 2 or b.shape[0] < 2:
        return float("inf")

    def _directed(u: np.ndarray, v: np.ndarray) -> float:
        dists = np.linalg.norm(u[:, None, :] - v[None, :, :], axis=-1)
        return float(np.max(np.min(dists, axis=1)))

    return max(_directed(a, b), _directed(b, a))


def endpoint_error_px(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] < 2 or b.shape[0] < 2:
        return float("inf")
    forward = (float(np.linalg.norm(a[0] - b[0])) + float(np.linalg.norm(a[-1] - b[-1]))) / 2.0
    reverse = (float(np.linalg.norm(a[0] - b[-1])) + float(np.linalg.norm(a[-1] - b[0]))) / 2.0
    return min(forward, reverse)


def compute_patch_metrics(pred_lines: Sequence[Dict], gt_lines: Sequence[Dict], thresholds: Sequence[float]) -> Dict[str, float]:
    pred_arrays = [np.asarray(line.get("points", []), dtype=np.float32) for line in pred_lines]
    gt_arrays = [np.asarray(line.get("points", []), dtype=np.float32) for line in gt_lines]
    pred_arrays = [arr for arr in pred_arrays if arr.ndim == 2 and arr.shape[0] >= 2 and arr.shape[1] == 2]
    gt_arrays = [arr for arr in gt_arrays if arr.ndim == 2 and arr.shape[0] >= 2 and arr.shape[1] == 2]
    results: Dict[str, float] = {
        "pred_count": float(len(pred_arrays)),
        "gt_count": float(len(gt_arrays)),
    }
    candidates: List[Dict[str, float]] = []
    for pred_idx, pred in enumerate(pred_arrays):
        for gt_idx, gt in enumerate(gt_arrays):
            candidates.append(
                {
                    "pred_idx": float(pred_idx),
                    "gt_idx": float(gt_idx),
                    "hausdorff_px": polyline_hausdorff_distance_px(pred, gt),
                    "endpoint_px": endpoint_error_px(pred, gt),
                }
            )
    candidates.sort(key=lambda item: (item["hausdorff_px"], item["endpoint_px"]))
    for threshold in thresholds:
        used_pred = set()
        used_gt = set()
        matches: List[Dict[str, float]] = []
        for cand in candidates:
            if cand["hausdorff_px"] > float(threshold):
                continue
            pred_idx = int(cand["pred_idx"])
            gt_idx = int(cand["gt_idx"])
            if pred_idx in used_pred or gt_idx in used_gt:
                continue
            used_pred.add(pred_idx)
            used_gt.add(gt_idx)
            matches.append(cand)
        precision = float(len(matches)) / float(max(1, len(pred_arrays)))
        recall = float(len(matches)) / float(max(1, len(gt_arrays)))
        f1 = 0.0 if precision + recall <= 1e-9 else 2.0 * precision * recall / (precision + recall)
        prefix = f"h{int(round(float(threshold)))}"
        results[f"{prefix}_precision"] = precision
        results[f"{prefix}_recall"] = recall
        results[f"{prefix}_f1"] = f1
        results[f"{prefix}_matched"] = float(len(matches))
        results[f"{prefix}_mean_hausdorff_px"] = float(np.mean([m["hausdorff_px"] for m in matches])) if matches else float("inf")
        results[f"{prefix}_mean_endpoint_px"] = float(np.mean([m["endpoint_px"] for m in matches])) if matches else float("inf")
    return results


def render_panel(
    image: Image.Image,
    lines: Sequence[Dict],
    state_lines: Sequence[Dict],
    title: str,
    line_color: Tuple[int, int, int],
) -> Image.Image:
    panel = image.convert("RGB")
    draw = ImageDraw.Draw(panel)
    for line in lines:
        pts = [tuple(int(v) for v in p) for p in line.get("points", [])]
        if len(pts) >= 2:
            draw.line(pts, fill=line_color, width=3)
    for line in state_lines:
        pts = [tuple(int(v) for v in p) for p in line.get("points", [])]
        if len(pts) >= 2:
            draw.line(pts, fill=(255, 160, 40), width=4)
    draw.text((8, 8), title, fill=(255, 255, 255))
    return panel


def stack_panels(left: Image.Image, right: Image.Image) -> Image.Image:
    width = int(left.width + right.width)
    height = int(max(left.height, right.height))
    canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas


def mean_metric(records: Sequence[Dict[str, Any]], key: str) -> float:
    values = [float(item["metrics"][key]) for item in records if key in item.get("metrics", {}) and math.isfinite(float(item["metrics"][key]))]
    return float(sum(values) / max(1, len(values))) if values else float("nan")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    predictions_dir = output_root / "predictions"
    metrics_dir = output_root / "metrics"
    viz_dir = output_root / "visualizations"
    for path in [output_root, predictions_dir, metrics_dir, viz_dir]:
        ensure_dir(path)

    families = load_jsonl(Path(args.family_manifest).resolve())
    wanted_family_ids = {x.strip() for x in str(args.family_ids).split(",") if x.strip()}
    selected_families: List[Dict] = []
    for family in families:
        if str(family.get("split")) != str(args.split):
            continue
        if wanted_family_ids and str(family.get("family_id")) not in wanted_family_ids:
            continue
        selected_families.append(family)
        if int(args.max_families) > 0 and len(selected_families) >= int(args.max_families):
            break

    processor = None
    model = None
    chat_model = None
    if args.engine == "custom":
        processor_path = args.processor_path or args.adapter
        processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
        torch_dtype = resolve_torch_dtype(args.precision)
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.base_model,
            torch_dtype=torch_dtype,
            device_map="auto" if str(args.device).startswith("cuda") else None,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, args.adapter)
        if not str(args.device).startswith("cuda"):
            model = model.to(args.device)
        model.eval()
    else:
        from llamafactory.chat import ChatModel

        infer_dtype = "bfloat16"
        resolved_dtype = resolve_torch_dtype(args.precision)
        if resolved_dtype == torch.float16:
            infer_dtype = "float16"
        elif resolved_dtype == torch.float32:
            infer_dtype = "float32"

        infer_args = {
            "model_name_or_path": args.base_model,
            "adapter_name_or_path": args.adapter,
            "finetuning_type": "lora",
            "stage": "sft",
            "template": args.template,
            "infer_backend": "huggingface",
            "infer_dtype": infer_dtype,
            "trust_remote_code": True,
            "image_max_pixels": int(args.image_max_pixels),
        }
        chat_model = ChatModel(infer_args)

    include_lane = True if (not bool(args.include_lane) and not bool(args.include_intersection_boundary)) else bool(args.include_lane)
    include_intersection_boundary = True if (not bool(args.include_lane) and not bool(args.include_intersection_boundary)) else bool(args.include_intersection_boundary)

    overall_results: List[Dict[str, Any]] = []
    overall_patch_records: List[Dict[str, Any]] = []
    for family in selected_families:
        raw_image_hwc, review_mask = load_family_raster_and_mask(family=family)
        global_lines = family_global_lines(
            family=family,
            include_lane=include_lane,
            include_intersection=include_intersection_boundary,
        )
        pred_owned_segments_by_patch: Dict[int, List[Dict]] = {}
        family_records: List[Dict[str, Any]] = []
        for patch in sorted(list(family["patches"]), key=lambda item: int(item["patch_id"])):
            patch_id = int(patch["patch_id"])
            patch_image = build_patch_image(raw_image_hwc=raw_image_hwc, patch=patch)
            gt_segments = build_full_segments_for_patch(
                patch=patch,
                global_lines=global_lines,
                review_mask=review_mask,
                resample_step_px=float(args.resample_step_px),
                boundary_tol_px=float(args.boundary_tol_px),
            )
            gt_lines = build_patch_target_lines_float(gt_segments, patch=patch)
            raw_state_lines = extract_state_lines(
                patch=patch,
                family=family,
                owned_segments_by_patch=pred_owned_segments_by_patch,
                trace_points=int(args.trace_points),
                boundary_tol_px=float(args.boundary_tol_px),
            )
            patch_size = int(patch["crop_box"]["x_max"] - patch["crop_box"]["x_min"])
            state_lines = apply_state_mode(
                raw_state_lines=raw_state_lines,
                state_mode=str(args.state_mode),
                patch_size=patch_size,
                weak_trace_points=int(args.state_weak_trace_points),
                state_line_dropout=float(args.state_line_dropout),
                state_point_jitter_px=float(args.state_point_jitter_px),
                state_truncate_prob=float(args.state_truncate_prob),
                rng=np.random.default_rng(seed=patch_id),
            )
            state_lines_uv = local_lines_to_uv(state_lines, patch=patch)
            prompt_text, system_text = build_prompt_and_system(
                state_lines=state_lines_uv,
                use_patch_only_prompt_when_empty=bool(args.use_patch_only_prompt_when_empty),
            )
            image_ref = f"{family['family_id']}_p{patch_id:04d}.png"
            if args.engine == "custom":
                pred_text = generate_with_custom_engine(
                    prompt_text=prompt_text,
                    system_text=system_text,
                    image=patch_image,
                    image_ref=image_ref,
                    processor=processor,
                    model=model,
                    max_new_tokens=int(args.max_new_tokens),
                    do_sample=bool(args.do_sample),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                )
            else:
                user_text = prompt_text.replace("<image>\n", "", 1).replace("<image>", "", 1).strip()
                responses = chat_model.chat(
                    messages=[{"role": "user", "content": user_text}],
                    system=system_text,
                    images=[patch_image],
                    do_sample=bool(args.do_sample),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    max_new_tokens=int(args.max_new_tokens),
                )
                pred_text = responses[0].response_text if responses else ""
            pred_obj, cleaned_pred = parse_generated_json(pred_text)
            pred_lines_uv = sanitize_pred_lines_uv(list((pred_obj or {"lines": []}).get("lines", [])))
            pred_lines = uv_lines_to_local(pred_lines_uv, patch=patch)
            pred_global_lines = local_lines_to_global(pred_lines, patch=patch)
            pred_owned_segments_by_patch.update(
                build_owned_segments_by_patch(
                    family={"patches": [patch]},
                    global_lines=pred_global_lines,
                    review_mask=review_mask,
                    resample_step_px=float(args.resample_step_px),
                    boundary_tol_px=float(args.boundary_tol_px),
                )
            )
            metrics = compute_patch_metrics(pred_lines=pred_lines, gt_lines=gt_lines, thresholds=[2.0, 4.0, 8.0])
            record = {
                "patch_id": patch_id,
                "prompt_text": prompt_text,
                "state_lines": state_lines_uv,
                "state_lines_float": state_lines,
                "raw_state_lines": raw_state_lines,
                "pred_text": pred_text,
                "pred_json_text": cleaned_pred,
                "pred_lines": pred_lines_uv,
                "pred_lines_float": pred_lines,
                "gt_lines": gt_lines,
                "parse_ok": pred_obj is not None,
                "metrics": metrics,
            }
            family_records.append(record)
            overall_patch_records.append(record)
            if bool(args.export_visualizations):
                family_viz_dir = viz_dir / str(family["family_id"])
                ensure_dir(family_viz_dir)
                gt_panel = render_panel(patch_image, gt_lines, state_lines, f"{family['family_id']}_p{patch_id:04d} | GT", (40, 220, 255))
                pred_panel = render_panel(patch_image, pred_lines, state_lines, f"{family['family_id']}_p{patch_id:04d} | Pred", (255, 80, 80))
                stack_panels(gt_panel, pred_panel).save(family_viz_dir / f"p{patch_id:04d}.png")

        family_summary = {
            "family_id": str(family["family_id"]),
            "num_patches": len(family_records),
            "num_parse_ok": sum(1 for item in family_records if bool(item["parse_ok"])),
            "mean_h2_f1": mean_metric(family_records, "h2_f1"),
            "mean_h4_f1": mean_metric(family_records, "h4_f1"),
            "mean_h8_f1": mean_metric(family_records, "h8_f1"),
        }
        with (predictions_dir / f"{family['family_id']}.json").open("w", encoding="utf-8") as f:
            json.dump({"family_id": family["family_id"], "patches": family_records}, f, ensure_ascii=False, indent=2)
        with (metrics_dir / f"{family['family_id']}.json").open("w", encoding="utf-8") as f:
            json.dump(family_summary, f, ensure_ascii=False, indent=2)
        overall_results.append(family_summary)

    overall_summary = {
        "num_families": len(overall_results),
        "num_patches": int(sum(item["num_patches"] for item in overall_results)),
        "num_parse_ok": int(sum(item["num_parse_ok"] for item in overall_results)),
        "mean_h2_f1": mean_metric(overall_patch_records, "h2_f1"),
        "mean_h4_f1": mean_metric(overall_patch_records, "h4_f1"),
        "mean_h8_f1": mean_metric(overall_patch_records, "h8_f1"),
        "families": overall_results,
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(overall_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
