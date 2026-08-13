"""RefCOCO inference + evaluation — one-pass per question, no grouping.

Usage:
    python scripts/run_refcoco_inference.py --split refcoco_val_questions
    python scripts/run_refcoco_inference.py --split refcoco_test_questions
    python scripts/run_refcoco_inference.py --split refcoco_testB_questions
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path


def build_prompt(question_text: str, model_config, conv_mode: str) -> str:
    """Build LLaVA conversation prompt for a single question."""
    if model_config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question_text
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + question_text

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def extract_bbox(text: str) -> Optional[list[float]]:
    """Parse the first 4 numbers from model output as normalised bbox."""
    numbers = re.findall(r"(\d+\.\d+|\d+\.|\.\d+)", text)
    if len(numbers) < 4:
        pct = re.findall(r"(\d+)%?", text)
        nums = [float(p) for p in pct]
        if len(nums) >= 4:
            return nums[:4] if max(nums[:4]) > 1 else [n / 100 for n in nums[:4]]
        return None
    coords = [float(n) for n in numbers[:4]]
    return [max(0.0, min(1.0, c)) for c in coords]


def compute_iou(pred: list[float], gt: list[float]) -> float:
    """IoU between two [x1, y1, x2, y2] boxes."""
    ix1, iy1 = max(pred[0], gt[0]), max(pred[1], gt[1])
    ix2, iy2 = min(pred[2], gt[2]), min(pred[3], gt[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_pred = max(0.0, (pred[2] - pred[0]) * (pred[3] - pred[1]))
    area_gt = max(0.0, (gt[2] - gt[0]) * (gt[3] - gt[1]))
    union = area_pred + area_gt - inter
    return inter / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="RefCOCO inference + evaluation")
    parser.add_argument("--model-path", default="/opt/data/private/vanilla-llava/models/llava-v1.5-7b")
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--split", default="refcoco_val_questions",
                        choices=["refcoco_val_questions", "refcoco_test_questions", "refcoco_testB_questions"])
    parser.add_argument("--data-dir", default="/opt/data/private/vanilla-llava/data/refcoco")
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--model-name", default="llava-v1.5-7b")
    parser.add_argument("--pruning-method", default="random", choices=["random", "uniform"], help="Method for visual token pruning")
    parser.add_argument("--keep-ratio", type=float, default=1.0, help="Ratio of visual tokens to keep (0.0 to 1.0)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    question_file = data_dir / "converted" / f"{args.split}.jsonl"
    image_dir = data_dir / "images"
    answer_dir = data_dir / "answers" / args.split / args.model_name / f"{args.pruning_method}_{args.keep_ratio}"
    answer_dir.mkdir(parents=True, exist_ok=True)
    answer_file = answer_dir / "merge.jsonl"
    metrics_file = answer_dir / "metrics.txt"

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Load questions
    questions = []
    with open(question_file) as f:
        for line in f:
            questions.append(json.loads(line))

    print(f"Split:       {args.split}")
    print(f"Questions:   {len(questions)}")
    print()

    # Load model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name)

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'

    # Inference — one question at a time
    results = []
    start = time.time()

    with open(answer_file, "w") as ans_f:
        for q in tqdm(questions, desc="Questions", unit="q"):
            image_path = image_dir / q["image"]
            if not image_path.exists():
                print(f"  WARNING: missing {image_path}, skipping qid={q['question_id']}")
                continue

            image = Image.open(image_path).convert('RGB')
            image_tensor = process_images([image], image_processor, model.config)
            image_tensor = image_tensor.to(dtype=torch.float16, device='cuda')

            prompt = build_prompt(q["text"], model.config, args.conv_mode)
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            input_ids = input_ids.unsqueeze(0).to(device='cuda')

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image.size],
                    do_sample=args.temperature > 0,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    pruning_method=args.pruning_method,
                    keep_ratio=args.keep_ratio,
                )

            text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            result = {
                "question_id": q["question_id"],
                "prompt": q["text"],
                "text": text,
                "model_id": args.model_name,
            }
            results.append(result)
            ans_f.write(json.dumps(result) + "\n")

    elapsed = time.time() - start
    print(f"\nInference done in {elapsed/60:.1f} min ({elapsed/len(questions):.3f}s per question)")
    print(f"Output: {answer_file}")

    # Evaluation
    iou_thresholds = (0.5, 0.6, 0.7, 0.8, 0.9)
    gt_map = {q["question_id"]: q for q in questions}

    correct = {t: 0 for t in iou_thresholds}
    parse_failures = 0
    total = 0

    for pred_item in results:
        qid = pred_item["question_id"]
        if qid not in gt_map:
            continue
        gt = gt_map[qid]
        pred_bbox = extract_bbox(pred_item["text"])

        if pred_bbox is None:
            parse_failures += 1
            continue

        iou = compute_iou(pred_bbox, gt["bbox"])
        for t in iou_thresholds:
            if iou >= t:
                correct[t] += 1
        total += 1

    valid = total - parse_failures

    lines = [
        f"Total samples:      {len(results)}",
        f"Matched GT:         {total}",
        f"Parse failures:     {parse_failures}",
        f"Valid predictions:  {valid}",
        "",
        "Accuracy@IoU thresholds:",
    ]
    for t in iou_thresholds:
        lines.append(f"  Acc@IoU={t:.1f}:  {correct[t]:5d}/{total:5d}  ({100.0*correct[t]/total:.2f}%)")
    lines.append("")
    lines.append("Accuracy@IoU (failures → wrong):")
    for t in iou_thresholds:
        lines.append(f"  Acc@IoU={t:.1f}:  {correct[t]:5d}/{total:5d}  ({100.0*correct[t]/total:.2f}%)")

    report = "\n".join(lines)
    print()
    print(report)

    with open(metrics_file, "w") as f:
        f.write(report + "\n")
    print(f"\nResults saved to: {metrics_file}")


if __name__ == "__main__":
    main()
