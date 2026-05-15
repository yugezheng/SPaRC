#!/usr/bin/env python3

import json
import re
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


INPUT_FILE  = "./constraint_verify/data" #your verify dir
GT_FILE     = "./cwq/ground/predictions.jsonl"
OUTPUT_ROOT = "./cwq/results/"
MODEL_PATH  = "./llama3.1-8B-Instruct"

MAX_NEW_TOKENS = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_ground_truth(gt_file: str) -> dict:
    gt = {}
    with open(gt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj["id"]
            answers = obj.get("answers", obj.get("ground_truth", []))
            if isinstance(answers, str):
                answers = [answers]
            gt[str(qid)] = answers
    return gt


def load_done_ids(output_file: Path) -> set:
    done = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["id"])
                    except Exception:
                        pass
    return done




def select_paths(item: dict) -> tuple:
    reranked = item.get("reranked", [])
    if not reranked:
        return [], 1

    all_paths  = [r["path_text"] for r in reranked]
    top5_paths = item.get("top5_for_inference", all_paths[:5])

    top3_paths = [r["path_text"] for r in reranked[:3]]
    candidates = _extract_candidates(top3_paths)

    expected_count = 1 if len(candidates) == 1 else max(2, min(len(candidates), 5))

    return top5_paths, expected_count


def _extract_candidates(paths: list) -> list:
    freebase_re = re.compile(r"^m\.[0-9a-z_]+$", re.IGNORECASE)
    seen = []
    for path in paths:
        parts = path.split("->")
        if len(parts) < 2:
            continue
        for c in re.split(r"[|,]", parts[-1]):
            c = c.strip()
            if c and not freebase_re.match(c) and c not in seen:
                seen.append(c)
    return seen



def build_prompt(question: str, paths: list, expected_count: int) -> str:
    path_block = "\n".join(f"  [{i+1}] {p}" for i, p in enumerate(paths))
    prompt = (
        "You are a precise question-answering assistant.\n"
        "Given a question and knowledge-graph paths "
        "(entity -> relation -> answer_candidate), "
        "select the correct answer(s).\n\n"
        "Rules:\n"
        "1. Judge each path against the question — not all paths are correct.\n"
        "2. If unsure which path is correct, use the FIRST path's answer as default.\n"  
        "3. Reject paths whose answer is a Freebase ID like 'm.xxxxxxx' — never output these.\n"
        "4. Copy answer values EXACTLY as they appear after the last '->' in the paths.\n"
        "5. Separate multiple answers with ' | ' only — no 'and', no commas, no periods.\n"
        "6. No explanation, no extra words. Output ONE line only.\n\n"
        f"Question: {question}\n\n"
        "Knowledge-graph paths:\n"
        f"{path_block}\n\n"
        "ans: <Answer1> | <Answer2> | ..."
    )
    return prompt



FREEBASE_RE = re.compile(r"^m\.[0-9a-z_]+$", re.IGNORECASE)


def normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def clean_answers(parts: list) -> list:
    seen_norm = set()
    result = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if FREEBASE_RE.match(p):
            continue
        n = normalize(p)
        if n not in seen_norm:
            seen_norm.add(n)
            result.append(p)
    return result


def extract_answer(raw: str, fallback_path: str = "") -> list:
    def split_answers(text: str) -> list:
        parts = [p.strip() for p in text.split("|")]
        if len(parts) == 1:
            parts = [p.strip() for p in parts[0].split(",")]
        return parts

    for line in raw.splitlines():
        line = line.strip()
        m = re.match(r"(?i)^ans\s*:\s*(.+)$", line)
        if m:
            result = clean_answers(split_answers(m.group(1)))
            if result:
                return result

    for line in raw.splitlines():
        line = line.strip()
        if line:
            result = clean_answers(split_answers(line))
            if result:
                return result

    if fallback_path:
        tail = fallback_path.split("->")[-1].strip()
        fallback = clean_answers([p.strip() for p in tail.split(",")])
        if fallback:
            print(f"  [FALLBACK] using top1 tail: {fallback}")
            return fallback

    return []



def load_model(model_path: str):
    print(f"[INFO] Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[INFO] Loading model on {DEVICE} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    model.eval()
    print("[INFO] Model loaded.")
    return tokenizer, model

def infer(prompt: str, tokenizer, model) -> str:
    try:
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(DEVICE)
    except Exception:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

    prompt_len = input_ids.shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def main():
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir  = Path(OUTPUT_ROOT) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "predictions.jsonl"
    print(f"[INFO] Results will be saved to: {output_file}")

    done_ids = load_done_ids(output_file)
    if done_ids:
        print(f"[INFO] Resuming — {len(done_ids)} items already done.")

    print(f"[INFO] Loading ground truth from {GT_FILE} ...")
    gt_map = load_ground_truth(GT_FILE)
    print(f"[INFO] Ground truth: {len(gt_map)} entries.")

    tokenizer, model = load_model(MODEL_PATH)

    total = skipped = resumed = 0
    count_dist = {}

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(output_file, "a", encoding="utf-8") as fout:

        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {line_no}: JSON parse error – {e}")
                skipped += 1
                continue

            qid      = str(item.get("id", line_no))
            question = item.get("question", "")

            if qid in done_ids:
                resumed += 1
                continue

            if not question:
                print(f"[WARN] Line {line_no} (id={qid}): missing question, skipping.")
                skipped += 1
                continue

            paths, expected_count = select_paths(item)
            if not paths:
                print(f"[WARN] Line {line_no} (id={qid}): no paths, skipping.")
                skipped += 1
                continue

            count_dist[expected_count] = count_dist.get(expected_count, 0) + 1

            prompt     = build_prompt(question, paths, expected_count)
            raw_out    = infer(prompt, tokenizer, model)
            top1_path  = paths[0] if paths else ""
            prediction = extract_answer(raw_out, fallback_path=top1_path)

            ground_truth = gt_map.get(qid, [])

            pred_norm = {normalize(p) for p in prediction}
            gt_norm   = {normalize(g) for g in ground_truth}
            hit = "✓" if pred_norm & gt_norm else "✗"

            record = {
                "id":             qid,
                "question":       question,
                "prediction":     prediction,
                "ground_truth":   ground_truth,
                "raw_output":     raw_out,
                "paths_used":     paths,
                "expected_count": expected_count,
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            total += 1
            print(
                f"[{total:>5}] {hit} (exp={expected_count}) "
                f"id={qid} | pred={prediction} | gt={ground_truth}"
            )

    print(f"\n[DONE] New:{total} | Skip:{skipped} | Resume:{resumed}")
    print(f"[DONE] Expected-count distribution: {dict(sorted(count_dist.items()))}")
    print(f"[DONE] Output: {output_file}")


if __name__ == "__main__":
    main()