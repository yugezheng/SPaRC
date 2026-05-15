import json
import re
import os
import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm


RERANK_SYSTEM_PROMPT = """You are a answer filter for a knowledge graph QA system.
Given a question and candidate paths (entity -> relation -> answer),
score each path from 0.0 to 1.0 based on whether the answer (the part after the last '->') 
is the correct TYPE of answer the question is asking for.

Rules:
- Focus ONLY on the answer entity after the last '->',ignore the relation name.
- Score 1.0 if the answer type matches what the question asks for.
- Score 0.0 if the answer is clearly wrong type (e.g. a date when question asks for a person).
- Score 0.5 if uncertain.
- Answers like 'm.xxxxxx' are Freebase IDs with no meaning, always score 0.0.

Output a JSON array of scores in the same order as the input paths.
Example output: [1.0, 0.5, 0.0]
Output the JSON array only, no explanation."""


def _safe_str(v) -> str:
    if isinstance(v, list):
        return ",".join(str(x) for x in v)
    return str(v) if v is not None else "None"


def extract_tail(path_text: str) -> str:
    parts = path_text.split("->")
    return parts[-1].strip() if len(parts) >= 2 else ""
class LLMReranker:
    def __init__(self, model_path: str, max_new_tokens: int = 64):
        print(f"[LLMReranker] Loading model from {model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        print("[LLMReranker] Model loaded.")

    def score_paths(self, question: str, path_texts: List[str],
                    constraints: List[Dict]) -> List[float]:

        if not path_texts:
            return []
        n = len(path_texts)
        tails_str = "\n".join(
            f"{i}. answer: {extract_tail(pt)}"
            for i, pt in enumerate(path_texts)
        )
        if constraints:
            constraint_desc = " | ".join(
                f"{c.get('description', '')} (expected type: {_safe_str(c.get('value'))})"
                for c in constraints
                if c.get("constraint_type") == "entity_type"
            )
        else:
            constraint_desc = "any relevant answer"

        user_content = (
            f"Question: {question}\n"
            f"The answer should be: {constraint_desc}\n\n"
            f"Candidate answers:\n{tails_str}\n\n"
            f"Output a JSON array of {n} scores (0.0 to 1.0), "
            f"same order as above."
        )

        messages = [
            {"role": "system", "content": RERANK_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        outputs = self.pipe(prompt, return_full_text=False)
        raw = outputs[0]["generated_text"].strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        raw = match.group(0) if match else "[]"
        open_b = raw.count("[") - raw.count("]")
        if open_b > 0:
            raw += "]" * open_b

        try:
            scores = json.loads(raw)
            if len(scores) != n:
                raise ValueError(f"length mismatch: got {len(scores)}, expect {n}")
            return [max(0.0, min(1.0, float(s))) for s in scores]
        except Exception as e:
            print(f"  [WARN] score parse failed: {e} | raw={raw[:150]}")
            return [
                0.0 if re.match(r"^m\.[0-9a-z_]+$",
                                extract_tail(pt), re.IGNORECASE)
                else 0.5
                for pt in path_texts
            ]
class ConstraintReranker:
    def __init__(self, llm_reranker: LLMReranker, alpha: float = 0.4):
        self.llm_reranker = llm_reranker
        self.alpha        = alpha

    def rerank(self, question: str, path_texts: List[str],
               path_keys: List[list], constraints: List[Dict],
               original_scores: List[float] = None) -> List[Dict]:

        n = len(path_texts)
        if original_scores is None:
            original_scores = [1.0 - i / n for i in range(n)]
        if not constraints:
            results = []
            for i in range(n):
                results.append({
                    "original_index": i,
                    "path_text":      path_texts[i],
                    "path_key":       path_keys[i],
                    "llm_score":      -1.0,
                    "original_score": round(original_scores[i], 4),
                    "final_score":    round(original_scores[i], 4),
                })
            results.sort(key=lambda x: x["final_score"], reverse=True)
            return results
        llm_scores = self.llm_reranker.score_paths(
            question, path_texts, constraints
        )

        results = []
        for i in range(n):
            final = self.alpha * llm_scores[i] + (1 - self.alpha) * original_scores[i]
            results.append({
                "original_index": i,
                "path_text":      path_texts[i],
                "path_key":       path_keys[i],
                "llm_score":      round(llm_scores[i],   4),
                "original_score": round(original_scores[i], 4),
                "final_score":    round(final,           4),
            })

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results
def run_reranking(constraints_file: str, output_file: str, model_path: str,
                  alpha: float = 0.4, max_new_tokens: int = 64):

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(constraints_file, "r", encoding="utf-8") as f:
        total = sum(1 for line in f if line.strip())

    llm_reranker = LLMReranker(model_path, max_new_tokens=max_new_tokens)
    reranker     = ConstraintReranker(llm_reranker, alpha=alpha)

    with open(constraints_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        pbar = tqdm(total=total, desc="Reranking paths",
                    unit="sample", dynamic_ncols=True)

        for line in fin:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)

            qid         = sample["id"]
            question    = sample["question"]
            path_texts  = sample["path_texts"]
            path_keys   = sample["path_keys"]
            constraints = sample["constraints"]
            key_paths   = sample.get("key_paths", [])

            if not path_texts:
                fout.write(json.dumps({
                    "id":                 qid,
                    "question":           question,
                    "constraints":        constraints,
                    "key_paths":          key_paths,
                    "reranked":           [],
                    "top1_for_inference": [],
                    "top3_for_inference": [],
                    "top5_for_inference": [],
                }, ensure_ascii=False) + "\n")
                pbar.update(1)
                continue

            reranked = reranker.rerank(
                question, path_texts, path_keys, constraints
            )

            top1 = reranked[0]
            pbar.set_postfix({
                "id":   qid,
                "top1": f"{top1['final_score']:.3f}",
                "path": top1["path_text"][:40] + "..."
            })

            record = {
                "id":                 qid,
                "question":           question,
                "constraints":        constraints,
                "key_paths":          key_paths,
                "reranked":           reranked,
                "top1_for_inference": [r["path_text"] for r in reranked[:1]],
                "top3_for_inference": [r["path_text"] for r in reranked[:3]],
                "top5_for_inference": [r["path_text"] for r in reranked[:5]],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            pbar.update(1)

        pbar.close()

    print(f"\n[Done] Reranked results saved to {output_file}")