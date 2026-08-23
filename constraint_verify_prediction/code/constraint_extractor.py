import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dataclasses import dataclass, field
from typing import List, Optional
from tqdm import tqdm
import os
@dataclass
class Constraint:
    constraint_type: str          # entity_type / relation / temporal / count
    description: str
    target: str                   # tail_entity / relation / head_entity
    value: Optional[str] = None   # person / location / language / year / number ...
    keywords: List[str] = field(default_factory=list)

SYSTEM_PROMPT = """You are a constraint analyzer for a knowledge graph QA system. Given a natural language question, extract constraints mentioned in the question. Constraints are of the following types:

1. entity_type – Specifies the category of any entity mentioned in the question or the target answer.
2. relation – Captures a relational constraint in the question.
3. temporal – Involves a time condition that applies to any part of the question.
4. count – Involves a numerical quantity, comparison, or ordering relevant to any part of the question.

Each constraint must be a JSON object containing exactly the following three fields:
- "constraint_type": string, one of "entity_type", "relation", "temporal", "count".
- "description": a short natural language description of the constraint.
- "value": the concrete value of the constraint.

Output a JSON array of constraints extracted from the question. Do not introduce constraints beyond the defined constraint types."""

def parse_constraints(raw: str, question: str,
                      use_relation_constraint: bool = False) -> List[Constraint]:
        raw = re.sub(r"```json|```", "", raw).strip()
        match = re.search(r"\[.*\]", raw, re.DOTALL)  
        if match:
            raw = match.group(0)
        else:
            objects = re.findall(r"\{[^{}]*\}", raw, re.DOTALL)
            if objects:
                raw = "[" + ",".join(objects) + "]"
                print(f"  [FIX] Truncated JSON recovered: {len(objects)} object(s)")
            else:
                print(f"  [WARN] Cannot recover JSON for: {question!r}")
                print(f"  Raw: {raw[:200]}")
                return []
        open_braces  = raw.count("{") - raw.count("}")
        open_brackets = raw.count("[") - raw.count("]")
        if open_braces > 0:
            raw += "}" * open_braces
        if open_brackets > 0:
            raw += "]" * open_brackets

        try:
            items = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  [WARN] JSON parse failed for: {question!r} | err={e}")
            print(f"  Raw: {raw[:300]}")
            return []

        constraints = []
        for item in items:
            ctype = item.get("constraint_type", "")
            if ctype == "relation" and not use_relation_constraint:
                continue
            raw_value = item.get("value")
            if isinstance(raw_value, list):
                value = raw_value[0] if raw_value else None
            else:
                value = raw_value

            constraints.append(Constraint(
                constraint_type=ctype,
                description=item.get("description", ""),
                target=item.get("target", "tail_entity"),
                value=value,          
                keywords=item.get("keywords", [])
            ))
        return constraints
class ConstraintExtractor:
    def __init__(self, model_path: str, max_new_tokens: int = 512,
                 batch_size: int = 8):          
        print(f"[ConstraintExtractor] Loading model from {model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size            
        print(f"[ConstraintExtractor] Model loaded. batch_size={batch_size}")

    def _build_prompt(self, question: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Question: {question}"}
        ]
        # Llama3 chat template
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    def extract_batch(self, questions: List[str],
                  use_relation_constraint: bool = False) -> List[List[Constraint]]:
        prompts = [self._build_prompt(q) for q in questions]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        results = []
        for i, output in enumerate(outputs):
            new_tokens = output[input_len:]
            raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            results.append(
                parse_constraints(raw, questions[i],
                                use_relation_constraint=use_relation_constraint)
            )
        return results
def run_extraction(input_file: str, output_file: str, model_path: str,
                   top_k: int = 5, max_new_tokens: int = 512,
                   batch_size: int = 8,
                   use_relation_constraint: bool = False):   
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"[Info] Total samples: {len(samples)}, batch_size={batch_size}, "
          f"use_relation_constraint={use_relation_constraint}")

    top_key = f"top{top_k}"
    extractor = ConstraintExtractor(model_path, max_new_tokens=max_new_tokens,
                                    batch_size=batch_size)

    with open(output_file, "w", encoding="utf-8") as fout:
        pbar = tqdm(total=len(samples), desc="Extracting constraints",
                    unit="sample", dynamic_ncols=True)

        for batch_start in range(0, len(samples), batch_size):
            batch = samples[batch_start: batch_start + batch_size]
            questions = [s["question"] for s in batch]
            batch_constraints = extractor.extract_batch(
                questions,
                use_relation_constraint=use_relation_constraint
            )

            for sample, constraints in zip(batch, batch_constraints):
                top_pred   = sample["top_predictions"].get(top_key, {})
                path_texts = top_pred.get("path_texts", [])
                path_keys  = top_pred.get("path_keys",  [])
                key_paths  = sample.get("key_paths", [])

                record = {
                    "id":          sample["id"],
                    "question":    sample["question"],
                    "top_key":     top_key,
                    "path_texts":  path_texts,
                    "path_keys":   path_keys,
                    "key_paths":   key_paths,      
                    "constraints": [
                        {
                            "constraint_type": c.constraint_type,
                            "description":     c.description,
                            "target":          c.target,
                            "value":           c.value,
                            "keywords":        c.keywords,
                        }
                        for c in constraints
                    ]
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            fout.flush()
            pbar.set_postfix({
                "batch": f"{batch_start // batch_size + 1}",
                "last_q": questions[-1][:30] + "..."
            })
            pbar.update(len(batch))

        pbar.close()

    print(f"\n[Done] Constraints saved to {output_file}")
