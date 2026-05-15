# Constraint Extraction, Reranking, Inference & Evaluation Pipeline

This module takes top-k retrieved paths and runs them through four sequential stages: constraint extraction, path reranking, answer inference, and evaluation.

---

## Project Structure

```
.
├── code/
│   ├── config.py                  # Centralized config for Steps 1–2 (edit before running)
│   ├── run_extract.py             # Step 1: LLM-based constraint extraction
│   ├── run_rerank.py              # Step 2: path reranking with extracted constraints
│   ├── eval.py                    # Step 4: evaluation
│   ├── constraint_extractor.py    # Extraction logic (called by run_extract.py)
│   └── path_reranker.py           # Reranking logic (called by run_rerank.py)
└── prediction/
    ├── eval.py                    # Step 4: evaluation
    ├── ground/
    │   └── predictions.jsonl      # Ground truth answers
    ├── cwq/
    │   └── prediction.py          # Step 3: answer inference for CWQ
    └── webqsp/
        └── prediction.py          # Step 3: answer inference for WebQSP
```

---

## Environment Setup

```bash
conda create -n stage2 python=3.10 -y
conda activate stage2
pip install -r requirements.txt
```
---

## Step 1 — Constraint Extraction

Edit **`config.py`** before running:

```python
# ⚠️ Modify these before running

MODEL_PATH = "/path/to/llama3.1-8B-Instruct"   # local LLM path

# I/O
INPUT_FILE       = "./mapping.jsonl"              # top-k predictions from retrieval stage
CONSTRAINTS_FILE = "./constraintoutput.jsonl"     # output of Step 1, input of Step 2
RERANKED_FILE    = "./reranked_output.jsonl"      # output of Step 2, input of Step 3

# Extraction settings
TOP_K                   = 5      # number of top paths to consider
MAX_NEW_TOKENS          = 512
BATCH_SIZE              = 8
USE_RELATION_CONSTRAINT = False  # whether to apply relation-level constraints

# Reranking settings
RERANK_ALPHA = 0.35   # score interpolation weight
RULE_WEIGHT  = 0.6    # weight for rule-based score
LLM_WEIGHT   = 0.4    # weight for LLM-based score
TEMPERATURE  = 0.0
```

```bash
python run_extract.py
```

---

## Step 2 — Path verify-Reranking

Uses `CONSTRAINTS_FILE` (output of Step 1) and writes reranked results to `RERANKED_FILE`. No additional config changes needed if `config.py` is already set.

```bash
python run_rerank.py
```

---

## Step 3 — Answer Inference

Edit the constants at the top of **`prediction.py`** before running:

```python
# ⚠️ Modify these before running
INPUT_FILE     = "./constraint_verify/data"        # reranked output from Step 2
GT_FILE        = "./cwq/ground/predictions.jsonl"  # ground truth JSONL
OUTPUT_ROOT    = "./cwq/results/"                  # directory for inference output
MODEL_PATH     = "./llama3.1-8B-Instruct"          # local LLM path
MAX_NEW_TOKENS = 128
```

```bash
python prediction.py
```

Output is saved to `OUTPUT_ROOT/<timestamp>/predictions.jsonl`. The script supports **resuming** — if interrupted, it skips already-processed IDs on re-run.

---

## Step 4 — Evaluation

Edit the `predict_file` path at the bottom of **`eval.py`** before running:

```python
# ⚠️ Modify this before running
predict_file = "./cwq/results/<timestamp>/predictions.jsonl"  # output from Step 3
```

```bash
python eval.py
```

Results are saved alongside the predictions file under `eval_<timestamp>/`:

- `eval_summary.txt` — Hit, Hit@1, Macro/Micro Precision, Recall, F1
- `eval_detailed.jsonl` — per-sample breakdown

---

## Quick Reference

```bash
# 1. Create environment
conda create -n stage2 python=3.10 -y && conda activate stage2
pip install -r requirements.txt

# 2. Edit config.py  (MODEL_PATH, INPUT_FILE, CONSTRAINTS_FILE, RERANKED_FILE)

# 3. Constraint extraction
python run_extract.py

# 4. Reranking
python run_rerank.py

# 5. Edit prediction.py  (INPUT_FILE, GT_FILE, OUTPUT_ROOT, MODEL_PATH)
# 6. Answer inference
python prediction.py

# 7. Edit eval.py  (predict_file)
# 8. Evaluate
python eval.py
```

