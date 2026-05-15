# SPaRC — Overview
Paper:Super Paths, Better Reasoning: Multi-Hop Knowledge Graphs Reasoning
with LLM via Super-Path Ranking and Constraint Verification
---
This repository implements a two-stage pipeline for knowledge graph question answering: super-path ranking and constraint verification.


## Two-Stage Pipeline

### Stage 1 — super-path ranking
Trains a dual/cross-encoder to retrieve relevant reasoning paths from the knowledge graph for a given question.

→ See ./Super-paths_rank_first_stage/README.md for setup and instructions.

### Stage 2 — constraint verification, Inference & Evaluation
Uses an LLM to extract constraints from the top-k retrieved paths, verifies them, generates answers, and evaluates against ground truth.

→ See ./constraint_verify_prediction/README_rerank_inference_eval.md for setup and instructions.
