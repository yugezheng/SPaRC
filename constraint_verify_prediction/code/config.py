MODEL_PATH = "/home/admin1/code/zyh/model/llama3.1-8B-Instruct"
TOP_K = 5
RERANK_ALPHA = 0.35
MAX_NEW_TOKENS = 512        
TEMPERATURE = 0.0
BATCH_SIZE = 8
RULE_WEIGHT = 0.6
LLM_WEIGHT = 0.4
USE_RELATION_CONSTRAINT = False

INPUT_FILE   = "./mapping.jsonl"   # your top-k input
CONSTRAINTS_FILE = "./constraintoutput.jsonl"  # constraintoutput dir
RERANKED_FILE    = "./output dir"     # verify rerank output dir