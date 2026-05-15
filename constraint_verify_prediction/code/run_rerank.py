from config import *
from path_reranker import run_reranking

if __name__ == "__main__":
    run_reranking(
        constraints_file = CONSTRAINTS_FILE,
        output_file      = RERANKED_FILE,
        model_path       = MODEL_PATH,
        alpha            = 0.4,
        max_new_tokens   = 64,  
    )