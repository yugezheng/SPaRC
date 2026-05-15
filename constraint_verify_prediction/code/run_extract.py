from config import *
from constraint_extractor import run_extraction

if __name__ == "__main__":
    run_extraction(
        input_file              = INPUT_FILE,
        output_file             = CONSTRAINTS_FILE,
        model_path              = MODEL_PATH,
        top_k                   = TOP_K,
        max_new_tokens          = MAX_NEW_TOKENS,
        batch_size              = BATCH_SIZE,
        use_relation_constraint = USE_RELATION_CONSTRAINT,  
    )