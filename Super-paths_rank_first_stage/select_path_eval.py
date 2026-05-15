import json
import os
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from select_path import DualEncoder,  collate_eval, path_to_text, path_to_key  

def path_to_eval_key(path: list) -> tuple:
    return (path[0], tuple(path[1]))

def safe_model_name(model_name: str) -> str:
    return model_name.replace('/', '_').replace('\\', '_').replace('.', '_')
class EvalDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len=128):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        qid=sample['id']
        question = sample['question']
        topic_ent = sample.get('gold_link', {}).get('topic_ent', {})
        key_paths = sample.get('key_paths', [])
        paths = sample['paths']
        def path_to_eval_key(path):
            return (path[0], tuple(path[1]))
        eval_key_paths_keys = [path_to_eval_key(kp) for kp in key_paths] if key_paths else []
        eval_path_keys = [path_to_eval_key(p) for p in paths]
        encode_path_texts = [path_to_text(p, topic_ent, for_save=False) for p in paths]
        save_path_texts = [path_to_text(p, topic_ent, for_save=True) for p in paths]
        save_path_keys = [path_to_key(p) for p in paths] if paths else []  
        save_key_paths_keys = [path_to_key(kp) for kp in key_paths] if key_paths else []

        return {
            'id': qid,
            'question': question,
            'encode_path_texts': encode_path_texts,
            'save_path_texts': save_path_texts,      
            'eval_path_keys': eval_path_keys,       
            'save_path_keys': save_path_keys,       
            'eval_key_paths_keys': eval_key_paths_keys,  
            'save_key_paths_keys': save_key_paths_keys  
        }

def evaluate_and_save(model, tokenizer, test_file, device, base_model_name, output_dir,
                      top_ks=[1,3,5,10], batch_size=64):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = safe_model_name(base_model_name)

    pred_file = os.path.join(output_dir, f"{safe_name}_predictions_{timestamp}.jsonl")
    metric_file = os.path.join(output_dir, f"{safe_name}_eval_results_{timestamp}.json")
    test_samples = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                test_samples.append(json.loads(line))
    print(f"Loaded {len(test_samples)} test samples.")

    test_dataset = EvalDataset(test_samples, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_eval)
    hit_at_k = {k: 0 for k in top_ks}
    precision_at_k = {k: [] for k in top_ks}
    recall_at_k = {k: [] for k in top_ks}
    f1_at_k = {k: [] for k in top_ks}
    total_eval_samples = 0
    total_processed = 0

    with torch.no_grad():
        with open(pred_file, 'w', encoding='utf-8') as f_pred:
            for batch in tqdm(test_loader, desc="Evaluating and saving predictions"):
                for item in batch:
                    qid = item['id']
                    question = item['question']
                    encode_path_texts = item['encode_path_texts']
                    eval_path_keys = item['eval_path_keys']
                    eval_key_paths_keys = set(item['eval_key_paths_keys'])
                    save_path_texts = item['save_path_texts']
                    save_path_keys = item['save_path_keys']
                    save_key_paths_keys = item['save_key_paths_keys']

                    num_paths = len(encode_path_texts)
                    has_key = len(eval_key_paths_keys) > 0
                    q_emb = model.encode_question([question], tokenizer, device)
                    if num_paths > 0:
                        path_embs = model.encode_path(encode_path_texts, tokenizer, device)  
                        similarities = torch.matmul(q_emb, path_embs.T)[0].cpu().numpy()
                    else:
                        similarities = np.array([])

                    sorted_indices = np.argsort(similarities)[::-1] if num_paths > 0 else []
                    sample_result = {
                        'id': qid,
                        'question': question,
                        'num_paths': num_paths,
                        'key_paths': save_key_paths_keys,
                        'top_predictions': {}
                    }

                    for k in top_ks:
                        effective_k = min(k, num_paths)
                        if effective_k == 0:
                            hit = 0
                            prec = 0
                            rec = 0
                            f1 = 0
                            retrieved_save_keys = []
                            retrieved_save_texts = []
                        else:
                            retrieved_indices = sorted_indices[:effective_k].tolist()
                            retrieved_eval_keys = [eval_path_keys[i] for i in retrieved_indices]
                            correct = sum(1 for rk in retrieved_eval_keys if rk in eval_key_paths_keys)
                            retrieved_save_keys = [save_path_keys[i] for i in retrieved_indices]
                            retrieved_save_texts = [save_path_texts[i] for i in retrieved_indices]

                            hit = 1 if correct > 0 else 0
                            prec = correct / effective_k
                            rec = correct / len(eval_key_paths_keys) if has_key else 0
                            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

                        sample_result['top_predictions'][f'top{k}'] = {
                            'indices': retrieved_indices,
                            'path_keys': retrieved_save_keys, 
                            'path_texts': retrieved_save_texts, 
                            'correct_count': correct,
                            'hit': hit,
                            'precision': float(prec),
                            'recall': float(rec),
                            'f1': float(f1)
                        }

                        if has_key:
                            hit_at_k[k] += hit
                            precision_at_k[k].append(prec)
                            recall_at_k[k].append(rec)
                            f1_at_k[k].append(f1)

                    f_pred.write(json.dumps(sample_result) + '\n')
                    total_processed += 1
                    if has_key:
                        total_eval_samples += 1
    results = {}
    for k in top_ks:
        if total_eval_samples > 0:
            results[f'Hit@{k}'] = hit_at_k[k] / total_eval_samples
            results[f'Precision@{k}'] = np.mean(precision_at_k[k]) if precision_at_k[k] else 0.0
            results[f'Recall@{k}'] = np.mean(recall_at_k[k]) if recall_at_k[k] else 0.0
            results[f'F1@{k}'] = np.mean(f1_at_k[k]) if f1_at_k[k] else 0.0
        else:
            results[f'Hit@{k}'] = 0.0
            results[f'Precision@{k}'] = 0.0
            results[f'Recall@{k}'] = 0.0
            results[f'F1@{k}'] = 0.0

    results['num_samples_with_key'] = total_eval_samples
    results['num_samples_total'] = total_processed
    results['base_model'] = base_model_name
    results['timestamp'] = timestamp
    results['top_ks'] = list(top_ks)

    with open(metric_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation results saved to: {metric_file}")
    print(f"Predictions saved to: {pred_file}")
    print(f"Total samples processed: {total_processed}")
    print(f"Samples with key_paths: {total_eval_samples}")
    print("\n=== Average Metrics (on samples with key_paths) ===")
    for metric, value in results.items():
        if metric.startswith(('Hit', 'Precision', 'Recall', 'F1')):
            print(f"{metric}: {value:.4f}")

    return results


def load_model(model_path, base_model, projection_dim=256, device='cuda'):
    model = DualEncoder(base_model, projection_dim=projection_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_file = r"./data/WebQSP/test.json"
    base_model = "bert-large-uncased"#model dir
    model_path = "model.pt"#pt dir
    output_dir = f"./output" 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    projection_dim = 256
    top_ks = [1, 3, 5,10]
    batch_size = 16
    print(f"Loading tokenizer from {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    print(f"Loading model from {model_path}")
    model = load_model(model_path, base_model, projection_dim, device)
    results = evaluate_and_save(
        model=model,
        tokenizer=tokenizer,
        test_file=test_file,
        device=device,
        base_model_name=base_model,
        output_dir=output_dir,
        top_ks=top_ks,
        batch_size=batch_size
    )