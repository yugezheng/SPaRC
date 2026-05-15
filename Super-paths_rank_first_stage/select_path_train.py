import json
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from typing import List, Dict, Tuple
import argparse
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def path_to_text(path: List, topic_ent: Dict[str, str]) -> str:
    head_id = path[0]
    relations = path[1]
    head_name = topic_ent.get(head_id, head_id)
    relation_text = ", ".join(relations)
    return f"{head_name} [{head_id}] -> {relation_text}"


def path_to_key(path: List) -> Tuple:
    return (path[0], tuple(path[1]))


class PathRetrievalDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len=128, is_training=True, neg_num=1):
        self.samples = [s for s in samples if s.get('key_paths')]
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_training = is_training
        self.neg_num = neg_num

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        question = sample['question']
        topic_ent = sample.get('gold_link', {}).get('topic_ent', {})
        key_paths = sample.get('key_paths', [])

        if not self.is_training:
            paths = sample['paths']
            key_paths_keys = [path_to_key(kp) for kp in key_paths]
            path_texts = [path_to_text(p, topic_ent) for p in paths]
            path_keys = [path_to_key(p) for p in paths]
            return {
                'question': question,
                'path_texts': path_texts,
                'path_keys': path_keys,
                'key_paths_keys': key_paths_keys
            }

        pos_path = random.choice(key_paths)
        pos_text = path_to_text(pos_path, topic_ent)

        paths = sample['paths']
        key_set = set(path_to_key(kp) for kp in key_paths)
        neg_candidates = [p for p in paths if path_to_key(p) not in key_set]
        if not neg_candidates:
            neg_candidates = paths
        sampled_negs = random.choices(neg_candidates, k=self.neg_num)
        neg_texts = [path_to_text(p, topic_ent) for p in sampled_negs]

        return question, pos_text, neg_texts


def collate_train(batch):
    questions, pos_texts, neg_texts_list = zip(*batch)
    return list(questions), list(pos_texts), list(neg_texts_list)


def collate_eval(batch):
    return batch


class DualEncoder(nn.Module):
    def __init__(self, base_model_name, projection_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        self.projection = nn.Linear(self.encoder.config.hidden_size, projection_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_vec = outputs.last_hidden_state[:, 0, :]
        projected = self.projection(cls_vec)
        projected = nn.functional.normalize(projected, p=2, dim=-1)
        return projected

    def encode_question(self, questions, tokenizer, device, max_len=128):
        encoded = tokenizer(questions, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
        return self.forward(encoded['input_ids'].to(device), encoded['attention_mask'].to(device))

    def encode_path(self, path_texts, tokenizer, device, max_len=128):
        encoded = tokenizer(path_texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
        return self.forward(encoded['input_ids'].to(device), encoded['attention_mask'].to(device))


def train_epoch(model, dataloader, optimizer, tokenizer, device, args):
    model.train()
    total_loss = 0

    for questions, pos_texts, neg_texts_list in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()

        q_emb = model.encode_question(questions, tokenizer, device, args.max_len)   # [B, D]
        p_emb = model.encode_path(pos_texts, tokenizer, device, args.max_len)       # [B, D]

        all_neg_texts = []
        neg_counts = []
        for negs in neg_texts_list:
            all_neg_texts.extend(negs)
            neg_counts.append(len(negs))
        neg_embs = model.encode_path(all_neg_texts, tokenizer, device, args.max_len) if all_neg_texts else None

        B = q_emb.size(0)
        temperature = args.temperature
        losses = []
        start_neg = 0

        for i in range(B):
            pos_sim = torch.sum(q_emb[i] * p_emb[i]) / temperature
            neg_sims = [torch.sum(q_emb[i] * p_emb[j]) / temperature for j in range(B) if j != i]

            cnt = neg_counts[i]
            if neg_embs is not None and cnt > 0:
                for k in range(start_neg, start_neg + cnt):
                    neg_sims.append(torch.sum(q_emb[i] * neg_embs[k]) / temperature)
            start_neg += cnt

            if not neg_sims:
                continue
            logits = torch.cat([pos_sim.unsqueeze(0), torch.stack(neg_sims)])
            labels = torch.zeros(1, dtype=torch.long, device=device)
            losses.append(nn.CrossEntropyLoss()(logits.unsqueeze(0), labels))

        loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, tokenizer, device, args):
    model.eval()
    top_ks = args.top_ks
    hit_at_k = {k: 0 for k in top_ks}
    precision_at_k = {k: [] for k in top_ks}
    recall_at_k = {k: [] for k in top_ks}
    f1_at_k = {k: [] for k in top_ks}
    total_samples = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        for item in batch:
            question = item['question']
            path_texts = item['path_texts']
            path_keys = item['path_keys']
            key_paths_keys = set(item['key_paths_keys'])
            num_paths = len(path_texts)

            if num_paths == 0:
                continue

            q_emb = model.encode_question([question], tokenizer, device, args.max_len)
            path_embs = model.encode_path(path_texts, tokenizer, device, args.max_len)
            similarities = torch.matmul(q_emb, path_embs.T)[0].cpu().numpy()
            sorted_indices = np.argsort(similarities)[::-1]

            for k in top_ks:
                effective_k = min(k, num_paths)
                retrieved_keys = [path_keys[i] for i in sorted_indices[:effective_k]]
                correct = sum(1 for rk in retrieved_keys if rk in key_paths_keys)
                hit = 1 if correct > 0 else 0
                prec = correct / effective_k
                rec = correct / len(key_paths_keys) if key_paths_keys else 0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

                hit_at_k[k] += hit
                precision_at_k[k].append(prec)
                recall_at_k[k].append(rec)
                f1_at_k[k].append(f1)

            total_samples += 1

    return {
        f'Hit@{k}': hit_at_k[k] / total_samples for k in top_ks
    } | {
        f'Precision@{k}': np.mean(precision_at_k[k]) for k in top_ks
    } | {
        f'Recall@{k}': np.mean(recall_at_k[k]) for k in top_ks
    } | {
        f'F1@{k}': np.mean(f1_at_k[k]) for k in top_ks
    }


def split_train_val(samples, val_ratio=0.1, seed=42):
    random.seed(seed)
    indices = list(range(len(samples)))
    random.shuffle(indices)
    val_size = int(len(samples) * val_ratio)
    return [samples[i] for i in indices[val_size:]], [samples[i] for i in indices[:val_size]]


def save_model(model, tokenizer, save_dir, base_model, args, suffix):
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(
        save_dir,
        f"model_{base_model.replace('/', '_')}_bs{args.batch_size}_lr{args.lr}_{suffix}.pt"
    )
    torch.save(model.state_dict(), model_path)
    tokenizer.save_pretrained(save_dir)
    logger.info(f"Model saved to {model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file',      type=str,   required=True)
    parser.add_argument('--test_file',       type=str,   required=True)
    parser.add_argument('--base_model',      type=str,   default='bert-base-uncased')
    parser.add_argument('--output_dir',      type=str,   default='./retrieval_result')
    parser.add_argument('--batch_size',      type=int,   default=32)
    parser.add_argument('--eval_batch_size', type=int,   default=64)
    parser.add_argument('--epochs',          type=int,   default=5)
    parser.add_argument('--lr',              type=float, default=2e-5)
    parser.add_argument('--temperature',     type=float, default=0.05)
    parser.add_argument('--max_len',         type=int,   default=128)
    parser.add_argument('--val_ratio',       type=float, default=0.0)
    parser.add_argument('--seed',            type=int,   default=42)
    parser.add_argument('--projection_dim',  type=int,   default=256)
    parser.add_argument('--top_ks',          type=int,   nargs='+', default=[1, 3, 5])
    parser.add_argument('--neg_num',         type=int,   default=1)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    exp_dir = os.path.join(args.output_dir, args.base_model.replace('/', '_'))
    os.makedirs(exp_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    logger.info("Loading training data...")
    all_samples = []
    with open(args.train_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                all_samples.append(json.loads(line))
    all_samples = [s for s in all_samples if s.get('key_paths')]
    logger.info(f"Samples with key_paths: {len(all_samples)}")

    train_samples, val_samples = split_train_val(all_samples, args.val_ratio, args.seed)
    logger.info(f"Train: {len(train_samples)} | Val: {len(val_samples)}")

    train_dataset = PathRetrievalDataset(
        train_samples, tokenizer, max_len=args.max_len, is_training=True, neg_num=args.neg_num
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_train
    )

    logger.info("Loading test data...")
    test_samples = []
    with open(args.test_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                test_samples.append(json.loads(line))
    test_samples = [s for s in test_samples if s.get('key_paths')]
    test_dataset = PathRetrievalDataset(test_samples, tokenizer, max_len=args.max_len, is_training=False)
    test_loader = DataLoader(
        test_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_eval
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    model = DualEncoder(args.base_model, projection_dim=args.projection_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_hit_at_1 = 0.0
    for epoch in range(1, args.epochs + 1):
        logger.info(f"========== Epoch {epoch} ==========")
        train_loss = train_epoch(model, train_loader, optimizer, tokenizer, device, args)
        logger.info(f"Train loss: {train_loss:.4f}")

        results = evaluate(model, test_loader, tokenizer, device, args)
        for metric, value in results.items():
            logger.info(f"  {metric}: {value:.4f}")

        if results.get('Hit@1', 0) > best_hit_at_1:
            best_hit_at_1 = results['Hit@1']
            save_model(model, tokenizer, exp_dir, args.base_model, args, "best_model")

    logger.info("All done.")


if __name__ == "__main__":
    main()