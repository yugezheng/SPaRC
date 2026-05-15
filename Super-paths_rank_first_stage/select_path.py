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
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import argparse
import logging
from datetime import datetime

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def path_to_text(path: List, topic_ent: Dict[str, str], for_save: bool = False) -> str:
    head_id = path[0]
    relations = path[1]

    head_name = topic_ent.get(head_id, head_id)
    relation_text = ", ".join(relations)
    basic_text = f"{head_name} -> {relation_text}"
    if for_save and len(path) >= 3:
        tail_info = path[2]
        if isinstance(tail_info, list):
            tail_parts = []
            for tid in tail_info:
                tail_name = topic_ent.get(str(tid), str(tid))
                tail_parts.append(f"{tail_name}")
            tail_text = ",".join(tail_parts)
        else:
            tail_name = topic_ent.get(str(tail_info), str(tail_info))
            tail_text = f"{tail_name}"
        return f"{basic_text} -> {tail_text}"
    else:
        return basic_text

def path_to_key(path: List) -> Tuple:
    return (path[0], tuple(path[1]),tuple(path[2]))



def collate_eval(batch):
    return batch
class DualEncoder(nn.Module):
    def __init__(self, base_model_name, projection_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        self.projection = nn.Linear(self.encoder.config.hidden_size, projection_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_vec = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]
        projected = self.projection(cls_vec)
        projected = nn.functional.normalize(projected, p=2, dim=-1)
        return projected

    def encode_question(self, questions, tokenizer, device, max_len=128):
        encoded = tokenizer(questions, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        return self.forward(input_ids, attention_mask)

    def encode_path(self, path_texts, tokenizer, device, max_len=128):
        encoded = tokenizer(path_texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        return self.forward(input_ids, attention_mask)

