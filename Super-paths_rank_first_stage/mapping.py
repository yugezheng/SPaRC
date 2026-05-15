# -*- coding: utf-8 -*-
import json
import re
from tqdm import tqdm
from collections import defaultdict
import os
FILE_A = "./eval.jsonl" #your dir
FILE_B = "./finalentities.json"
FILE_C = "output/mapping.json"
FILE_MISSING = './missing_entities1.txt'
def map_entities():
    print("正在加载映射表 B...")
    if not os.path.exists(FILE_B):
        print(f"错误: 找不到文件 {FILE_B}")
        return
    with open(FILE_B, 'r', encoding='utf-8') as f:
        entity_map = json.load(f)
    missing_entity_ids = set()

    def get_mapped_name(eid):
        if not isinstance(eid, str):
            return eid
        
        eid_strip = eid.strip()
        if eid_strip.startswith('m.') or eid_strip.startswith('g.'):
            if eid_strip in entity_map:
                return entity_map[eid_strip]
            else:
                missing_entity_ids.add(eid_strip)
                return eid_strip
        return eid_strip

    print("开始逐行处理 A 文件...")
    with open(FILE_A, 'r', encoding='utf-8') as f_in, \
         open(FILE_C, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            if not line.strip():
                continue
            
            item = json.loads(line)
            
            # --- key_paths ---
            if "key_paths" in item:
                for path in item["key_paths"]:
                    path[0] = get_mapped_name(path[0])
                    if len(path) > 2 and isinstance(path[2], list):
                        path[2] = [get_mapped_name(e) for e in path[2]]

            # ---  top_predictions ---
            if "top_predictions" in item:
                for top_k in item["top_predictions"].values():
                    # 1. path_keys
                    if "path_keys" in top_k:
                        for pk in top_k["path_keys"]:
                            pk[0] = get_mapped_name(pk[0])
                            if len(pk) > 2 and isinstance(pk[2], list):
                                pk[2] = [get_mapped_name(e) for e in pk[2]]
                    
                    # 2. path_texts 
                    if "path_texts" in top_k:
                        new_texts = []
                        for text in top_k["path_texts"]:
                            parts = text.split(" -> ")
                            mapped_parts = []
                            
                            for p in parts:
                                p_strip = p.strip()
                                if "," in p_strip:
                                    sub_ids = [sid.strip() for sid in p_strip.split(",") if sid.strip()]
                                    mapped_subs = [get_mapped_name(sid) for sid in sub_ids]
                                    mapped_parts.append(",".join(mapped_subs))
                                else:
                                    mapped_parts.append(get_mapped_name(p_strip))
                            
                            new_text = " -> ".join(mapped_parts)
                            new_texts.append(new_text)
                        top_k["path_texts"] = new_texts

            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
    with open(FILE_MISSING, 'w', encoding='utf-8') as f_miss:
        for mid in sorted(list(missing_entity_ids)):
            f_miss.write(mid + '\n')

    print("-" * 30)
    print(f"映射完成！")
    print(f"结果文件: {FILE_C}")
    print(f"缺失 ID 统计文件: {FILE_MISSING}")
    print(f"缺失实体总数: {len(missing_entity_ids)}")
    print("-" * 30)

if __name__ == "__main__":
    map_entities()