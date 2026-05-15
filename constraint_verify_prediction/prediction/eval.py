
import json
import re
import string
from copy import deepcopy
from datetime import datetime
import os

def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\b()\b", " ", s)
    s = " ".join(s.split())
    return s

def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1

def remove_duplicates(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result

def get_pred(prediction, split=None):
    if split is not None:
        return prediction.split(split)
    
    res = [p for p in prediction.split("\n") if 'ans:' in p and 'none' not in p.lower()]
    
    if len(res) >= 1:
        res = [p for p in res if "ans: not available" not in p.lower() and 
               "ans: no information available" not in p.lower()]
    
    return remove_duplicates(res)

def all_answers_unavailable(res):
    if not res:
        return True
    
    matches = re.findall(r"ans:\s*(.*)", res, re.IGNORECASE)
    if not matches:
        return True
    
    for answer in matches:
        norm = answer.strip().lower()
        if not any(x in norm for x in ["not available", "no information available", "none"]):
            return False
    return True

def eval_recall(prediction, answer, double_check):
    prediction = deepcopy(prediction)
    prediction = sorted(prediction, key=len, reverse=True)
    matched = 0.
    
    for a in answer:
        for pred in prediction:
            if match(pred, a):
                matched += 1
                prediction.remove(pred)
                break
            elif double_check:
                if match(a, pred.split('ans:')[-1].strip()) or match(a, pred):
                    matched += 1
                    prediction.remove(pred)
                    break
    
    return matched / len(answer) if len(answer) > 0 else 0, matched, len(answer)

def eval_precision(prediction, answer, double_check):
    prediction = deepcopy(prediction)
    prediction = sorted(prediction, key=len, reverse=True)
    num_pred = len(prediction)
    
    if num_pred == 0:
        return 0, 0, 0
    
    matched = 0.
    for a in answer:
        for pred in prediction:
            if match(pred, a):
                matched += 1
                prediction.remove(pred)
                break
            elif double_check:
                if match(a, pred.split('ans:')[-1].strip()) or match(a, pred):
                    matched += 1
                    prediction.remove(pred)
                    break
    
    return matched / num_pred, matched, num_pred

def eval_f1(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)

def eval_hit(prediction_list, answer, double_check):
    if not isinstance(prediction_list, list) or len(prediction_list) == 0:
        return 0
    
    first_pred = prediction_list[0]
    for a in answer:
        if match(first_pred, a):
            return 1
        elif double_check:

            clean_p = first_pred.split('ans:')[-1].strip()
            if match(a, clean_p) or match(a, first_pred):
                return 1
    return 0

def eval_hit_any(prediction_list, answer, double_check):
    if not isinstance(prediction_list, list) or len(prediction_list) == 0:
        return 0
    
    for a in answer:
        for each_pred in prediction_list:
            if match(each_pred, a):
                return 1
            elif double_check:
                clean_p = each_pred.split('ans:')[-1].strip()
                if match(a, clean_p) or match(a, each_pred):
                    return 1
    return 0

def evaluate_file(predict_file, output_dir):
    detailed_results = []
    
    hit_list = [] 
    hit_at_1_list = [] 
    f1_list = []
    precision_list = []
    recall_list = []
    
    total_pred = 0
    total_answer = 0
    total_match = 0
    
    print(f"\n正在评估文件: {predict_file}")
    
    with open(predict_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
            except:
                print(f"警告: 第 {line_num} 行JSON解析失败")
                continue
            
            id = data['id']
            answer = sorted(remove_duplicates(data['ground_truth']), key=len, reverse=True)
            question = data['question']
            
            if isinstance(data['prediction'], list):
                pred_list = data['prediction']
                prediction_str = data.get('raw_prediction', '\n'.join(pred_list))
            else:
                prediction_str = data['prediction']
                pred_list = get_pred(prediction_str)
            if 'when' in question.lower() or 'what year' in question.lower():
                for idx in range(len(answer)):
                    if '-' in answer[idx] and answer[idx].split('-')[0].isdigit():
                        answer[idx] = answer[idx].split('-')[0]
            double_check = any([keyword in question.lower() for keyword in 
                               ['when', 'what year', 'which year', 'where', 'sport', 
                                "what countr", "language", 'nba finals', 'world series']])
            no_ans_flag = len(pred_list) == 0 or all_answers_unavailable(prediction_str)
            precision_score, matched_1, num_pred = eval_precision(pred_list, answer, double_check)
            recall_score, matched_2, num_answer = eval_recall(pred_list, answer, double_check)
            f1_score = eval_f1(precision_score, recall_score)
            hit_at_1 = eval_hit(pred_list, answer, double_check)
            hit = eval_hit_any(pred_list, answer, double_check)
            total_pred += num_pred
            total_answer += num_answer
            total_match += matched_1
            
            hit_list.append(hit)
            hit_at_1_list.append(hit_at_1)
            f1_list.append(f1_score)
            precision_list.append(precision_score)
            recall_list.append(recall_score)
            detailed_results.append({
                'id': id,
                'question': question,
                'prediction': pred_list,
                'ground_truth': answer,
                'hit': hit,
                'hit@1': hit_at_1,
                'precision': precision_score,
                'recall': recall_score,
                'f1': f1_score,
                'no_answer': no_ans_flag
            })
    if len(hit_list) == 0:
        print("警告: 没有有效的评估样本")
        return None
    
    results = {
        'total_samples': len(hit_list),
        'hit': sum(hit_list) * 100 / len(hit_list),
        'hit@1': sum(hit_at_1_list) * 100 / len(hit_at_1_list),
        'macro_precision': sum(precision_list) * 100 / len(precision_list),
        'macro_recall': sum(recall_list) * 100 / len(recall_list),
        'macro_f1': sum(f1_list) * 100 / len(f1_list),
        'micro_precision': (total_match / total_pred * 100) if total_pred > 0 else 0,
        'micro_recall': (total_match / total_answer * 100) if total_answer > 0 else 0,
    }
    
    # 计算Micro F1
    if results['micro_precision'] + results['micro_recall'] > 0:
        results['micro_f1'] = (2 * results['micro_precision'] * results['micro_recall'] / 
                               (results['micro_precision'] + results['micro_recall']))
    else:
        results['micro_f1'] = 0
    
    return results, detailed_results


def save_results(results, detailed_results, output_summary, output_detailed):
    os.makedirs(os.path.dirname(output_summary), exist_ok=True)
    
    # 保存汇总结果
    with open(output_summary, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("评估结果汇总\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"总样本数: {results['total_samples']}\n\n")
        
        f.write("主要指标:\n")
        f.write(f"  Hit (任意答案正确):     {results['hit']:.2f}%\n")
        f.write(f"  Hit@1 (首个答案正确):   {results['hit@1']:.2f}%\n\n")
        
        f.write("Macro 指标 (每个样本平均):\n")
        f.write(f"  Precision:              {results['macro_precision']:.2f}%\n")
        f.write(f"  Recall:                 {results['macro_recall']:.2f}%\n")
        f.write(f"  F1:                     {results['macro_f1']:.2f}%\n\n")
        
        f.write("Micro 指标 (所有预测答案整体):\n")
        f.write(f"  Precision:              {results['micro_precision']:.2f}%\n")
        f.write(f"  Recall:                 {results['micro_recall']:.2f}%\n")
        f.write(f"  F1:                     {results['micro_f1']:.2f}%\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    with open(output_detailed, 'w', encoding='utf-8') as f:
        for item in detailed_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n结果已保存:")
    print(f"  汇总结果: {output_summary}")
    print(f"  详细结果: {output_detailed}")


def main(predict_file):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.dirname(predict_file)
    output_dir = os.path.join(base_dir, f"eval_{ts}")
    output_summary = os.path.join(output_dir, "eval_summary.txt")
    output_detailed = os.path.join(output_dir, "eval_detailed.jsonl")
    results, detailed_results = evaluate_file(predict_file, output_dir)
    
    if results is None:
        print("评估失败!")
        return
    save_results(results, detailed_results, output_summary, output_detailed)
    print("\n" + "=" * 80)
    print("评估完成!")
    print("=" * 80)
    print(f"\n总样本数: {results['total_samples']}")
    print(f"\nHit (任意答案正确):     {results['hit']:.2f}%")
    print(f"Hit@1 (首个答案正确):   {results['hit@1']:.2f}%")
    print(f"\nMacro Precision:        {results['macro_precision']:.2f}%")
    print(f"Macro Recall:           {results['macro_recall']:.2f}%")
    print(f"Macro F1:               {results['macro_f1']:.2f}%")
    print(f"\nMicro Precision:        {results['micro_precision']:.2f}%")
    print(f"Micro Recall:           {results['micro_recall']:.2f}%")
    print(f"Micro F1:               {results['micro_f1']:.2f}%")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    predict_file = "./cwq/results/predictions.jsonl"   # your prediction.jsonl dir
    main(predict_file)