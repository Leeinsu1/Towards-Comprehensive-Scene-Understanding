import os
import json
import re
from typing import List, Dict, Optional
import statistics

def extract_characters_regex(s: str) -> str:
    if s is None:
        print("None included in model_answer")
        return ""
    s = s.strip() if isinstance(s, str) else s[0]
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is"
        "The correct option is",
        "Best answer:"
        "Best option:",
        "Answer:",
        "Option:",
        "The correct answer",
        "The correct option",
    ]
    for prefix in answer_prefixes:
        s = s.replace(prefix, "")

    match = re.findall(r'[ABCD]\)', s)
    if match:
        return match[-1]
    match = re.search(r'[ABCD][:.]', s)
    if match:
        return match[0]
    match = re.search(r'[ABCD]', s)
    if match:
        return match[0]
    return s.lower()

CATEGORIES = [
        "action_ego", "action_exo", "object_ego", "object_exo", "numerical_ego",
        "numerical_exo", "spatial_ego", "spatial_exo"
    ]
def eval_results(gt_dir: str, result_dir: str):

    category_accuracy = {category: {"correct": 0, "total": 0} for category in CATEGORIES}

    for category in CATEGORIES:
        
        gt_file_path = os.path.join(gt_dir, f"{category.split('_')[0]}_{category.split('_')[1]}.jsonl")
        result_file_path = os.path.join(result_dir, f"{category.split('_')[0]}_{category.split('_')[1]}.jsonl")

        if not os.path.exists(gt_file_path) or not os.path.exists(result_file_path):
            continue

        with open(gt_file_path, 'r') as gt_f, open(result_file_path, 'r') as res_f:
            gt_data = [json.loads(line.strip()) for line in gt_f]
            res_data = [json.loads(line.strip()) for line in res_f]

            gt_dict = {item["id"]: item for item in gt_data}
            res_dict = {item["id"]: item for item in res_data}

            for qid in gt_dict.keys():
                if qid not in res_dict.keys():
                    continue
                gt_answer = gt_dict[qid].get("answer", "").strip().lower()
                options = res_dict[qid]["options"]
                gt_answer = next((chr(65 + i) for i, option in enumerate(options) if gt_answer == option.strip().lower()), "No match found")
                model_answer = extract_characters_regex(res_dict[qid].get("model_answer", ""))


                if model_answer and gt_answer.strip() == model_answer.strip()[0]:
                    category_accuracy[category]["correct"] += 1
                category_accuracy[category]["total"] += 1


    print(f"Evaulation Folder: {result_dir}")
    print("=====================================")
    print("Per category Accuracy:")
    print("=====================================")
    accuracy_per_category = {}
    for category, values in category_accuracy.items():
        correct, total = values["correct"], values["total"]
        accuracy = (100 * correct / total) if total > 0 else 0
        accuracy_per_category[category] = accuracy
        print(f"{category}: {accuracy:.2f}% Eval number: {total}")

    total_correct = sum(values["correct"] for values in category_accuracy.values())
    total_answered = sum(values["total"] for values in category_accuracy.values())
    overall_accuracy = (100 * total_correct / total_answered) if total_answered > 0 else 0

    print("=====================================")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    print("=====================================")

    return accuracy_per_category, overall_accuracy


eval_results("/path/to/groundtruth/annotations",
                    "/path/to/results/")