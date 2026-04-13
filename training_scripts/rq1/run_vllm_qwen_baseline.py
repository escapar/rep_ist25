import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))

import os
import sys
import json
import random
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, accuracy_score, precision_score, recall_score
MODEL_PATH = os.path.abspath('qwen_local_final')
SUBSET_JSON = '../config/dataset_splits.json'
LIMIT = 100
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def run_qwen_inference(lang, smell):
    print(f'\n--- [vLLM] Qwen Baseline Inference: {lang} {smell} ---')
    print(f'Loading vLLM model from {MODEL_PATH}...')
    llm = LLM(model=MODEL_PATH, tensor_parallel_size=1, trust_remote_code=True, gpu_memory_utilization=0.6, max_model_len=4096)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=200)
    with open(SUBSET_JSON, 'r') as f:
        subset = json.load(f)[lang][smell]
    input_base = f'data/{lang.lower()}_subset_unique/{smell}'
    eval_samples = []
    for cat in ['Positive', 'Negative']:
        label = 1 if cat == 'Positive' else 0
        files = subset[cat]
        if len(files) > LIMIT:
            files = random.sample(files, LIMIT)
        for f in files:
            eval_samples.append((os.path.join(input_base, cat, f), label))
    random.shuffle(eval_samples)
    prompts = []
    labels = []
    SMELL_DEFS = {'ComplexMethod': 'A method that has high cyclomatic complexity, excessive lines of code, and too many decision points (if/else, loops).', 'ComplexConditional': 'A conditional statement (if/while) that contains a deeply nested or excessively long logical expression with multiple AND/OR operators.', 'FeatureEnvy': 'A method that accesses the data or methods of another class more than its own, suggesting it should be moved.', 'MultifacetedAbstraction': 'A class that has more than one responsibility, violating the Single Responsibility Principle, often indicated by disjoint sets of methods and fields (low cohesion).'}
    smell_def = SMELL_DEFS.get(smell, '')
    print(f'Preparing {len(eval_samples)} prompts...')
    for (file_path, label) in eval_samples:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
        except:
            continue
        prompt = f"You are an expert software architect code reviewer analyzing {lang} source code for design flaws.\nDefinition of '{smell}': {smell_def}\nTask: Does this {lang} code suffer from the '{smell}' code smell?\nCode:\n{code[:2000]}\nLet's think step by step. First, analyze the code structure and logic based on the provided definition. Then, determine if the characteristics match the '{smell}'. Finally, conclude your analysis with a final answer in the format 'Final Answer: Yes' or 'Final Answer: No'.\nAnalysis:"
        prompts.append(prompt)
        labels.append(label)
    print(f'Generating answers using vLLM...')
    outputs = llm.generate(prompts, sampling_params)
    preds = []
    probs = []
    for output in outputs:
        ans = output.outputs[0].text.strip().lower()
        if 'final answer: yes' in ans:
            pred = 1
        elif 'final answer: no' in ans:
            pred = 0
        else:
            pred = 1 if 'yes' in ans else 0
        preds.append(pred)
        probs.append(1.0 if pred == 1 else 0.0)
    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    auc = roc_auc_score(labels, probs)
    print(f'RESULT: {lang} {smell} Qwen -> F1: {f1:.4f}, AUC: {auc:.4f}, MCC: {mcc:.4f}, ACC: {acc:.4f}, PREC: {prec:.4f}, REC: {rec:.4f}')
    res_file = f'../results/qwen_rq1_{smell}.csv'
    with open(res_file, 'a') as f:
        f.write(f'{lang},{smell},{f1},{auc},{mcc},{acc},{prec},{rec}\n')
if __name__ == '__main__':
    lang = sys.argv[1]
    smell = sys.argv[2]
    run_qwen_inference(lang, smell)