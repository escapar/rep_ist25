import os
import sys
import json
import random
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, accuracy_score
MODEL_PATH = os.path.abspath('qwen_local_final')
SUBSET_JSON = '../config/dataset_splits.json'
LIMIT = 100
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def run_rq2_qwen_inference(lang, smell, mode, limit=LIMIT):
    res_file = f'../data_csvs/rq2_qwen_{smell}.csv'
    if os.path.exists(res_file):
        with open(res_file, 'r') as f:
            for line in f:
                if f'{lang},{smell},{mode}' in line:
                    print(f'Skipping {lang} {smell} {mode} - result already exists in {res_file}')
                    return
    print(f'\n--- [vLLM] RQ2 Qwen Inference: {lang} {smell} [{mode}] ---')
    print(f'Loading vLLM model from {MODEL_PATH}...')
    llm = LLM(model=MODEL_PATH, tensor_parallel_size=1, trust_remote_code=True, gpu_memory_utilization=0.6, max_model_len=4096)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=200)
    with open(SUBSET_JSON, 'r') as f:
        subset = json.load(f)[lang][smell]
    if lang == 'CSharp':
        adv_base = f'data/attack_v2_cs_{mode}/{smell}'
    else:
        adv_base = f'data/attack_v2_{mode}/{smell}'
    eval_samples = []
    for cat in ['Positive', 'Negative']:
        label = 1 if cat == 'Positive' else 0
        files = subset[cat]
        eval_files = files[int(0.7 * len(files)):]
        if len(eval_files) > limit:
            eval_files = random.sample(eval_files, limit)
        for f in eval_files:
            file_path = os.path.join(adv_base, cat, f)
            if os.path.exists(file_path):
                eval_samples.append((file_path, label))
    random.shuffle(eval_samples)
    prompts = []
    labels = []
    SMELL_DEFS = {'ComplexMethod': 'A method that has high cyclomatic complexity, excessive lines of code, and too many decision points (if/else, loops).', 'ComplexConditional': 'A conditional statement (if/while) that contains a deeply nested or excessively long logical expression with multiple AND/OR operators.', 'FeatureEnvy': 'A method that accesses the data or methods of another class more than its own, suggesting it should be moved.', 'MultifacetedAbstraction': 'A class that has more than one responsibility, violating the Single Responsibility Principle, often indicated by disjoint sets of methods and fields (low cohesion).'}
    smell_def = SMELL_DEFS.get(smell, '')
    print(f'Preparing {len(eval_samples)} adversarial samples...')
    for (file_path, label) in eval_samples:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
        except:
            continue
        prompt = f"Does this {lang} code have a '{smell}' smell? Answer Yes/No.\nCode: {code[:2000]}\nAnswer:"
        prompts.append(prompt)
        labels.append(label)
    if not labels:
        print('No samples found.')
        return
    print(f'Generating answers using vLLM...')
    outputs = llm.generate(prompts, sampling_params)
    preds = []
    probs = []
    for output in outputs:
        ans = output.outputs[0].text.strip().lower()
        pred = 1 if 'yes' in ans else 0
        preds.append(pred)
        probs.append(1.0 if pred == 1 else 0.0)
    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.0
    print(f'RESULT: {lang} {smell} Qwen [{mode}] -> F1: {f1:.4f}, AUC: {auc:.4f}, MCC: {mcc:.4f}, ACC: {acc:.4f}')
    res_file = f'../data_csvs/rq2_qwen_{smell}.csv'
    with open(res_file, 'a') as f:
        f.write(f'{lang},{smell},{mode},{f1},{auc},{mcc},{acc}\n')
if __name__ == '__main__':
    lang = sys.argv[1]
    smell = sys.argv[2]
    mode = sys.argv[3]
    limit = LIMIT
    if os.environ.get('SMOKE_TEST') == '1':
        limit = 8
    run_rq2_qwen_inference(lang, smell, mode, limit=limit)