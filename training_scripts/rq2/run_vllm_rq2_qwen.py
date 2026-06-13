import sys, os
import json
import random
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, accuracy_score

MODEL_PATH = 'models/qwen_local_final'
SUBSET_JSON = 'config/dataset_splits.json'
LIMIT = 100
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def run_rq2_qwen_inference(lang, smell, mode, limit=LIMIT):
    print(f'\n--- [vLLM] RQ2 Qwen: {lang} {smell} [{mode}] ---')
    print(f'Loading vLLM model from {MODEL_PATH}...')
    llm = LLM(model=MODEL_PATH, tensor_parallel_size=1, trust_remote_code=True, gpu_memory_utilization=0.6, max_model_len=4096, enforce_eager=True)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=200)
    if lang == 'CSharp':
        adv_base = f'data/synonym_attack_raw/CSharp/{mode}/{smell}'
    else:
        adv_base = f'data/synonym_attack_raw/Java/{mode}/{smell}'
    eval_samples = []
    for cat in ['Positive', 'Negative']:
        label = 1 if cat == 'Positive' else 0
        cat_dir = os.path.join(adv_base, cat)
        if not os.path.exists(cat_dir):
            print(f'Warning: {cat_dir} not found')
            continue
        files = os.listdir(cat_dir)
        eval_files = files
        if len(eval_files) > limit:
            eval_files = random.sample(eval_files, limit)
        for f in eval_files:
            adv_p = os.path.join(cat_dir, f)
            if os.path.exists(adv_p):
                eval_samples.append((adv_p, label))
    random.shuffle(eval_samples)
    SMELL_DEFS = {'ComplexMethod': 'A method that has high cyclomatic complexity, excessive lines of code, and too many decision points (if/else, loops).', 'ComplexConditional': 'A conditional statement (if/while) that contains a deeply nested or excessively long logical expression with multiple AND/OR operators.', 'FeatureEnvy': 'A method that accesses the data or methods of another class more than its own, suggesting it should be moved.', 'MultifacetedAbstraction': 'A class that has more than one responsibility, violating the Single Responsibility Principle, often indicated by disjoint sets of methods and fields (low cohesion).'}
    smell_def = SMELL_DEFS.get(smell, '')
    prompts = []
    labels = []
    print(f'Preparing {len(eval_samples)} adversarial evaluation samples...')
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
    print(f'RESULT: {lang} {smell} RQ2 Qwen [{mode}] -> F1: {f1:.4f}, AUC: {auc:.4f}, MCC: {mcc:.4f}, ACC: {acc:.4f}')
    res_file = os.path.join(os.path.abspath('../results'), f'rq2_qwen_{smell}.csv')
    os.makedirs(os.path.dirname(res_file), exist_ok=True)
    with open(res_file, 'a') as f:
        f.write(f'{lang},{smell},{mode},{f1},{auc},{mcc},{acc}\n')

if __name__ == '__main__':
    lang = sys.argv[1]
    smell = sys.argv[2]
    mode = sys.argv[3]
    limit = LIMIT
    if len(sys.argv) > 4:
        limit = int(sys.argv[4])
    run_rq2_qwen_inference(lang, smell, mode, limit=limit)
