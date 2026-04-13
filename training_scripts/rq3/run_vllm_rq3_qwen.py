import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))

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

def run_rq3_qwen_inference(lang, smell, mode, ratio, limit=LIMIT):
    print(f'\n--- [vLLM] RQ3 Qwen Defense (ICL): {lang} {smell} [{mode}], Ratio: {ratio} ---')
    print(f'Loading vLLM model from {MODEL_PATH}...')
    llm = LLM(model=MODEL_PATH, tensor_parallel_size=1, trust_remote_code=True, gpu_memory_utilization=0.45, max_model_len=4096, enforce_eager=True)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=200)
    with open(SUBSET_JSON, 'r') as f:
        subset = json.load(f)[lang][smell]
    clean_base = f'data/{lang.lower()}_subset_unique/{smell}'
    if lang == 'CSharp':
        adv_base = f'data/attack_v2_cs_{mode}/{smell}'
    else:
        adv_base = f'data/attack_v2_{mode}/{smell}'
    eval_samples = []
    icl_pool = {'clean_pos': [], 'clean_neg': [], 'adv_pos': [], 'adv_neg': []}
    for cat in ['Positive', 'Negative']:
        label = 1 if cat == 'Positive' else 0
        files = subset[cat]
        train_files = files[:int(0.7 * len(files))]
        eval_files = files[int(0.7 * len(files)):]
        for f in train_files:
            clean_p = os.path.join(clean_base, cat, f)
            adv_p = os.path.join(adv_base, cat, f)
            if not os.path.exists(adv_p):
                for ext in ['.code', '.java', '.cs']:
                    alt_p = os.path.splitext(adv_p)[0] + ext
                    if os.path.exists(alt_p):
                        adv_p = alt_p
                        break
            if os.path.exists(clean_p):
                if label == 1:
                    icl_pool['clean_pos'].append(clean_p)
                else:
                    icl_pool['clean_neg'].append(clean_p)
            if os.path.exists(adv_p):
                if label == 1:
                    icl_pool['adv_pos'].append(adv_p)
                else:
                    icl_pool['adv_neg'].append(adv_p)
        if len(eval_files) > limit:
            eval_files = random.sample(eval_files, limit)
        for f in eval_files:
            clean_p = os.path.join(clean_base, cat, f)
            if os.path.exists(clean_p):
                eval_samples.append((clean_p, label))
    random.shuffle(eval_samples)
    SMELL_DEFS = {'ComplexMethod': 'A method that has high cyclomatic complexity, excessive lines of code, and too many decision points (if/else, loops).', 'ComplexConditional': 'A conditional statement (if/while) that contains a deeply nested or excessively long logical expression with multiple AND/OR operators.', 'FeatureEnvy': 'A method that accesses the data or methods of another class more than its own, suggesting it should be moved.', 'MultifacetedAbstraction': 'A class that has more than one responsibility, violating the Single Responsibility Principle, often indicated by disjoint sets of methods and fields (low cohesion).'}
    smell_def = SMELL_DEFS.get(smell, '')
    prompts = []
    labels = []
    print(f'Preparing {len(eval_samples)} clean evaluation samples with ICL defense...')
    for (file_path, label) in eval_samples:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
        except:
            continue
        context_prompt = f"You are an expert software architect code reviewer analyzing {lang} source code for design flaws.\nDefinition of '{smell}': {smell_def}\n\n"
        for (ex_label, cat_clean, cat_adv) in [(1, 'clean_pos', 'adv_pos'), (0, 'clean_neg', 'adv_neg')]:
            use_adv = random.random() < float(ratio)
            pool = icl_pool[cat_adv] if use_adv and icl_pool[cat_adv] else icl_pool[cat_clean]
            if pool:
                ex_file = random.choice(pool)
                try:
                    with open(ex_file, 'r', encoding='utf-8', errors='ignore') as exf:
                        ex_code = exf.read()
                    ans_text = 'Yes' if ex_label == 1 else 'No'
                    context_prompt += f"Example Task: Does this {lang} code suffer from the '{smell}' code smell?\nCode:\n{ex_code[:500]}\nFinal Answer: {ans_text}\n\n"
                except:
                    pass
        prompt = context_prompt + f"Task: Does this {lang} code suffer from the '{smell}' code smell?\nCode:\n{code[:2000]}\nLet's think step by step. First, analyze the code structure and logic based on the provided definition. Then, determine if the characteristics match the '{smell}'. Finally, conclude your analysis with a final answer in the format 'Final Answer: Yes' or 'Final Answer: No'.\nAnalysis:"
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
    mcc = matthews_corrcoef(labels, preds)
    auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.0
    print(f'RESULT: {lang} {smell} RQ3 Qwen [{mode}, ratio={ratio}] -> F1: {f1:.4f}, AUC: {auc:.4f}, MCC: {mcc:.4f}, ACC: {acc:.4f}')
    res_file = f'../results/rq3_qwen_{smell}.csv'
    with open(res_file, 'a') as f:
        f.write(f'{lang},{smell},{mode},{ratio},{f1},{auc},{mcc},{acc}\n')
if __name__ == '__main__':
    lang = sys.argv[1]
    smell = sys.argv[2]
    mode = sys.argv[3]
    ratio = float(sys.argv[4])
    limit = LIMIT
    if os.environ.get('SMOKE_TEST') == '1':
        limit = 8
    run_rq3_qwen_inference(lang, smell, mode, ratio, limit=limit)