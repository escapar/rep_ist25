import os
import re
import random
from concurrent.futures import ProcessPoolExecutor
import time
import sys
import json
INPUT_BASE = os.path.abspath('data/csharp_subset_unique')
NUM_CORES = 10
BATCH_SIZE = 50

def transform_for_to_while(code):
    pattern = 'for\\s*\\((.*?);(.*?);(.*?)\\)\\s*\\{'
    replacement = '{ \\1; while(\\2) { '
    return re.sub(pattern, replacement, code)

def semantic_rename(code):
    vars_to_rename = re.findall('\\b(?:int|float|double|string|long|bool|char|var)\\s+([a-zA-Z_][a-zA-Z0-9_]*)\\s*=', code)
    for v in set(vars_to_rename):
        if v not in ['if', 'while', 'for', 'return', 'public', 'class']:
            new_v = f'v_{random.randint(100, 999)}_{v}'
            code = re.sub('\\b' + v + '\\b', new_v, code)
    return code

def process_batch(args):
    (files, mode, batch_idx) = args
    for (f_in, f_out) in files:
        try:
            with open(f_in, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
            if mode == 'semantic' or mode == 'hybrid':
                code = transform_for_to_while(code)
                code = semantic_rename(code)
            if mode == 'structural' or mode == 'hybrid':
                if '{' in code:
                    code = code.replace('{', '{ { int _gate = 0; _gate++; } ', 1)
            with open(f_out, 'w', encoding='utf-8') as f:
                f.write(code)
        except:
            continue

def run_ablation_attack(mode):
    print(f'--- Running C# Ablation Attack: Mode [{mode.upper()}] ---')
    output_dir = os.path.abspath(f'data/attack_v2_cs_{mode}')
    smells = ['ComplexConditional', 'ComplexMethod', 'FeatureEnvy', 'MultifacetedAbstraction']
    with open('../config/dataset_splits.json', 'r') as f:
        subset = json.load(f)['CSharp']
    for smell in smells:
        for case in ['Positive', 'Negative']:
            src = os.path.join(INPUT_BASE, smell, case)
            dst = os.path.join(output_dir, smell, case)
            if not os.path.exists(src):
                print(f'   Skipping {smell}/{case} (Source not found at {src})')
                continue
            os.makedirs(dst, exist_ok=True)
            target_list = subset[smell][case]
            print(f'   Streaming {len(target_list)} targeted files for {smell}/{case}...')
            sys.stdout.flush()
            curr = []
            count = 0
            with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
                for fname in target_list:
                    f_path = os.path.join(src, fname)
                    f_out = os.path.join(dst, fname)
                    if os.path.exists(f_out):
                        continue
                    curr.append((f_path, f_out))
                    if len(curr) >= BATCH_SIZE:
                        executor.submit(process_batch, (curr, mode, count))
                        count += len(curr)
                        if count % 5000 == 0:
                            print(f'      - Submitted {count} files...')
                            sys.stdout.flush()
                        curr = []
                if curr:
                    executor.submit(process_batch, (curr, mode, count))
            print(f'      - Done with {smell}/{case}')
            sys.stdout.flush()
if __name__ == '__main__':
    if len(sys.argv) > 1:
        run_ablation_attack(sys.argv[1])
    else:
        for m in ['semantic', 'structural', 'hybrid']:
            run_ablation_attack(m)