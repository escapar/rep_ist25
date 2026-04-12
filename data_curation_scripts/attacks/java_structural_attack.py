import os
import subprocess
import re
import random
from concurrent.futures import ProcessPoolExecutor
import shutil
import time
JAVAC_PATH = 'javac'
INPUT_BASE = '../data/all_java_repos'
OUTPUT_BASE = '../data/attack_v2_structural'
NUM_CORES = 8

def structural_attack_logic(code):
    lines = code.split('\n')
    new_lines = []
    for line in lines:
        new_lines.append(line)
        if random.random() < 0.2:
            new_lines.append(f'// Structural perturbation ID: {random.randint(1000, 9999)}')
            new_lines.append('\n')
    code = '\n'.join(new_lines)
    code = re.sub('if\\s*\\((.*?)\\)\\s*([^{};\\s].*?);', 'if (\\1) { \\2; }', code)
    if '{' in code:
        injection = f'\n        {{ int _tmp_val = {random.randint(1, 100)}; _tmp_val *= 1; }}\n'
        code = code.replace('{', '{' + injection, 1)
    return code

def process_single_file(args):
    (f_in, f_out) = args
    try:
        with open(f_in, 'r', encoding='utf-8', errors='ignore') as r:
            original_code = r.read()
        attacked_code = structural_attack_logic(original_code)
        if 'class ' not in attacked_code:
            wrapped_code = f'public class TempWrapper {{ \n {attacked_code} \n }}'
        else:
            wrapped_code = attacked_code
        with open(f_out, 'w', encoding='utf-8') as w:
            w.write(wrapped_code)
        res = subprocess.run([JAVAC_PATH, f_out], capture_output=True, timeout=10)
        if res.returncode == 0:
            class_file = f_out.replace('.java', '.class')
            if 'TempWrapper' in wrapped_code:
                class_file = os.path.join(os.path.dirname(f_out), 'TempWrapper.class')
            if os.path.exists(class_file):
                os.remove(class_file)
            return True
        else:
            if os.path.exists(f_out):
                os.remove(f_out)
            return False
    except Exception:
        return False

def run_attack_for_category(smell, case):
    src_dir = os.path.join(INPUT_BASE, smell, case)
    dst_dir = os.path.join(OUTPUT_BASE, smell, case)
    os.makedirs(dst_dir, exist_ok=True)
    if not os.path.exists(src_dir):
        return
    files = [f for f in os.listdir(src_dir) if f.endswith('.code')]
    print(f'   [Task] Attacking {smell}/{case}: {len(files)} files...')
    tasks = []
    for f in files:
        f_in = os.path.join(src_dir, f)
        f_out = os.path.join(dst_dir, f.replace('.code', '.java'))
        tasks.append((f_in, f_out))
    success_count = 0
    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        results = list(executor.map(process_single_file, tasks))
        success_count = sum(results)
    print(f'   [Done] {smell}/{case}: {success_count}/{len(files)} files passed JAVAC validation.')

def main():
    print(f'Starting High-Performance Structural Attack & JAVAC Validation...')
    start_time = time.time()
    smells = ['ComplexConditional', 'ComplexMethod', 'FeatureEnvy', 'MultifacetedAbstraction']
    for smell in smells:
        print(f'Processing Smell: {smell}')
        for case in ['Positive', 'Negative']:
            run_attack_for_category(smell, case)
    print(f'Total time elapsed: {time.time() - start_time:.2f}s')
if __name__ == '__main__':
    main()