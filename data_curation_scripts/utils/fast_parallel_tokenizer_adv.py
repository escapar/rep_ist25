import os
import sys
import subprocess
from concurrent.futures import ProcessPoolExecutor
sys.path.append(os.path.abspath('program/data_curation'))
import tokenizer_runner
TOKENIZER_EXE_PATH = os.path.abspath('tokenizer-master/src/tokenizer')

def process_dir(args):
    (lang, smell, case, input_dir, output_dir, exe_path) = args
    if not os.path.exists(input_dir):
        return f'Skipped {lang} {smell} {case} (Input not found: {input_dir})'
    os.makedirs(output_dir, exist_ok=True)
    tokenizer_level = 'method'
    if smell in ['MultifacetedAbstraction', 'FeatureEnvy']:
        tokenizer_level = 'file'
    print(f'Starting {lang} {smell} {case}...', flush=True)
    try:
        tokenizer_runner._run_tokenizer(input_dir, output_dir, exe_path, lang, tokenizer_level)
        return f'Completed {lang} {smell} {case}'
    except Exception as e:
        return f'Failed {lang} {smell} {case}: {e}'
if __name__ == '__main__':
    if not os.path.exists(TOKENIZER_EXE_PATH):
        print(f'Error: Tokenizer executable not found at {TOKENIZER_EXE_PATH}')
        sys.exit(1)
    tasks = []
    smells = ['ComplexConditional', 'ComplexMethod', 'MultifacetedAbstraction', 'FeatureEnvy']
    cases = ['Positive', 'Negative']
    modes = ['semantic', 'structural', 'hybrid']
    for mode in modes:
        bases = {'CSharp': (os.path.abspath(f'data/attack_v2_cs_{mode}'), os.path.abspath(f'data/tokenizer_attack_v2_cs_{mode}')), 'Java': (os.path.abspath(f'data/attack_v2_{mode}'), os.path.abspath(f'data/tokenizer_attack_v2_{mode}'))}
        for (lang, (in_base, out_base)) in bases.items():
            for smell in smells:
                for case in cases:
                    input_dir = os.path.join(in_base, smell, case)
                    output_dir = os.path.join(out_base, smell, '1d', case)
                    tasks.append((lang, smell, case, input_dir, output_dir, TOKENIZER_EXE_PATH))
    print(f'--- Starting FAST Parallel Tokenization for RQ2 ({len(tasks)} Tasks) ---')
    with ProcessPoolExecutor(max_workers=16) as executor:
        for result in executor.map(process_dir, tasks):
            print(result, flush=True)
    print('--- ALL RQ2 TOKENIZATION TASKS COMPLETED ---')