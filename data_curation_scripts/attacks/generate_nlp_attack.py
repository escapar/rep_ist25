import os
import sys
import json
import random
import time
from multiprocessing import Pool, cpu_count
import nltk
try:
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('universal_tagset', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
except:
    pass
from textattack.transformations import WordSwapEmbedding, WordSwapHomoglyphSwap, WordSwapNeighboringCharacterSwap, WordSwapRandomCharacterDeletion, WordSwapQWERTY
from textattack.shared import AttackedText
try:
    transformations = [WordSwapEmbedding(), WordSwapHomoglyphSwap(), WordSwapNeighboringCharacterSwap(), WordSwapRandomCharacterDeletion(), WordSwapQWERTY()]
except Exception as e:
    print(f'Error initializing transformations: {e}')
    transformations = []

def generate_adversarial_text(text):
    if not transformations:
        return text
    try:
        attacked_text = AttackedText(text)
        transformation = random.choice(transformations)
        transformed_texts = transformation(attacked_text)
        if transformed_texts:
            return transformed_texts[0].text
    except Exception as e:
        pass
    return text

def process_file(args):
    (input_file, output_file) = args
    if os.path.exists(output_file):
        return 'Skip'
    try:
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        modified_lines = []
        for line in lines:
            if len(line.strip()) > 5:
                modified_lines.append(generate_adversarial_text(line) + ('' if line.endswith('\n') else '\n'))
            else:
                modified_lines.append(line)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(modified_lines)
        return 'Success'
    except Exception as e:
        return f'Error: {e}'
if __name__ == '__main__':
    smoke_test = os.environ.get('SMOKE_TEST') == '1'
    with open('../config/dataset_splits.json', 'r') as f:
        subset = json.load(f)
    tasks = []
    langs = ['CSharp', 'Java']
    smells = ['ComplexMethod', 'ComplexConditional', 'FeatureEnvy', 'MultifacetedAbstraction']
    for lang in langs:
        for smell in smells:
            for cat in ['Positive', 'Negative']:
                files = subset[lang][smell][cat]
                eval_files = files[int(0.7 * len(files)):]
                if smoke_test:
                    eval_files = eval_files[:2]
                input_dir = f'data/{lang.lower()}_subset_unique/{smell}/{cat}'
                output_dir = f"data/attack_v2_{('cs_' if lang == 'CSharp' else '')}nlp/{smell}/{cat}"
                for f in eval_files:
                    tasks.append((os.path.join(input_dir, f), os.path.join(output_dir, f)))
    print(f'Total files to process for NLP attack: {len(tasks)}')
    start_time = time.time()
    success_count = 0
    with Pool(16) as pool:
        for result in pool.imap_unordered(process_file, tasks):
            if result == 'Success':
                success_count += 1
            elif result != 'Skip':
                pass
    elapsed = time.time() - start_time
    print(f'Done generating NLP attacks. Processed {success_count} files in {elapsed:.2f}s')