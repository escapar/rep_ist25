import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../../'))
CS_TOKENIZED_DATA = os.path.join(PROJECT_ROOT, 'data/tokenizer_cs_unique')
JAVA_TOKENIZED_DATA = os.path.join(PROJECT_ROOT, 'data/tokenizer_java_unique')
MODEL_RESULTS_DIR = os.path.join(PROJECT_ROOT, '../data_csvs')

def get_smell_path(language, smell, dimension='1d'):
    base = CS_TOKENIZED_DATA if language.lower() in ['csharp', 'cs', 'c#'] else JAVA_TOKENIZED_DATA
    return os.path.join(base, smell, dimension)

def verify_paths():
    paths = [CS_TOKENIZED_DATA, JAVA_TOKENIZED_DATA]
    for p in paths:
        if not os.path.exists(p):
            print(f'WARNING: Path does not exist yet: {p}')
        else:
            print(f'VERIFIED: {p}')
if __name__ == '__main__':
    verify_paths()