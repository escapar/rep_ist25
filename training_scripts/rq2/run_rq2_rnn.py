import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))

import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, accuracy_score, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.abspath('program/dl_models'))
import inputs
import path_config
DIM = '1d'
OUT_FOLDER = os.path.abspath('../results')

def evaluate_rnn_rq2(lang, smell, mode):
    res_file = os.path.join(OUT_FOLDER, f'rq2_rnn_{smell}.csv')
    if os.path.exists(res_file):
        with open(res_file, 'r') as f:
            for line in f:
                if f'{lang},{smell},{mode}' in line:
                    print(f'Skipping RNN {lang} {smell} {mode} - result exists.')
                    return
    print(f'\n--- RQ2 RNN Evaluation: {lang} {smell} [{mode}] ---')
    model_name = 'weights_best.rnn.' + smell + ('_JavaBEST' if lang == 'Java' else 'BEST') + '.keras'
    model_path = os.path.join(OUT_FOLDER, model_name)
    if not os.path.exists(model_path):
        print(f'ERROR: Model not found at {model_path}')
        return
    print(f'Loading model: {model_name}')
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f'Failed to load model: {e}')
        return
    if lang == 'CSharp':
        adv_base = f'data/tokenizer_attack_v2_cs_{mode}/{smell}/{DIM}'
    else:
        adv_base = f'data/tokenizer_attack_v2_{mode}/{smell}/{DIM}'
    if not os.path.exists(adv_base):
        print(f'ERROR: Adversarial tokenized data not found at {adv_base}')
        return
    fixed_lengths = {'FeatureEnvy': 4963, 'MultifacetedAbstraction': 5649, 'ComplexMethod': 1071, 'ComplexConditional': 464}
    max_len = fixed_lengths.get(smell, 5000)
    print(f'Loading data with fixed max_len={max_len} from {adv_base}...')
    pos_path = os.path.join(adv_base, 'Positive')
    neg_path = os.path.join(adv_base, 'Negative')
    pos_data = inputs._retrieve_data(pos_path, max_len)
    neg_data = inputs._retrieve_data(neg_path, max_len)
    pos_eval = pos_data[int(0.7 * len(pos_data)):]
    neg_eval = neg_data[int(0.7 * len(neg_data)):]
    if len(pos_eval) + len(neg_eval) == 0:
        print(f'CRITICAL ERROR: No data loaded from {adv_base}. Check directory contents.')
        for (root, dirs, files) in os.walk(adv_base):
            print(f'  DEBUG: root={root}, files={files[:5]}')
        return
    eval_data = np.array(pos_eval + neg_eval, dtype=np.float32)
    eval_labels = np.array([1.0] * len(pos_eval) + [0.0] * len(neg_eval), dtype=np.float32)
    print(f'Evaluating on {len(eval_data)} adversarial samples...')
    CHUNK_SIZE = 1000
    probs = []
    for i in range(0, len(eval_data), CHUNK_SIZE):
        chunk = eval_data[i:min(i + CHUNK_SIZE, len(eval_data))]
        probs.append(model.predict(chunk, batch_size=256, verbose=0))
    probs = np.concatenate(probs, axis=0)
    CLASSIFIER_THRESHOLD = 0.7
    if probs.ndim > 1 and probs.shape[1] == 2:
        probs_flat = probs[:, 1]
    else:
        probs_flat = probs.flatten()
    preds = (probs_flat > CLASSIFIER_THRESHOLD).astype(int)
    f1_fixed = f1_score(eval_labels, preds)
    acc = accuracy_score(eval_labels, preds)
    mcc = matthews_corrcoef(eval_labels, preds)
    try:
        auc = roc_auc_score(eval_labels, probs_flat)
    except:
        auc = 0.0
    try:
        (prec, rec, thresh) = precision_recall_curve(eval_labels, probs_flat)
        f1s = 2 * (prec * rec) / (prec + rec + 1e-09)
        f1_max = np.max(f1s)
    except:
        f1_max = 0.0
    print(f'RESULT: {lang} {smell} RNN [{mode}] -> F1(fixed): {f1_fixed:.4f}, F1(max): {f1_max:.4f}, AUC: {auc:.4f}, MCC: {mcc:.4f}, ACC: {acc:.4f}')
    res_file = os.path.join(OUT_FOLDER, f'rq2_rnn_{smell}.csv')
    with open(res_file, 'a') as f:
        f.write(f'{lang},{smell},{mode},{f1_fixed},{f1_max},{auc},{mcc},{acc}\n')
if __name__ == '__main__':
    lang = sys.argv[1]
    smell = sys.argv[2]
    mode = sys.argv[3]
    evaluate_rnn_rq2(lang, smell, mode)