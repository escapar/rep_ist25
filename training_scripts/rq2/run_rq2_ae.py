import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))

import os
import sys
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score, matthews_corrcoef, accuracy_score
import json
import random
sys.path.append(os.path.abspath('program/dl_models'))
import inputs

DIM = '1d'
OUT_FOLDER = os.path.abspath('../results')

def load_data_ae_rq2(lang, smell, mode):
    config_path = os.path.join(os.path.dirname(__file__), '../config/dataset_splits.json')
    with open(config_path, 'r') as f:
        json_key = 'CSharp' if lang == 'CSharp' else 'Java'
        subset = json.load(f)[json_key][smell]
    if lang == 'CSharp':
        clean_path = f'data/tokenized/CSharp/{smell}/{DIM}'
        adv_base = f'data/synonym_attack/CSharp/{mode}/{smell}/{DIM}'
    else:
        clean_path = f'data/tokenized/Java/{smell}/{DIM}'
        adv_base = f'data/synonym_attack/Java/{mode}/{smell}/{DIM}'
    (_, _, _, _, max_len) = inputs.get_data(clean_path, max_training_samples=5000, max_eval_samples=50000, is_c2v=False)
    pos_files = subset['Positive']
    neg_files = subset['Negative']
    pos_train_names = pos_files[:int(0.7 * len(pos_files))]
    neg_train_names = neg_files[:int(0.7 * len(neg_files))]
    n_train_pos = min(len(pos_train_names), 5000)
    n_train_neg = min(len(neg_train_names), 5000)
    clean_pos_data = inputs._retrieve_data(os.path.join(clean_path, 'Positive'), max_len)[:n_train_pos]
    clean_neg_data = inputs._retrieve_data(os.path.join(clean_path, 'Negative'), max_len)[:n_train_neg]
    train_data = np.array(clean_pos_data + clean_neg_data, dtype=np.float32)
    pos_eval = inputs._retrieve_data(os.path.join(adv_base, 'Positive'), max_len)
    neg_eval = inputs._retrieve_data(os.path.join(adv_base, 'Negative'), max_len)
    pos_eval = pos_eval[int(0.7 * len(pos_eval)):]
    neg_eval = neg_eval[int(0.7 * len(neg_eval)):]
    eval_data = np.array(pos_eval + neg_eval, dtype=np.float32)
    eval_labels = np.array([1.0] * len(pos_eval) + [0.0] * len(neg_eval), dtype=np.float32)
    return (train_data, eval_data, eval_labels, max_len)

def evaluate_ae_rq2(lang, smell, mode, layer, encoding, ep, threshold_val):
    res_file = os.path.join(OUT_FOLDER, f'rq2_ae_{smell}.csv')
    if os.path.exists(res_file):
        with open(res_file, 'r') as f:
            for line in f:
                if f'{lang},{smell},{mode}' in line:
                    print(f'Skipping RQ2 AE {lang} {smell} {mode} - result exists.')
                    return
    print(f'\n--- RQ2 AE: {lang} {smell} [{mode}] ---', flush=True)
    (train_data, eval_data, eval_labels, max_input_length) = load_data_ae_rq2(lang, smell, mode)
    print(f'Training on {len(train_data)} clean samples, evaluating on {len(eval_data)} adversarial samples...')
    input_layer = Input(shape=(max_input_length,))
    prev_layer = input_layer
    for i in range(layer):
        prev_layer = Dense(int(encoding / pow(2, i)), activation='relu', activity_regularizer=regularizers.l1(0.01))(prev_layer)
    output_layer = Dense(max_input_length, activation='sigmoid')(prev_layer)
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder.fit(train_data, train_data, epochs=ep, batch_size=256, shuffle=True, verbose=0)
    predictions = autoencoder.predict(eval_data, batch_size=1024, verbose=0)
    mse = np.mean(np.power(eval_data - predictions, 2), axis=1)
    probs = mse / np.max(mse)
    preds = (mse > threshold_val).astype(int)
    f1_fixed = f1_score(eval_labels, preds)
    try:
        (prec, rec, thresh) = precision_recall_curve(eval_labels, probs)
        f1s = 2 * (prec * rec) / (prec + rec + 1e-09)
        f1_max = np.max(f1s)
    except:
        f1_max = 0.0
    try:
        auc = roc_auc_score(eval_labels, probs)
    except:
        auc = 0.0
    mcc = matthews_corrcoef(eval_labels, preds)
    acc = accuracy_score(eval_labels, preds)
    print(f'RESULT: {lang} {smell} RQ2 AE [{mode}] -> F1(fixed): {f1_fixed:.4f}, F1(max): {f1_max:.4f}, AUC: {auc:.4f}, MCC: {mcc:.4f}, ACC: {acc:.4f}')
    with open(res_file, 'a') as f:
        f.write(f'{lang},{smell},{mode},{f1_fixed},{f1_max},{auc},{mcc},{acc}\n')

if __name__ == '__main__':
    lang = sys.argv[1]
    smell = sys.argv[2]
    mode = sys.argv[3]
    params = {'ComplexMethod': (1, 32, 20, 325000), 'ComplexConditional': (1, 16, 20, 328000), 'FeatureEnvy': (2, 16, 20, 325000), 'MultifacetedAbstraction': (1, 16, 20, 328000)}
    (l, ed, ep, thr) = params[smell]
    evaluate_ae_rq2(lang, smell, mode, l, ed, ep, thr)
