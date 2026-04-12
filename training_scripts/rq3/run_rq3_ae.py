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
import path_config
DIM = '1d'
OUT_FOLDER = os.path.abspath('../data_csvs')

def load_data_ae_rq3(lang, smell, mode, ratio):
    with open('../config/dataset_splits.json', 'r') as f:
        subset = json.load(f)[lang][smell]
    clean_base = f'data/{lang.lower()}_subset_unique/{smell}'
    if lang == 'CSharp':
        adv_base = f'data/tokenizer_attack_v2_cs_{mode}/{smell}/{DIM}'
    else:
        adv_base = f'data/tokenizer_attack_v2_{mode}/{smell}/{DIM}'
    clean_path = path_config.get_smell_path(lang, smell, DIM)
    (_, _, _, _, max_len) = inputs.get_data(clean_path, max_training_samples=5000, max_eval_samples=50000, is_c2v=False)
    pos_files = subset['Positive']
    neg_files = subset['Negative']
    pos_train_names = pos_files[:int(0.7 * len(pos_files))]
    neg_train_names = neg_files[:int(0.7 * len(neg_files))]
    n_train_pos = min(len(pos_train_names), 5000)
    n_train_neg = min(len(neg_train_names), 5000)
    n_adv_pos = int(n_train_pos * ratio)
    n_adv_neg = int(n_train_neg * ratio)
    train_data = []
    clean_pos_data = inputs._retrieve_data(os.path.join(clean_path, 'Positive'), max_len)[:n_train_pos]
    adv_pos_data = inputs._retrieve_data(os.path.join(adv_base, 'Positive'), max_len)
    for i in range(n_train_pos):
        if i < n_adv_pos and i < len(adv_pos_data):
            train_data.append(adv_pos_data[i])
        else:
            train_data.append(clean_pos_data[i])
    clean_neg_data = inputs._retrieve_data(os.path.join(clean_path, 'Negative'), max_len)[:n_train_neg]
    adv_neg_data = inputs._retrieve_data(os.path.join(adv_base, 'Negative'), max_len)
    for i in range(n_train_neg):
        if i < n_adv_neg and i < len(adv_neg_data):
            train_data.append(adv_neg_data[i])
        else:
            train_data.append(clean_neg_data[i])
    train_data = np.array(train_data, dtype=np.float32)
    train_labels = np.array([1.0] * n_train_pos + [0.0] * n_train_neg, dtype=np.float32)
    pos_eval = inputs._retrieve_data(os.path.join(adv_base, 'Positive'), max_len)
    neg_eval = inputs._retrieve_data(os.path.join(adv_base, 'Negative'), max_len)
    pos_eval = pos_eval[int(0.7 * len(pos_eval)):]
    neg_eval = neg_eval[int(0.7 * len(neg_eval)):]
    eval_data = np.array(pos_eval + neg_eval, dtype=np.float32)
    eval_labels = np.array([1.0] * len(pos_eval) + [0.0] * len(neg_eval), dtype=np.float32)
    clean_pos_eval = inputs._retrieve_data(os.path.join(clean_path, 'Positive'), max_len)[int(0.7 * len(pos_files)):]
    clean_neg_eval = inputs._retrieve_data(os.path.join(clean_path, 'Negative'), max_len)[int(0.7 * len(neg_files)):]
    clean_eval_data = np.array(clean_pos_eval + clean_neg_eval, dtype=np.float32)
    clean_eval_labels = np.array([1.0] * len(clean_pos_eval) + [0.0] * len(clean_neg_eval), dtype=np.float32)
    return (train_data, train_labels, eval_data, eval_labels, clean_eval_data, clean_eval_labels, max_len)

def evaluate_ae_rq3(lang, smell, mode, ratio, layer, encoding, ep, threshold_val):
    res_file = os.path.join(OUT_FOLDER, f'rq3_ae_{smell}.csv')
    if os.path.exists(res_file):
        with open(res_file, 'r') as f:
            for line in f:
                if f'{lang},{smell},{mode},{ratio}' in line:
                    print(f'Skipping RQ3 AE {lang} {smell} {mode} {ratio} - result exists.')
                    return
    print(f'\n--- RQ3 AE Defense: {lang} {smell} [{mode}], Ratio: {ratio} ---', flush=True)
    (train_data, train_labels, eval_data, eval_labels, clean_eval_data, clean_eval_labels, max_input_length) = load_data_ae_rq3(lang, smell, mode, ratio)
    print(f'Training on {len(train_data)} mixed samples, evaluating on {len(eval_data)} adversarial samples...')
    input_layer = Input(shape=(max_input_length,))
    prev_layer = input_layer
    for i in range(layer):
        prev_layer = Dense(int(encoding / pow(2, i)), activation='relu', activity_regularizer=regularizers.l1(0.01))(prev_layer)
    output_layer = Dense(max_input_length, activation='sigmoid')(prev_layer)
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder.fit(train_data, train_data, epochs=ep, batch_size=256, shuffle=True, verbose=0)
    predictions = autoencoder.predict(eval_data, verbose=0)
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
    print(f'RESULT: {lang} {smell} RQ3 AE [{mode}, ratio={ratio}] -> F1(fixed): {f1_fixed:.4f}, F1(max): {f1_max:.4f}, AUC: {auc:.4f}, MCC: {mcc:.4f}, ACC: {acc:.4f}')
    with open(res_file, 'a') as f:
        f.write(f'{lang},{smell},{mode},{ratio},{f1_fixed},{f1_max},{auc},{mcc},{acc}\n')
    print('Evaluating Defense on Pure Clean Test Set...')
    c_predictions = autoencoder.predict(clean_eval_data, verbose=0)
    c_mse = np.mean(np.power(clean_eval_data - c_predictions, 2), axis=1)
    c_probs = c_mse / np.max(c_mse)
    c_preds = (c_mse > threshold_val).astype(int)
    c_f1_fixed = f1_score(clean_eval_labels, c_preds)
    try:
        (c_prec, c_rec, _) = precision_recall_curve(clean_eval_labels, c_probs)
        c_f1s = 2 * (c_prec * c_rec) / (c_prec + c_rec + 1e-09)
        c_f1_max = np.max(c_f1s)
    except:
        c_f1_max = 0.0
    try:
        c_auc = roc_auc_score(clean_eval_labels, c_probs)
    except:
        c_auc = 0.0
    c_mcc = matthews_corrcoef(clean_eval_labels, c_preds)
    c_acc = accuracy_score(clean_eval_labels, c_preds)
    print(f'CLEAN RESULT: {lang} {smell} RQ3 AE [{mode}, ratio={ratio}] -> F1(fixed): {c_f1_fixed:.4f}, F1(max): {c_f1_max:.4f}, AUC: {c_auc:.4f}, MCC: {c_mcc:.4f}, ACC: {c_acc:.4f}')
    c_res_file = f'../data_csvs/rq3_clean_ae_{smell}.csv'
    with open(c_res_file, 'a') as f:
        f.write(f'{lang},{smell},{mode},{ratio},{c_f1_fixed},{c_f1_max},{c_auc},{c_mcc},{c_acc}\n')
if __name__ == '__main__':
    lang = sys.argv[1]
    smell = sys.argv[2]
    mode = sys.argv[3]
    ratio = float(sys.argv[4])
    params = {'ComplexMethod': (1, 32, 20, 325000), 'ComplexConditional': (1, 16, 20, 328000), 'FeatureEnvy': (2, 16, 20, 325000), 'MultifacetedAbstraction': (1, 16, 20, 328000)}
    (l, ed, ep, thr) = params[smell]
    evaluate_ae_rq3(lang, smell, mode, ratio, l, ed, ep, thr)