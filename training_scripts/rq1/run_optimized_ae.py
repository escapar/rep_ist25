import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))

import os
import sys
import time
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
import path_config
import inputs
import metrics_util
DIM = '1d'
OUT_FOLDER = os.path.abspath('../results')
os.makedirs(OUT_FOLDER, exist_ok=True)

def run_ae_best(lang, smell, layer, encoding, ep, threshold_val):
    if os.environ.get('SMOKE_TEST'):
        print(f'!!! SMOKE TEST MODE ACTIVE for {smell} !!!')
        ep = 1
        max_train = 10
        max_eval = 10
    else:
        max_train = 5000
        max_eval = 150000
    print(f'\n--- Running AUTOENCODER for {lang} {smell} ---', flush=True)
    data_path = path_config.get_smell_path(lang, smell, DIM)
    inputs.preprocess_data(data_path)
    (train_data, training_labels, eval_data, eval_labels, max_input_length) = inputs.get_data(data_path, max_training_samples=max_train, max_eval_samples=max_eval, is_c2v=False)
    input_layer = Input(shape=(max_input_length,))
    prev_layer = input_layer
    for i in range(layer):
        prev_layer = Dense(int(encoding / pow(2, i)), activation='relu', activity_regularizer=regularizers.l1(0.01))(prev_layer)
    prev_layer = Dense(int(encoding / pow(2, layer)), activation='relu')(prev_layer)
    for j in range(layer - 1, -1, -1):
        prev_layer = Dense(int(encoding / pow(2, j)), activation='relu')(prev_layer)
    output_layer = Dense(max_input_length, activation='relu')(prev_layer)
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder.fit(train_data, train_data, epochs=ep, batch_size=128, verbose=1, validation_split=0.2, shuffle=True)
    CHUNK_SIZE = 1000
    all_mse = []
    num_samples = len(eval_data)
    for i in range(0, num_samples, CHUNK_SIZE):
        end_idx = min(i + CHUNK_SIZE, num_samples)
        chunk = eval_data[i:end_idx]
        preds = autoencoder.predict(chunk, verbose=0)
        preds = np.reshape(preds, chunk.shape)
        chunk_sse = np.sum(np.power(chunk - preds, 2), axis=1)
        all_mse.append(chunk_sse)
    error_combined = np.concatenate(all_mse)
    print(f'[DEBUG] Error array shape: {error_combined.shape}, labels shape: {eval_labels.shape}')
    y_pred_fixed = (error_combined > threshold_val).astype(int)
    f1_fixed = f1_score(eval_labels, y_pred_fixed)
    (prec, rec, thresh) = precision_recall_curve(eval_labels, error_combined)
    f1s = 2 * (prec * rec) / (prec + rec + 1e-10)
    best_idx = np.argmax(f1s)
    f1_max = f1s[best_idx]
    auc = roc_auc_score(eval_labels, error_combined)
    mcc = matthews_corrcoef(eval_labels, y_pred_fixed)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(eval_labels, y_pred_fixed)
    prec_fixed = precision_score(eval_labels, y_pred_fixed)
    rec_fixed = recall_score(eval_labels, y_pred_fixed)
    print(f'SUCCESS: {lang} {smell} AE -> F1(fixed): {f1_fixed:.4f}, F1(max): {f1_max:.4f}, AUC: {auc:.4f}, MCC: {mcc:.4f}, ACC: {acc:.4f}, PREC: {prec_fixed:.4f}, REC: {rec_fixed:.4f}')
    outfile = os.path.join(OUT_FOLDER, f'ae_rq1_{smell}.csv')
    with open(outfile, 'a') as f:
        f.write(f'{lang},{smell},{encoding},{layer},{f1_fixed},{f1_max},{auc},{mcc},{acc},{prec_fixed},{rec_fixed}\n')
if __name__ == '__main__':
    lang = sys.argv[1]
    smell = sys.argv[2]
    params = {'ComplexMethod': (1, 32, 20, 319000), 'ComplexConditional': (1, 16, 20, 328000), 'FeatureEnvy': (2, 16, 20, 325000), 'MultifacetedAbstraction': (1, 16, 20, 328000)}
    (l, ed, ep, thr) = params[smell]
    run_ae_best(lang, smell, l, ed, ep, thr)