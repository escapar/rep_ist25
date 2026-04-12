import os
import sys
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score, matthews_corrcoef, accuracy_score
sys.path.append(os.path.abspath('program/dl_models'))
import inputs
import path_config
DIM = '1d'
OUT_FOLDER = os.path.abspath('../data_csvs')

def evaluate_ae_rq2(lang, smell, mode, layer, encoding, ep, threshold_val):
    res_file = os.path.join(OUT_FOLDER, f'rq2_ae_{smell}.csv')
    if os.path.exists(res_file):
        with open(res_file, 'r') as f:
            for line in f:
                if f'{lang},{smell},{mode}' in line:
                    print(f'Skipping AE {lang} {smell} {mode} - result exists.')
                    return
    print(f'\n--- RQ2 Autoencoder Evaluation: {lang} {smell} [{mode}] ---', flush=True)
    clean_data_path = path_config.get_smell_path(lang, smell, DIM)
    inputs.preprocess_data(clean_data_path)
    if lang == 'CSharp':
        adv_base = f'data/tokenizer_attack_v2_cs_{mode}/{smell}/{DIM}'
    else:
        adv_base = f'data/tokenizer_attack_v2_{mode}/{smell}/{DIM}'
    if not os.path.exists(adv_base):
        print(f'ERROR: Adversarial tokenized data not found at {adv_base}')
        return
    max_eval = 150000 if smell in ['ComplexMethod', 'ComplexConditional'] else 50000
    (train_data, _, _, _, max_input_length) = inputs.get_data(clean_data_path, max_training_samples=5000, max_eval_samples=max_eval, is_c2v=False)
    pos_eval = inputs._retrieve_data(os.path.join(adv_base, 'Positive'), max_input_length, is_c2v=False)
    neg_eval = inputs._retrieve_data(os.path.join(adv_base, 'Negative'), max_input_length, is_c2v=False)
    pos_eval = pos_eval[int(0.7 * len(pos_eval)):]
    neg_eval = neg_eval[int(0.7 * len(neg_eval)):]
    if len(pos_eval) + len(neg_eval) > max_eval:
        ratio = max_eval / (len(pos_eval) + len(neg_eval))
        pos_eval = pos_eval[:int(len(pos_eval) * ratio)]
        neg_eval = neg_eval[:int(len(neg_eval) * ratio)]
    eval_data = np.concatenate([pos_eval, neg_eval], axis=0)
    eval_labels = np.concatenate([np.ones(len(pos_eval)), np.zeros(len(neg_eval))], axis=0)
    print(f'Training on {len(train_data)} clean samples, evaluating on {len(eval_data)} adversarial samples...')
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
    autoencoder.fit(train_data, train_data, epochs=ep, batch_size=128, verbose=0, validation_split=0.2, shuffle=True)
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
    y_pred_fixed = (error_combined > threshold_val).astype(int)
    f1_fixed = f1_score(eval_labels, y_pred_fixed)
    acc = accuracy_score(eval_labels, y_pred_fixed)
    mcc = matthews_corrcoef(eval_labels, y_pred_fixed)
    try:
        auc = roc_auc_score(eval_labels, error_combined)
    except:
        auc = 0.0
    try:
        (prec, rec, thresh) = precision_recall_curve(eval_labels, error_combined)
        f1s = 2 * (prec * rec) / (prec + rec + 1e-09)
        f1_max = np.max(f1s)
    except:
        f1_max = 0.0
    print(f'RESULT: {lang} {smell} AE [{mode}] -> F1(fixed): {f1_fixed:.4f}, F1(max): {f1_max:.4f}, AUC: {auc:.4f}, MCC: {mcc:.4f}, ACC: {acc:.4f}')
    res_file = os.path.join(OUT_FOLDER, f'rq2_ae_{smell}.csv')
    with open(res_file, 'a') as f:
        f.write(f'{lang},{smell},{mode},{f1_fixed},{f1_max},{auc},{mcc},{acc}\n')
if __name__ == '__main__':
    lang = sys.argv[1]
    smell = sys.argv[2]
    mode = sys.argv[3]
    params = {'ComplexMethod': (1, 32, 20, 319000), 'ComplexConditional': (1, 16, 20, 328000), 'FeatureEnvy': (2, 16, 20, 325000), 'MultifacetedAbstraction': (1, 16, 20, 328000)}
    (l, ed, ep, thr) = params[smell]
    evaluate_ae_rq2(lang, smell, mode, l, ed, ep, thr)