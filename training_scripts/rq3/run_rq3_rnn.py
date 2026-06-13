import os
import sys
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, accuracy_score, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath('program/dl_models'))
import inputs
import path_config

DIM = "1d"
OUT_FOLDER = os.path.abspath('../results')
os.makedirs(OUT_FOLDER, exist_ok=True)

BEST_PARAMS = {
    "ComplexMethod": {"emb_output": 32, "layers": 1, "lstm_units": 64, "epochs": 24, "dropout": 0.2},
    "ComplexConditional": {"emb_output": 32, "layers": 1, "lstm_units": 64, "epochs": 3, "dropout": 0.2},
    "FeatureEnvy": {"emb_output": 16, "layers": 2, "lstm_units": 64, "epochs": 16, "dropout": 0.2},
    "MultifacetedAbstraction": {"emb_output": 16, "layers": 2, "lstm_units": 128, "epochs": 11, "dropout": 0.2}
}

def build_rnn_model(max_features, config):
    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(Embedding(input_dim=max_features + 1,
                        output_dim=config["emb_output"],
                        mask_zero=True))
    for i in range(0, config["layers"] - 1):
        model.add(LSTM(config["lstm_units"], return_sequences=True, recurrent_dropout=0.1, dropout=0.1, use_cudnn=False))
    model.add(LSTM(config["lstm_units"], recurrent_dropout=0.1, dropout=0.1, use_cudnn=False))
    model.add(Dropout(config["dropout"]))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

def evaluate_rnn_rq3(lang, smell, mode, ratio, smoke=False):
    print(f"\n--- RQ3 Defense (RNN): {lang} {smell} [{mode}], Ratio: {ratio} ---")
    
    clean_base = path_config.get_smell_path(lang, smell, DIM)
    inputs.preprocess_data(clean_base)
    
    if lang == "CSharp":
        adv_base = f"data/tokenizer_attack_v2_cs_{mode}/{smell}/{DIM}"
    else:
        adv_base = f"data/tokenizer_attack_v2_{mode}/{smell}/{DIM}"
        
    if not os.path.exists(adv_base):
        print(f"ERROR: Adversarial tokenized data not found at {adv_base}")
        return
        
    max_eval = 150000 if smell in ["ComplexMethod", "ComplexConditional"] else 50000
    train_limit = 10 if smoke else 5000
    
    clean_train_data, clean_train_labels, clean_eval_data, clean_eval_labels, max_len_clean = \
        inputs.get_data(clean_base, max_training_samples=train_limit, max_eval_samples=max_eval, is_c2v=False)
        
    adv_train_data, adv_train_labels, adv_eval_data, adv_eval_labels, max_len_adv = \
        inputs.get_data(adv_base, max_training_samples=train_limit, max_eval_samples=max_eval, is_c2v=False)
        
    max_input_length = max(max_len_clean, max_len_adv)
    max_features = int(max(np.max(clean_train_data), np.max(adv_train_data), np.max(adv_eval_data)))
    
    clean_train_data = clean_train_data.reshape((len(clean_train_data), max_len_clean))
    adv_train_data = adv_train_data.reshape((len(adv_train_data), max_len_adv))
    adv_eval_data = adv_eval_data.reshape((len(adv_eval_data), max_len_adv))
    clean_eval_data = clean_eval_data.reshape((len(clean_eval_data), max_len_clean))
    
    clean_train_data = tf.keras.preprocessing.sequence.pad_sequences(clean_train_data, maxlen=max_input_length, padding='pre')
    adv_train_data = tf.keras.preprocessing.sequence.pad_sequences(adv_train_data, maxlen=max_input_length, padding='pre')
    adv_eval_data = tf.keras.preprocessing.sequence.pad_sequences(adv_eval_data, maxlen=max_input_length, padding='pre')
    clean_eval_data_pad = tf.keras.preprocessing.sequence.pad_sequences(clean_eval_data, maxlen=max_input_length, padding='pre')
        
    n_total = len(clean_train_data)
    n_adv = int(n_total * ratio)
    n_clean = n_total - n_adv
    
    mixed_train_data = np.concatenate([clean_train_data[:n_clean], adv_train_data[:n_adv]])
    mixed_train_labels = np.concatenate([clean_train_labels[:n_clean], adv_train_labels[:n_adv]])
    
    idx = np.random.permutation(len(mixed_train_data))
    mixed_train_data = mixed_train_data[idx]
    mixed_train_labels = mixed_train_labels[idx]
    
    print(f"Data: {len(mixed_train_data)} Mixed Train (Ratio {ratio}), {len(adv_eval_data)} Adversarial Eval")
    
    config = BEST_PARAMS.get(smell, BEST_PARAMS["ComplexMethod"])
    model = build_rnn_model(max_features, config)
    
    epochs = 1 if smoke else config["epochs"]
    model.fit(mixed_train_data, mixed_train_labels, batch_size=32, epochs=epochs, verbose=1, validation_split=0.2)
    
    print("Evaluating on Adversarial Test Set...")
    CHUNK_SIZE = 1000
    probs = []
    for i in range(0, len(adv_eval_data), CHUNK_SIZE):
        chunk = adv_eval_data[i:min(i+CHUNK_SIZE, len(adv_eval_data))]
        probs.append(model.predict(chunk, batch_size=256, verbose=0))
    probs = np.concatenate(probs, axis=0)
    
    CLASSIFIER_THRESHOLD = 0.7
    probs_flat = probs.flatten()
        
    preds = (probs_flat > CLASSIFIER_THRESHOLD).astype(int)
    f1_fixed = f1_score(adv_eval_labels, preds)
    acc = accuracy_score(adv_eval_labels, preds)
    mcc = matthews_corrcoef(adv_eval_labels, preds)
    try:
        auc = roc_auc_score(adv_eval_labels, probs_flat)
    except:
        auc = 0.0
        
    try:
        prec, rec, thresh = precision_recall_curve(adv_eval_labels, probs_flat)
        f1s = 2 * (prec * rec) / (prec + rec + 1e-9)
        f1_max = np.max(f1s)
    except:
        f1_max = 0.0
    
    print(f"RESULT: {lang} {smell} RQ3 RNN [{mode}, ratio={ratio}] -> F1(fixed): {f1_fixed:.4f}, F1(max): {f1_max:.4f}, AUC: {auc:.4f}, MCC: {mcc:.4f}, ACC: {acc:.4f}")
    
    res_file = os.path.join(OUT_FOLDER, f"rq3_rnn_{smell}.csv")
    with open(res_file, "a") as f:
        f.write(f"{lang},{smell},{mode},{ratio},{f1_fixed},{f1_max},{auc},{mcc},{acc}\n")
        
    print("Evaluating Defense on Pure Clean Test Set...")
    c_probs = []
    for i in range(0, len(clean_eval_data_pad), CHUNK_SIZE):
        chunk = clean_eval_data_pad[i:min(i+CHUNK_SIZE, len(clean_eval_data_pad))]
        c_probs.append(model.predict(chunk, batch_size=256, verbose=0))
    c_probs = np.concatenate(c_probs, axis=0)
    
    c_probs_flat = c_probs.flatten()
        
    c_preds = (c_probs_flat > CLASSIFIER_THRESHOLD).astype(int)
    c_f1_fixed = f1_score(clean_eval_labels, c_preds)
    try:
        c_prec, c_rec, _ = precision_recall_curve(clean_eval_labels, c_probs_flat)
        c_f1s = 2 * (c_prec * c_rec) / (c_prec + c_rec + 1e-9)
        c_f1_max = np.max(c_f1s)
    except:
        c_f1_max = 0.0
    try:
        c_auc = roc_auc_score(clean_eval_labels, c_probs_flat)
    except:
        c_auc = 0.0
    c_mcc = matthews_corrcoef(clean_eval_labels, c_preds)
    c_acc = accuracy_score(clean_eval_labels, c_preds)
    
    print(f"CLEAN RESULT: {lang} {smell} RQ3 RNN [{mode}, ratio={ratio}] -> F1(fixed): {c_f1_fixed:.4f}, F1(max): {c_f1_max:.4f}, AUC: {c_auc:.4f}, MCC: {c_mcc:.4f}, ACC: {c_acc:.4f}")
    c_res_file = os.path.join(OUT_FOLDER, f"rq3_clean_rnn_{smell}.csv")
    with open(c_res_file, "a") as f:
        f.write(f"{lang},{smell},{mode},{ratio},{c_f1_fixed},{c_f1_max},{c_auc},{c_mcc},{c_acc}\n")

if __name__ == "__main__":
    lang = sys.argv[1]
    smell = sys.argv[2]
    mode = sys.argv[3]
    ratio = float(sys.argv[4])
    smoke = os.environ.get("SMOKE_TEST") == "1"
    evaluate_rnn_rq3(lang, smell, mode, ratio, smoke)
