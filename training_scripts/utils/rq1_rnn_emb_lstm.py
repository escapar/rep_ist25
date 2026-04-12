from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
import configuration
import input_data
import inputs
import datetime
import numpy as np
import gc
import time
import metrics_util
import plot_util
from sklearn.dummy import DummyClassifier
DIM = '1d'
C2V = False
if C2V:
    TOKENIZER_OUT_PATH = '..\\..\\data\\c2v_vectors'
    OUT_FOLDER = '..\\results\\rq1\\raw'
else:
    TOKENIZER_OUT_PATH = '/root/autodl-tmp/cs_token_cpu3/'
    OUT_FOLDER = '../data_csvs'
TRAIN_VALIDATE_RATIO = 0.7
CLASSIFIER_THRESHOLD = 0.7

def embedding_lstm(data, config, smell, out_folder=OUT_FOLDER, dim=DIM, iteration=0, is_final=False):
    tf.keras.backend.clear_session()
    max_features = int(max(np.max(data.train_data), np.max(data.eval_data)))
    print('max features: ' + str(max_features))
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=max_features + 1, output_dim=config.emb_output, mask_zero=True))
    for i in range(0, config.layers - 1):
        model.add(tf.keras.layers.LSTM(config.lstm_units, return_sequences=True, recurrent_dropout=0.1, dropout=0.1))
    model.add(tf.keras.layers.LSTM(config.lstm_units, recurrent_dropout=0.1, dropout=0.1))
    model.add(tf.keras.layers.Dropout(config.dropout))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    class ThresholdCalibrationCallback(tf.keras.callbacks.Callback):

        def __init__(self, train_data, train_labels):
            super().__init__()
            split = int(len(train_data) * 0.8)
            self.v_data = train_data[split:]
            self.v_labels = train_labels[split:]

        def on_epoch_end(self, epoch, logs=None):
            val_probs = self.model.predict(self.v_data, batch_size=1024, verbose=0)
            from sklearn.metrics import precision_recall_curve
            (prec, rec, thresh) = precision_recall_curve(self.v_labels, val_probs)
            f1s = 2 * (prec * rec) / (prec + rec + 1e-10)
            idx = np.argmax(f1s)
            print(f'\n[Epoch {epoch + 1}] Val-Calibration: Max-F1={f1s[idx]:.4f} at Threshold={(thresh[idx] if idx < len(thresh) else 0.5):.4f}')
            sys.stdout.flush()
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto')
    best_model_filepath = '../data_csvs/weights_best.rnn.' + smell + str(iteration) + '.keras'
    if os.path.exists(best_model_filepath):
        print('deleting the old weights file..')
        os.remove(best_model_filepath)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=best_model_filepath, monitor='val_loss', verbose=1, save_best_only=True)
    calib_callback = ThresholdCalibrationCallback(data.train_data, data.train_labels)
    callbacks_list = [earlystop, checkpoint, calib_callback]
    b_size = 256
    import sys
    if is_final:
        print('[DEBUG] Starting model.fit (final mode)...')
        sys.stdout.flush()
        model.fit(data.train_data, data.train_labels, epochs=config.epochs, batch_size=b_size)
        stopped_epoch = config.epochs
        print('[DEBUG] model.fit finished.')
        sys.stdout.flush()
    else:
        print('[DEBUG] Starting model.fit (validation mode)...')
        sys.stdout.flush()
        model.fit(data.train_data, data.train_labels, validation_split=0.2, epochs=config.epochs, batch_size=b_size, callbacks=callbacks_list)
        stopped_epoch = earlystop.stopped_epoch
        print(f'[DEBUG] model.fit finished at epoch {stopped_epoch}. Loading best weights...')
        sys.stdout.flush()
        model.load_weights(best_model_filepath)
    print('[DEBUG] Calibrating optimal threshold on validation set...')
    sys.stdout.flush()
    CHUNK_SIZE = 1000
    if not is_final:
        val_split_idx = int(len(data.train_data) * 0.8)
        val_data = data.train_data[val_split_idx:]
        val_labels = data.train_labels[val_split_idx:]
        val_prob = []
        for i in range(0, len(val_data), CHUNK_SIZE):
            chunk = val_data[i:min(i + CHUNK_SIZE, len(val_data))]
            val_prob.append(model.predict(chunk, batch_size=256, verbose=0))
        val_prob = np.concatenate(val_prob, axis=0)
        from sklearn.metrics import precision_recall_curve
        if val_prob.ndim > 1 and val_prob.shape[1] == 1:
            val_prob_flat = val_prob.ravel()
        elif val_prob.ndim > 1 and val_prob.shape[1] == 2:
            val_prob_flat = val_prob[:, 1]
        else:
            val_prob_flat = val_prob
    CLASSIFIER_THRESHOLD = 0.7
    print(f'[DEBUG] Using FIXED Sharma Threshold: {CLASSIFIER_THRESHOLD}')
    sys.stdout.flush()
    print(f'[DEBUG] Starting chunked model.predict on {len(data.eval_data)} samples...')
    sys.stdout.flush()
    CHUNK_SIZE = 1000
    all_probs = []
    num_samples = len(data.eval_data)
    for i in range(0, num_samples, CHUNK_SIZE):
        end_idx = min(i + CHUNK_SIZE, num_samples)
        chunk = data.eval_data[i:end_idx]
        chunk_prob = model.predict(chunk, batch_size=256, verbose=0)
        all_probs.append(chunk_prob)
        if i % 5000 == 0:
            print(f'  - Predicted {end_idx}/{num_samples} samples...')
            sys.stdout.flush()
    prob = np.concatenate(all_probs, axis=0)
    print('[DEBUG] Prediction finished. Saving snapshots...')
    sys.stdout.flush()
    snapshot_prefix = f'../data_csvs/snapshot_{smell}_{iteration}'
    np.save(f'{snapshot_prefix}_prob.npy', prob)
    np.save(f'{snapshot_prefix}_labels.npy', data.eval_labels)
    print('[DEBUG] Applying threshold...')
    sys.stdout.flush()
    y_pred = inputs.get_predicted_y(prob, CLASSIFIER_THRESHOLD)
    print('[DEBUG] Calculating all metrics...')
    sys.stdout.flush()
    (auc, accuracy, precision, recall, f1, average_precision, fpr, tpr, mcc) = metrics_util.get_all_metrics(model, data.eval_data, data.eval_labels, y_pred)
    print('[DEBUG] Metrics calculation complete.')
    sys.stdout.flush()
    if is_final:
        print('[DEBUG] Saving plots...')
        sys.stdout.flush()
        plot_util.save_roc_curve(fpr, tpr, auc, smell, config, out_folder, DIM)
        plot_util.save_precision_recall_curve(data.eval_labels, y_pred, average_precision, smell, config, out_folder, dim, 'rnn')
    print('[DEBUG] Clearing Keras session...')
    sys.stdout.flush()
    tf.keras.backend.clear_session()
    return (auc, accuracy, precision, recall, f1, average_precision, stopped_epoch, mcc)

def start_training(data, config, conn, smell):
    try:
        return embedding_lstm(data, config, conn, smell)
    except Exception as ex:
        print(ex)
        return [-1, -1, -1]

def get_all_data(data_path, smell, max_training_samples=5000, max_eval_samples=None):
    print('reading data...')
    if max_eval_samples is None:
        if smell in ['ComplexConditional', 'ComplexMethod']:
            max_eval_samples = 150000
        else:
            max_eval_samples = 50000
    (train_data, train_labels, eval_data, eval_labels, max_input_length) = inputs.get_data(data_path, train_validate_ratio=TRAIN_VALIDATE_RATIO, max_training_samples=max_training_samples, max_eval_samples=max_eval_samples)
    train_data = train_data.reshape((len(train_labels), max_input_length))
    eval_data = eval_data.reshape((len(eval_labels), max_input_length))
    print('reading data... done.')
    return input_data.Input_data(train_data, train_labels, eval_data, eval_labels, max_input_length)

def write_result(file, str):
    f = open(file, 'a+')
    f.write(str)
    f.close()

def get_out_file_fixed(smell):
    now = datetime.datetime.now()
    if not os.path.exists(OUT_FOLDER):
        os.makedirs(OUT_FOLDER)
    if C2V:
        c2v = 'c2v'
    else:
        c2v = ''
    return os.path.join(OUT_FOLDER, 'rnn_rq1_' + smell + '_' + c2v + str(now.strftime('%d%m%Y_%H%M') + '.csv'))

def main(data_path, smell, skip_iter=-1, iterations_to_process=100):
    print('Starting preprocessing (de-duplication)...')
    sys.stdout.flush()
    inputs.preprocess_data(data_path)
    print('Preprocessing complete.')
    sys.stdout.flush()
    data = get_all_data(data_path, smell)
    configurations_list = [(16, 1, 32), (16, 1, 64), (16, 1, 128), (32, 1, 32), (32, 1, 64), (32, 1, 128), (16, 2, 32), (16, 2, 64), (16, 2, 128), (32, 2, 32), (32, 2, 64), (32, 2, 128)]
    total_iterations = len(configurations_list)
    cur_iter = 1
    outfile = get_out_file_fixed(smell)
    if not os.path.exists(outfile) or os.path.getsize(outfile) == 0:
        write_result(outfile, 'embedding_out,rnn_layers,lstm_units,epochs,stopped_epoch,auc,accuracy,precision,recall,f1,average_precision,mcc,time\n')
    for (emb_output, layer, lstm_units) in configurations_list:
        epoch = 50
        dropout = 0.2
        if cur_iter <= skip_iter:
            print('** Skipping Iteration {0} (skip_iter) **'.format(cur_iter))
            cur_iter += 1
            continue
        if os.path.exists(outfile):
            with open(outfile, 'r') as f:
                lines = f.readlines()
                exists = False
                for line in lines:
                    parts = line.split(',')
                    if len(parts) >= 4:
                        if parts[0] == str(emb_output) and parts[1] == str(layer) and (parts[2] == str(lstm_units)) and (parts[3] == str(epoch)):
                            exists = True
                            break
                if exists:
                    print('** Skipping Iteration {0} (Result already in CSV) **'.format(cur_iter))
                    cur_iter += 1
                    continue
        print('** Iteration {0} of 12 **'.format(cur_iter))
        config_obj = configuration.RNN_emb_lstm_config(emb_output=emb_output, lstm_units=lstm_units, layers=layer, epochs=epoch, dropout=dropout)
        try:
            start_time = time.time()
            (auc, accuracy, precision, recall, f1, average_precision, stopped_epoch, mcc) = embedding_lstm(data, config_obj, smell, iteration=cur_iter)
            end_time = time.time()
            elapsed_time = end_time - start_time
            result_str = str(emb_output) + ',' + str(layer) + ',' + str(lstm_units) + ',' + str(epoch) + ',' + str(stopped_epoch) + ',' + str(auc) + ',' + str(accuracy) + ',' + str(precision) + ',' + str(recall) + ',' + str(f1) + ',' + str(average_precision) + ',' + str(mcc) + ',' + str(elapsed_time) + '\n'
            write_result(outfile, result_str)
            gc.collect()
        except Exception as ex:
            print('Skipping combination layer: {}, emb_output: {}, lstm_units: {}, epoch: {}'.format(layer, emb_output, lstm_units, epoch))
            print(ex)
            write_result(outfile, str(emb_output) + ',' + str(layer) + ',' + str(lstm_units) + ',' + str(epoch) + ',-1,-1,-1,-1,-1,-1,-1,-1,-1\n')
        cur_iter += 1
        if cur_iter > iterations_to_process:
            print('Done with the specified number of iterations.')
            return

def run_rnn_with_best_params(smell, input_data, rnn_layers, emb_output, lstm_units, epochs):
    config = configuration.RNN_emb_lstm_config(emb_output=emb_output, lstm_units=lstm_units, layers=rnn_layers, epochs=epochs, dropout=0.2)
    outfile = get_out_file(smell + 'final')
    write_result(outfile, 'embedding_out,rnn_layers,lstm_units,epochs,stopped_epoch,auc,accuracy,precision,recall,f1,average_precision,mcc,time\n')
    try:
        start_time = time.time()
        (auc, accuracy, precision, recall, f1, average_precision, stopped_epoch, mcc) = embedding_lstm(input_data, config, smell, is_final=True)
        end_time = time.time()
        elapsed_time = end_time - start_time
        result_str = str(emb_output) + ',' + str(rnn_layers) + ',' + str(lstm_units) + ',' + str(epochs) + ',' + str(stopped_epoch) + ',' + str(auc) + ',' + str(accuracy) + ',' + str(precision) + ',' + str(recall) + ',' + str(f1) + ',' + str(average_precision) + ',' + str(elapsed_time) + '\n'
        write_result(outfile, result_str)
        gc.collect()
    except Exception as ex:
        print('Skipping combination layer: {}, emb_output: {}, lstm_units: {}, epoch: {}'.format(rnn_layers, emb_output, lstm_units, epochs))
        print(ex)
        write_result(outfile, str(emb_output) + ',' + str(rnn_layers) + ',' + str(lstm_units) + ',' + str(epochs) + ',-1,-1,-1,-1,-1,-1,-1\n')

def run_final():
    smell = 'ComplexMethod'
    data_path1 = os.path.join(os.path.join(TOKENIZER_OUT_PATH, smell), DIM)
    input_data1 = get_all_data(data_path1, smell)
    run_rnn_with_best_params(smell, input_data=input_data1, emb_output=32, rnn_layers=1, lstm_units=64, epochs=24)

def measure_random_performance():
    smell_list = {'ComplexMethod', 'EmptyCatchBlock', 'MagicNumber', 'MultifacetedAbstraction'}
    outfile = get_out_file('random_classifier')
    write_result(outfile, 'smell,auc,precision,recall,f1,average_precision\n')
    for smell in smell_list:
        data_path = os.path.join(os.path.join(TOKENIZER_OUT_PATH, smell), DIM)
        input_data = get_all_data(data_path, smell)
        y_pred = np.random.randint(2, size=len(input_data.eval_labels))
        (auc, precision, recall, f1, average_precision, fpr, tpr) = metrics_util.get_all_metrics_(input_data.eval_labels, y_pred)
        write_result(outfile, smell + ',' + str(auc) + ',' + str(precision) + ',' + str(recall) + ',' + str(f1) + ',' + str(average_precision) + '\n')

def measure_performance_dummy_classifier():
    outfile = get_out_file('dummy_classifier')
    write_result(outfile, 'smell,auc,precision,recall,f1,average_precision\n')
    for smell in smell_list:
        data_path = os.path.join(os.path.join(TOKENIZER_OUT_PATH, smell), DIM)
        input_data = get_all_data(data_path, smell)
        clf = DummyClassifier(strategy='most_frequent', random_state=0)
        inverted_train_labels = inputs.invert_labels(input_data.train_labels)
        clf.fit(input_data.train_data, inverted_train_labels)
        y_pred = clf.predict(input_data.eval_data)
        (auc, precision, recall, f1, average_precision, fpr, tpr) = metrics_util.get_all_metrics_(input_data.eval_labels, y_pred)
        write_result(outfile, smell + ',' + str(auc) + ',' + str(precision) + ',' + str(recall) + ',' + str(f1) + ',' + str(average_precision) + '\n')
if __name__ == '__main__':
    smell_list = {'ComplexMethod'}

def get_out_file_fixed(smell):
    out_dir = '../data_csvs'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    c2v = 'c2v' if C2V else ''
    return os.path.join(out_dir, f'rnn_rq1_{smell}_{c2v}.csv')