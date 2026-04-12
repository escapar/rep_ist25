import os
import sys
import numpy as np
import path_config
import rq1_rnn_emb_lstm as rnn
import configuration

def run_best(lang, smell, ed, l, lstm, epochs):
    if os.environ.get('SMOKE_TEST'):
        print(f'!!! SMOKE TEST MODE ACTIVE for {smell} !!!')
        epochs = 1
        max_train = 10
        max_eval = 10
    else:
        max_train = 5000
        max_eval = 150000
    print(f'--- Running {lang} {smell} with Sharma BEST params ---')
    data_path = path_config.get_smell_path(lang, smell, '1d')
    data_obj = rnn.get_all_data(data_path, smell, max_training_samples=max_train, max_eval_samples=max_eval)
    config = configuration.RNN_emb_lstm_config(emb_output=ed, lstm_units=lstm, layers=l, epochs=epochs, dropout=0.2)
    smell_name = f'{smell}_Java' if lang == 'Java' else smell
    outfile = rnn.get_out_file_fixed(smell_name)
    with open(outfile, 'w') as f:
        f.write('embedding_out,rnn_layers,lstm_units,epochs,stopped_epoch,auc,accuracy,precision,recall,f1,average_precision,mcc,time\n')
    (auc, accuracy, precision, recall, f1, average_precision, stopped_epoch, mcc) = rnn.embedding_lstm(data_obj, config, smell_name, iteration='BEST')
    print(f'SUCCESS: {lang} {smell} -> F1: {f1:.4f}, AUC: {auc:.4f}, MCC: {mcc:.4f}')
if __name__ == '__main__':
    lang = sys.argv[1]
    smell = sys.argv[2]
    params = {'ComplexMethod': (32, 1, 64, 24), 'ComplexConditional': (32, 1, 64, 3), 'FeatureEnvy': (16, 2, 64, 16), 'MultifacetedAbstraction': (16, 2, 128, 11)}
    (ed, l, lstm, ep) = params[smell]
    run_best(lang, smell, ed, l, lstm, ep)