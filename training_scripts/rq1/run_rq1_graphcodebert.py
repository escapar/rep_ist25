import os
import sys
import json
import random
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, accuracy_score, precision_score, recall_score
MAX_LEN = 512
BATCH_SIZE = 16
LR = 2e-05
EPOCHS = 3
MODEL_NAME = 'microsoft/graphcodebert-base'
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

class SmellDataset(Dataset):

    def __init__(self, file_paths, labels, tokenizer, max_len):
        self.file_paths = file_paths
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
        except:
            code = ''
        encoding = self.tokenizer(code, add_special_tokens=True, max_length=self.max_len, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'labels': torch.tensor(self.labels[idx], dtype=torch.long)}

def load_data_from_json(lang, smell, smoke=False):
    with open('../config/dataset_splits.json', 'r') as f:
        subset = json.load(f)[lang][smell]
    input_base = f'data/{lang.lower()}_subset_unique/{smell}'
    pos_files = [os.path.join(input_base, 'Positive', f) for f in subset['Positive']]
    neg_files = [os.path.join(input_base, 'Negative', f) for f in subset['Negative']]
    random.shuffle(pos_files)
    random.shuffle(neg_files)
    pos_train = pos_files[:int(0.7 * len(pos_files))]
    pos_eval = pos_files[int(0.7 * len(pos_files)):]
    neg_train = neg_files[:int(0.7 * len(neg_files))]
    neg_eval = neg_files[int(0.7 * len(neg_files)):]
    train_limit = 10 if smoke else 5000
    n_train = min(len(pos_train), len(neg_train), train_limit)
    train_files = pos_train[:n_train] + neg_train[:n_train]
    train_labels = [1] * n_train + [0] * n_train
    combined = list(zip(train_files, train_labels))
    random.shuffle(combined)
    (train_files, train_labels) = zip(*combined)
    (train_files, train_labels) = (list(train_files), list(train_labels))
    eval_limit = 10 if smoke else 150000 if smell in ['ComplexMethod', 'ComplexConditional'] else 50000
    total_eval_avail = len(pos_eval) + len(neg_eval)
    if total_eval_avail > eval_limit:
        ratio = len(pos_eval) / total_eval_avail
        n_pos_eval = int(eval_limit * ratio)
        n_neg_eval = eval_limit - n_pos_eval
        eval_files = pos_eval[:n_pos_eval] + neg_eval[:n_neg_eval]
        eval_labels = [1] * n_pos_eval + [0] * n_neg_eval
    else:
        eval_files = pos_eval + neg_eval
        eval_labels = [1] * len(pos_eval) + [0] * len(neg_eval)
    return (train_files, train_labels, eval_files, eval_labels)

def eval_model(model, dataloader, device):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    return (np.array(all_probs), np.array(all_labels))

def run_fine_tuning(lang, smell, smoke=False):
    print(f'\n--- RQ1 Baseline (GraphCodeBERT): {lang} {smell} ---')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    (train_f, train_l, eval_f, eval_l) = load_data_from_json(lang, smell, smoke)
    print(f'Data: {len(train_f)} Train, {len(eval_f)} Eval')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = SmellDataset(train_f, train_l, tokenizer, MAX_LEN)
    eval_ds = SmellDataset(eval_f, eval_l, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * (1 if smoke else EPOCHS)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    for epoch in range(1 if smoke else EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch + 1}')
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        print(f'Epoch {epoch + 1} Loss: {total_loss / len(train_loader):.4f}')
    print('Evaluating...')
    (probs, labels) = eval_model(model, eval_loader, device)
    preds = (probs > 0.5).astype(int)
    f1 = f1_score(labels, preds)
    auc = roc_auc_score(labels, probs)
    mcc = matthews_corrcoef(labels, preds)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    print(f'RESULT: {lang} {smell} GraphCodeBERT -> F1: {f1:.4f}, AUC: {auc:.4f}, MCC: {mcc:.4f}, ACC: {acc:.4f}, PREC: {prec:.4f}, REC: {rec:.4f}')
    os.makedirs('../data_csvs', exist_ok=True)
    res_file = f'../data_csvs/graphcodebert_rq1_{smell}.csv'
    with open(res_file, 'a') as f:
        f.write(f'{lang},{smell},{f1},{auc},{mcc},{acc},{prec},{rec}\n')
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python run_rq1_graphcodebert.py <Lang> <Smell>')
        sys.exit(1)
    lang = sys.argv[1]
    smell = sys.argv[2]
    smoke = os.environ.get('SMOKE_TEST') == '1'
    run_fine_tuning(lang, smell, smoke)