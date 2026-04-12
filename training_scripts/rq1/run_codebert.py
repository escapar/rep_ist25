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
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef
MAX_LEN = 512
BATCH_SIZE = 16
LR = 2e-05
EPOCHS = 3
MODEL_NAME = 'microsoft/codebert-base'
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

class SmellDataset(Dataset):

    def __init__(self, files, labels, tokenizer, max_len):
        self.files = files
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        file_path = self.files[item]
        label = self.labels[item]
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
        except:
            code = ''
        encoding = self.tokenizer(code, add_special_tokens=True, max_length=self.max_len, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'labels': torch.tensor(label, dtype=torch.long)}

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

def train_epoch(model, data_loader, optimizer, device, scheduler):
    model.train()
    losses = []
    for d in tqdm(data_loader):
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        labels = d['labels'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return np.mean(losses)

def eval_model(model, data_loader, device):
    model.eval()
    predictions = []
    real_values = []
    with torch.no_grad():
        for d in tqdm(data_loader):
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            labels = d['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            predictions.extend(probs[:, 1].cpu().numpy())
            real_values.extend(labels.cpu().numpy())
    return (np.array(predictions), np.array(real_values))

def run_fine_tuning(lang, smell, smoke=False):
    print(f'\n--- CodeBERT Fine-Tuning: {lang} {smell} (Smoke={smoke}) ---')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    (train_f, train_l, eval_f, eval_l) = load_data_from_json(lang, smell, smoke)
    print(f'Data: {len(train_f)} Train, {len(eval_f)} Eval')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = SmellDataset(train_f, train_l, tokenizer, MAX_LEN)
    eval_ds = SmellDataset(eval_f, eval_l, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * (1 if smoke else EPOCHS)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    for epoch in range(1 if smoke else EPOCHS):
        loss = train_epoch(model, train_loader, optimizer, device, scheduler)
        print(f'Epoch {epoch + 1} Loss: {loss:.4f}')
    (probs, labels) = eval_model(model, eval_loader, device)
    preds = (probs > 0.7).astype(int)
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    f1 = f1_score(labels, preds)
    auc = roc_auc_score(labels, probs)
    mcc = matthews_corrcoef(labels, preds)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    print(f'RESULT: {lang} {smell} CodeBERT -> F1: {f1:.4f}, AUC: {auc:.4f}, MCC: {mcc:.4f}, ACC: {acc:.4f}, PREC: {prec:.4f}, REC: {rec:.4f}')
    res_file = f'../data_csvs/codebert_rq1_{smell}.csv'
    with open(res_file, 'a') as f:
        f.write(f'{lang},{smell},{f1},{auc},{mcc},{acc},{prec},{rec}\n')
if __name__ == '__main__':
    lang = sys.argv[1]
    smell = sys.argv[2]
    smoke = os.environ.get('SMOKE_TEST') == '1'
    run_fine_tuning(lang, smell, smoke)