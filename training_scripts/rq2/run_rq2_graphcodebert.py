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
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, accuracy_score
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

def load_data(lang, smell, mode, smoke=False):
    with open('../config/dataset_splits.json', 'r') as f:
        subset = json.load(f)[lang][smell]
    clean_base = f'data/{lang.lower()}_subset_unique/{smell}'
    if lang == 'CSharp':
        adv_base = f'data/attack_v2_cs_{mode}/{smell}'
    else:
        adv_base = f'data/attack_v2_{mode}/{smell}'
    pos_files = subset['Positive']
    neg_files = subset['Negative']
    random.shuffle(pos_files)
    random.shuffle(neg_files)
    pos_train_names = pos_files[:int(0.7 * len(pos_files))]
    neg_train_names = neg_files[:int(0.7 * len(neg_files))]
    train_limit = 10 if smoke else 5000
    n_train = min(len(pos_train_names), len(neg_train_names), train_limit)
    train_files = [os.path.join(clean_base, 'Positive', f) for f in pos_train_names[:n_train]] + [os.path.join(clean_base, 'Negative', f) for f in neg_train_names[:n_train]]
    train_labels = [1] * n_train + [0] * n_train
    pos_eval_names = pos_files[int(0.7 * len(pos_files)):]
    neg_eval_names = neg_files[int(0.7 * len(neg_files)):]
    eval_limit = 10 if smoke else 150000 if smell in ['ComplexMethod', 'ComplexConditional'] else 50000
    eval_files = []
    eval_labels = []
    if not os.path.exists(adv_base):
        print(f'Warning: Adv base {adv_base} does not exist!')
    for f in pos_eval_names:
        p = os.path.join(adv_base, 'Positive', f)
        if not os.path.exists(p):
            for ext in ['.code', '.java', '.cs']:
                alt_p = os.path.splitext(p)[0] + ext
                if os.path.exists(alt_p):
                    p = alt_p
                    break
        if os.path.exists(p):
            eval_files.append(p)
            eval_labels.append(1)
    for f in neg_eval_names:
        p = os.path.join(adv_base, 'Negative', f)
        if not os.path.exists(p):
            for ext in ['.code', '.java', '.cs']:
                alt_p = os.path.splitext(p)[0] + ext
                if os.path.exists(alt_p):
                    p = alt_p
                    break
        if os.path.exists(p):
            eval_files.append(p)
            eval_labels.append(0)
    if len(eval_files) > eval_limit:
        combined = list(zip(eval_files, eval_labels))
        combined = random.sample(combined, eval_limit)
        (eval_files, eval_labels) = zip(*combined)
        (eval_files, eval_labels) = (list(eval_files), list(eval_labels))
    return (train_files, train_labels, eval_files, eval_labels)

def train_epoch(model, data_loader, optimizer, device, scheduler):
    model.train()
    losses = []
    for d in tqdm(data_loader, desc='Training'):
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
    probs = []
    real_values = []
    with torch.no_grad():
        for d in tqdm(data_loader, desc='Evaluating (Adv)'):
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            labels = d['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            batch_probs = torch.softmax(logits, dim=1)
            probs.extend(batch_probs[:, 1].cpu().numpy())
            real_values.extend(labels.cpu().numpy())
    return (np.array(probs), np.array(real_values))

def run_rq2_codebert(lang, smell, mode, smoke=False):
    res_file = f'../data_csvs/rq2_graphcodebert_{smell}.csv'
    if os.path.exists(res_file):
        with open(res_file, 'r') as f:
            for line in f:
                if f'{lang},{smell},{mode}' in line:
                    print(f'Skipping {lang} {smell} {mode} - result already exists in {res_file}')
                    return
    print(f'\n--- RQ2 GraphCodeBERT: {lang} {smell} [{mode}] ---')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    (train_f, train_l, eval_f, eval_l) = load_data(lang, smell, mode, smoke)
    print(f'Data: {len(train_f)} Clean Train, {len(eval_f)} Adversarial Eval')
    if not eval_f:
        print('No adversarial eval data found!')
        return
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = SmellDataset(train_f, train_l, tokenizer, MAX_LEN)
    eval_ds = SmellDataset(eval_f, eval_l, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
    print('Re-fine-tuning on Clean Training Set to recreate RQ1 model state...')
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * (1 if smoke else EPOCHS)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    for epoch in range(1 if smoke else EPOCHS):
        loss = train_epoch(model, train_loader, optimizer, device, scheduler)
        print(f'Epoch {epoch + 1} Loss: {loss:.4f}')
    print('Evaluating on Adversarial Test Set...')
    (probs, labels) = eval_model(model, eval_loader, device)
    preds = (probs > 0.7).astype(int)
    f1_fixed = f1_score(labels, preds)
    from sklearn.metrics import precision_recall_curve
    try:
        (prec, rec, thresh) = precision_recall_curve(labels, probs)
        f1s = 2 * (prec * rec) / (prec + rec + 1e-09)
        f1_max = np.max(f1s)
    except:
        f1_max = 0.0
    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = 0.0
    mcc = matthews_corrcoef(labels, preds)
    acc = accuracy_score(labels, preds)
    print(f'RESULT: {lang} {smell} GraphCodeBERT [{mode}] -> F1(fixed): {f1_fixed:.4f}, F1(max): {f1_max:.4f}, AUC: {auc:.4f}, MCC: {mcc:.4f}, ACC: {acc:.4f}')
    os.makedirs('../data_csvs', exist_ok=True)
    with open(res_file, 'a') as f:
        f.write(f'{lang},{smell},{mode},{f1_fixed},{f1_max},{auc},{mcc},{acc}\n')
if __name__ == '__main__':
    lang = sys.argv[1]
    smell = sys.argv[2]
    mode = sys.argv[3]
    smoke = os.environ.get('SMOKE_TEST') == '1'
    run_rq2_codebert(lang, smell, mode, smoke)