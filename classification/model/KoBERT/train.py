import gluonnlp as nlp
import pandas as pd
import torch
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer
from BERT import BERTDataset, BERTClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup


def load_data(train_dir: str, test_dir: str):
    df_train = pd.read_csv(train_dir)
    df_train = df_train.dropna()
    train_x, train_y = df_train["text"].tolist(), df_train["label"].tolist()

    df_test = pd.read_csv(test_dir)
    df_test = df_test.dropna()
    test_x, test_y = df_test["text"].tolist(), df_test["label"].tolist()

    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, stratify=train_y)

    label_enc = LabelEncoder()
    train_y = label_enc.fit_transform(train_y)
    val_y = label_enc.transform(val_y)
    test_y = label_enc.transform(test_y)

    return train_x, train_y, val_x, val_y, test_x, test_y, label_enc


def calculate_accuracy(x, y):
    max_vals, max_indices = torch.max(x, 1)
    train_acc = (max_indices == y).sum().data.cpu().numpy() / max_indices.size()[0]
    return train_acc


def train(model, train_loader, val_loader, epoch: int, optimizer, loss_fn, scheduler):
    best_acc = 0.0
    for i in range(epoch):
        train_acc, val_acc = 0.0, 0.0

        model.train()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(train_loader):
            optimizer.zero_grad()
            pred_y = model(token_ids.long().to(device), valid_length, segment_ids.long().to(device))
            y = label.long().to(device)
            loss = loss_fn(pred_y, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            train_acc += calculate_accuracy(pred_y, y)

            if batch_id % (len(train_loader) // 10) == 0: print(batch_id, len(train_loader))

        print("epoch {} train acc {}".format(i+1, train_acc/(batch_id+1)))

        model.eval()
        with torch.no_grad():
            for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(val_loader):
                pred_y = model(token_ids.long().to(device), valid_length, segment_ids.long().to(device))
                y = label.long().to(device)
                val_acc += calculate_accuracy(pred_y, y)

        val_score = val_acc / (batch_id + 1)
        print("epoch {} val acc {}".format(i+1, val_score))

        if val_score > best_acc:
            best_acc = val_score
            best_model = model

    return best_model


def test(test_x: list, test_y: list, test_loader, model, label_enc):
    test_y, pred_y = label_enc.inverse_transform(test_y), []

    model.eval()
    with torch.no_grad():
        for token_ids, valid_length, segment_ids, label in test_loader:
            output = model(token_ids.long().to(device), valid_length, segment_ids.long().to(device))
            _, output = torch.max(output, 1)
            output = label_enc.inverse_transform(output.cpu())
            pred_y.extend(output)

    df_result = pd.DataFrame({"sentence": test_x, "pred_y": pred_y, "test_y": test_y})
    df_result.to_csv("result.csv", index=False, encoding="utf8")

    f1 = f1_score(test_y, pred_y, average='weighted')
    precision = precision_score(test_y, pred_y, average='weighted')
    recall = recall_score(test_y, pred_y, average='weighted')
    return f1, precision, recall


if __name__ == "__main__":
    # Setting Directory
    train_dir = "../../data/train_binary.csv"
    test_dir = "../../data/test_binary.csv"

    # Setting Parameter
    device = torch.device("cuda")

    # Hyper Parameter
    epoch = 1
    max_len = 10
    batch_size = 128
    max_grad_norm = 1
    warmup_ratio = 0.1
    log_interval = 200
    lr = 5e-5
    dr_rate = 0.5

    # Flow
    print("1. load data")
    train_x, train_y, val_x, val_y, test_x, test_y, label_enc = load_data(train_dir, test_dir)

    print("2. text to vector")
    kobert_model, vocab = get_pytorch_kobert_model()
    tokenizer = get_tokenizer()
    tokenizer = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    train_set = BERTDataset(train_x, train_y, tokenizer, max_len, pad=True, pair=False)
    val_set = BERTDataset(val_x, val_y, tokenizer, max_len, pad=True, pair=False)
    test_set = BERTDataset(test_x, test_y, tokenizer, max_len, pad=True, pair=False)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print("3. build model")
    model = BERTClassifier(kobert_model, num_classes=len(label_enc.classes_), dr_rate=dr_rate).to(device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    loss_fn = nn.CrossEntropyLoss()
    total_step = len(train_set) * epoch
    warmup_step = int(total_step * warmup_ratio)
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=total_step)

    print("3. train")
    model = train(model, train_loader, val_loader, epoch, optimizer, loss_fn, scheduler)

    print("4. test")
    f1, precision, recall = test(test_x, test_y, test_loader, model, label_enc)