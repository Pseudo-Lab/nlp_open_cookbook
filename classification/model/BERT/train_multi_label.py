import gluonnlp as nlp
import pandas as pd
import numpy as np
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
from sklearn.metrics import classification_report


def load_data(train_dir: str, test_dir: str):
    df_train = pd.read_csv(train_dir)
    df_train = df_train.dropna()

    df_train, df_val = train_test_split(df_train, test_size=0.1)
    
    train_x, train_y = df_train["문장"].tolist(), []
    for i in range(0, len(df_train)):
        train_y.append(list(df_train.iloc[i][1:].values))

    val_x, val_y = df_val["문장"].tolist(), []
    for i in range(0, len(df_val)):
        val_y.append(list(df_val.iloc[i][1:].values))

    df_test = pd.read_csv(test_dir)
    df_test = df_test.dropna()

    test_x, test_y = df_test["문장"].tolist(), []
    for i in range(0, len(df_test)):
        test_y.append(list(df_test.iloc[i][1:].values))

    return train_x, train_y, val_x, val_y, test_x, test_y


def calculate_accuracy(x, y):
    max_vals, max_indices = torch.max(x, 1)
    train_acc = (max_indices == y).sum().data.cpu().numpy() / max_indices.size()[0]
    return train_acc


def train(model, train_loader, val_loader, epoch, optimizer, loss_fn):
    best_loss = np.Inf
    for i in range(epoch):
        train_loss, val_loss = 0.0, 0.0

        model.train()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(train_loader):
            optimizer.zero_grad()
            pred_y = model(token_ids.long().to(device), valid_length, segment_ids.long().to(device))
            y = label.long().to(device, dtype = torch.float)
            loss = loss_fn(pred_y, y)
            loss.backward()
            optimizer.step()

            train_loss = train_loss + ((1 / (batch_id + 1)) * (loss.item() - train_loss))
            if batch_id % (len(train_loader) // 10) == 0: print(batch_id, len(train_loader))

        print("epoch {} train loss {}".format(epoch, train_loss/len(train_loader)))

        model.eval()
        with torch.no_grad():
            for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(val_loader):
                pred_y = model(token_ids.long().to(device), valid_length, segment_ids.long().to(device))
                y = label.long().to(device, dtype = torch.float)

                loss = loss_fn(pred_y, y)
                val_loss = val_loss + ((1 / (batch_id + 1)) * (loss.item() - val_loss))

        print("epoch {} val loss {}".format(epoch, val_loss/len(val_loader)))

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model

    return best_model


def test(test_x: list, test_y: list, test_loader, model):
    test_y, pred_y=[], []

    model.eval()
    with torch.no_grad():
        for token_ids, valid_length, segment_ids, label in test_loader:
            output = model(token_ids.long().to(device), valid_length, segment_ids.long().to(device))

            test_y.extend(label.cpu().detach().numpy().tolist())
            pred_y.extend(torch.sigmoid(output).cpu().detach().numpy().tolist())

    pred_y = (np.array(pred_y) > 0.5).astype(int)
    print(classification_report(test_y, pred_y, target_names=["여성/가족", "남성", "성소수자", "인종/국적", "연령", "지역", "종교", "기타 혐오", "악플/욕설", "clean", "개인지칭"]))


if __name__ == "__main__":
    # Setting Directory
    train_dir = "../../data/train_multi_label.csv"
    test_dir = "../../data/test_multi_label.csv"

    model_dir = "model_kobert.pt"
    tokenizer_dir = "tokenizer_kobert.pickle"
    label_enc_dir = "label_enc_kobert.pickle"

    # Setting Parameter
    device = torch.device("cuda")

    # Hyper Parameter
    epoch = 2
    max_len = 10
    batch_size = 128
    max_grad_norm = 1
    warmup_ratio = 0.1
    log_interval = 200
    lr = 5e-5
    dr_rate = 0.5

    # Flow
    print("1. load data")
    train_x, train_y, val_x, val_y, test_x, test_y = load_data(train_dir, test_dir)

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
    model = BERTClassifier(kobert_model, num_classes=len(train_y[0]), dr_rate=dr_rate).to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    print("3. train")
    model = train(model, train_loader, val_loader, epoch, optimizer, loss_fn)

    print("4. test")
    test(test_x, test_y, test_loader, model)