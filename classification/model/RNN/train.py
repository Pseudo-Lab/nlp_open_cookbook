import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from konlpy.tag import Okt, Komoran, Kkma
from torchtext.vocab import build_vocab_from_iterator
from typing import Type

import dataset
import rnn


def load_data(train_dir: str, test_dir: str):
    """_summary_

    Args:
        train_dir (str): _description_
        test_dir (str): _description_

    Returns:
        _type_: _description_
    """    
    df_train = pd.read_csv(train_dir)
    df_train = df_train.dropna()
    train_x, train_y = df_train['text'].tolist(), df_train['label'].tolist()
    
    df_test = pd.read_csv(test_dir)
    df_test = df_test.dropna()
    test_x, test_y = df_test['text'].tolist(), df_test['label'].tolist()
    
    return train_x, train_y, test_x, test_y 

def get_tokenizer(name: str='okt'):
    if name == 'okt':
        tokenizer = Okt()
    
    elif name == 'komoran':
        tokenizer = Komoran()
    
    elif name == 'kkma':
        tokenizer = Kkma()
    
    return tokenizer

def yield_token(sentences: str, tokenizer):
    for sentence in sentences:
        yield tokenizer.nouns(sentence)

def set_vocab(iterator):
    vocab = build_vocab_from_iterator(iterator, specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])
    return vocab

def integer_encoding(iterator, vocab):
    return [torch.tensor(vocab(sentence)) for sentence in iterator]

def pad_sequence(sequences, padding_length=32, padding_value=0):
    out_dim = (len(sequences), padding_length)
    out_tensor = sequences[0].data.new(*out_dim).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length] = tensor[:padding_length]
    
    return out_tensor

def build_dataloader(train_x, train_y, test_x, test_y, batch_size):
    train_dataset = dataset.TextDataset(train_x, train_y)
    test_dataset = dataset.TextDataset(test_x, test_y)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    return train_dataloader, test_dataloader
    
    
def build_model(n_classes, vocab_size, embedding_dim, n_layers=2, hidden_size=64):
    """_summary_

    Args:
        train_x (list): _description_
        train_y (list): _description_
    """
    model = rnn.SimpleRNN(n_classes, vocab_size, embedding_dim, n_layers, hidden_size)
    
    return model

def train_model(model, dataloader, n_epochs, lr):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(n_epochs):
        for text, label in dataloader:
            text = text.to(device)
            label = label.to(device)
            
            logit = model(text)
            loss = criterion(logit, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'epoch: {epoch} / {n_epochs}: loss: {loss.detach().item()}')
    
    return model

def predict(model: Type[build_model], dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    y_pred = []
    
    model.eval()
    with torch.no_grad():
        for text, label in dataloader:
            text = text.to(device)
            label = label.to(device)
            output = model(text)
            predicted_label = output.max(1)[1].item()
            y_pred.append(predicted_label)
    
    return y_pred
            
def evaluate(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    
    return f1, precision, recall

if __name__ == "__main__":
    
    print('# Directory setting')
    train_dir = "../../data/train_binary.csv"
    test_dir = "../../data/test_binary.csv"    
    
    print('# load data')
    train_x, train_y, test_x, test_y = load_data(train_dir, test_dir)
    tokenizer = get_tokenizer(name='okt')
    
    print('# build vocabulary')
    vocab = set_vocab(yield_token(train_x, tokenizer))
    
    print('# integer encoding')
    train_x = integer_encoding(yield_token(train_x, tokenizer), vocab)
    test_x = integer_encoding(yield_token(test_x, tokenizer), vocab)
    
    print('# padding')
    train_x = pad_sequence(train_x)
    test_x = pad_sequence(test_x)
    
    print(test_x.shape, len(test_y))
    print('# build dataloader')
    train_dataloader, test_dataloader = build_dataloader(train_x, train_y, test_x, test_y, 8)
    
    print('# train model')
    model = build_model(n_classes=2, vocab_size=len(vocab), embedding_dim=32)
    model = train_model(model, train_dataloader, n_epochs=10, lr=0.1)
    
    print('# evaluate')
    y_pred = predict(model, test_dataloader)
    f1, precision, recall = evaluate(test_y, y_pred)
    print(f1, precision, recall)