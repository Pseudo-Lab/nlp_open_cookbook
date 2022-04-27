import random
import logging
from attrdict import AttrDict

import torch
import numpy as np
import pandas as pd
import sys

from sklearn import metrics as sklearn_metrics, utils
import os


def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)]
        #stream=sys.stdout
    )

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASH"] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)  # type: ignore
    torch.cuda.manual_seed_all(args.seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

def read_txt(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = []
        for line in f:
            lines.append(line.strip())

    texts, labels = [], []
    for (i, line) in enumerate(lines[1:]):
        line = line.split("\t")
        text_a = line[1]
        label = line[2]
        texts.append(text_a)
        labels.append(label)
    return pd.DataFrame({'text': texts, 'label':labels})

def load_data(args:AttrDict):
    if args.task == 'nsmc':
        data_paths = {"train": f"../../data/ratings_train.txt",
                      "test": f"../../data/ratings_test.txt",}
    elif args.task == 'ynat':
        data_paths = {"train": f"../../data/train_multi.csv",
                      "test": f"../../data/test_multi.csv",}

    train_df = read_txt(data_paths["train"])
    test_df = read_txt(data_paths["test"])

    train_df, test_df = train_df[train_df['text'].notnull()], test_df[test_df['text'].notnull()]

    print('train data sample :', train_df.head(1))
    print('test data sample :', test_df.head(1))

    return train_df, test_df

def simple_accuracy(labels, preds):
    return (labels == preds).mean()

def acc_score(labels, preds):
    return {
        "acc": simple_accuracy(labels, preds),
    }

def f1_score(labels, preds):
    return {
            "macro_f1": sklearn_metrics.f1_score(labels, preds, average='macro'),
            "micro_f1": sklearn_metrics.f1_score(labels, preds, average='micro'),
            "weighted_f1": sklearn_metrics.f1_score(labels, preds, average='weighted'),
        }

def f1_pre_rec(labels, preds, is_ner=True):
    return {
            "precision": sklearn_metrics.precision_score(labels, preds, average="macro"),
            "recall": sklearn_metrics.recall_score(labels, preds, average="macro"),
            "f1": sklearn_metrics.f1_score(labels, preds, average="macro"),
        }


def compute_metrics(metric, labels, preds):
    assert len(preds) == len(labels)
    if metric == "acc":
        return acc_score(labels, preds)
    if metric == "f1":
        return f1_score(labels, preds)

def save_results(results, path):
    with open(path, "w") as f_w:
        for key in sorted(results.keys()):
            f_w.write("{} = {}\n".format(key, str(results[key])))

