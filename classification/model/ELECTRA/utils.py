import random
import logging

import torch
import numpy as np
import sys

from sklearn import metrics as sklearn_metrics
import os

def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)]
        #stream=sys.stdout
    )

# seed 고정
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASH"] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)  # type: ignore
    torch.cuda.manual_seed_all(args.seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

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