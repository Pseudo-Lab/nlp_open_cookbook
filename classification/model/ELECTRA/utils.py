import torch
import numpy as np
import pandas as pd
from attrdict import AttrDict
from sklearn import metrics as sklearn_metrics, utils

import os
import sys
import random
import logging
from typing import Tuple, Dict


def init_logger() -> None:
    """
    Logging 시작을 위한 config를 정의합니다
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)]
        #stream=sys.stdout
    )

def set_seed(args) -> None:
    """
    결과 재현을 위한 Seed를 고정합니다 (Pytorch)

    Args:
        args (_type_): _description_
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASH"] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)  # type: ignore
    torch.cuda.manual_seed_all(args.seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

def read_txt(filepath: str) -> pd.DataFrame:
    """
    네이버 영화리뷰 데이터 (NSMC) 를 불러와 데이터프레임 형태로 저장합니다

    Args:
        filepath (str): NSMC 데이터 경로

    Returns:
        pd.DataFrame: "text" 와 "label" 로 이루어진 NSMC 데이터프레임 데이터
    """
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

def load_data(args:AttrDict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    NSMC, KLUE-TC 데이터를 불러와 데이터프레임으로 반환합니다

    커스텀 데이터 사용 시, 따로 해당 함수를 작성하여 사용할 수 있습니다

    Args:
        args (AttrDict): task 정보를 포함합니다

    Raises:
        ValueError: train 코드에서 task argument 값이 nsmc 나 ynat 이 아닐 경우 예외발생

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 데이터프레임 형태의 train, test 데이터
    """
    if args.task == 'nsmc':
        data_paths = {"train": f"../../data/ratings_train.txt",
                      "test": f"../../data/ratings_test.txt",}
                      
        train_df = read_txt(data_paths["train"])
        test_df = read_txt(data_paths["test"])

    elif args.task == 'ynat':
        data_paths = {"train": f"../../data/train_multi_class.csv",
                      "test": f"../../data/test_multi_class.csv",}

        train_df = pd.read_csv(data_paths["train"])
        test_df = pd.read_csv(data_paths["test"])

    else:
        raise ValueError("task should be either 'nsmc' or 'ynat'")

    train_df, test_df = train_df[train_df['text'].notnull()], test_df[test_df['text'].notnull()]

    print('train data sample :', train_df.head(1))
    print('test data sample :', test_df.head(1))

    return train_df, test_df

def simple_accuracy(labels: np.array, preds: np.array) -> float:
    """
    single label 예측 모델의 Accuracy 를 계산합니다

    Args:
        labels (np.array): 정답 label 배열
        preds (np.array): 예측 label 배열

    Returns:
        float: Accuracy 점수 
    """
    return (labels == preds).mean()

def acc_score(labels: np.array, preds: np.array) -> Dict[str, float]:
    """

    Accuracy 를 계산합니다
    여러 task 에서의 Accuracy 계산 함수를 추가할 수 있습니다

    Args:
        labels (np.array): 정답 label 배열
        preds (np.array): 예측 label 배열

    Returns:
        Dict[str, float] : Accuracy 점수를 포함한 dict 객체 반환

    """
    return {
        "acc": simple_accuracy(labels, preds),
    }

def f1_score(labels: np.array, preds: np.array) -> Dict[str, float]:
    """
    f1-score 를 계산합니다

    Args:
        labels (np.array): 정답 label 배열
        preds (np.array): 예측 label 배열

    Returns:
        Dict[str, float]: macro_f1, micro_f1, weighted_f1 을 포함한 dict 객체 반환
    """
    return {
            "macro_f1": sklearn_metrics.f1_score(labels, preds, average='macro'),
            "micro_f1": sklearn_metrics.f1_score(labels, preds, average='micro'),
            "weighted_f1": sklearn_metrics.f1_score(labels, preds, average='weighted'),
        }

def f1_pre_rec(labels: np.array, preds: np.array, is_ner: bool =True) -> Dict[str, float]:
    return {
            "precision": sklearn_metrics.precision_score(labels, preds, average="macro"),
            "recall": sklearn_metrics.recall_score(labels, preds, average="macro"),
            "f1": sklearn_metrics.f1_score(labels, preds, average="macro"),
        }


def compute_metrics(metric: str, labels: np.array, preds: np.array) -> Dict[str, float]:
    """

    평가 시 모델의 평가지표 점수를 계산하여 반환합니다

    Args:
        metric (str): 사용할 평가지표
        labels (np.array): 정답 label 배열
        preds (np.array): 예측 label 배열

    Returns:
        Dict[str, float]: 평가지표와 점수를 포함한 dict 객체 반환

    """
    assert len(preds) == len(labels)
    if metric == "acc":
        return acc_score(labels, preds)
    if metric == "f1":
        return f1_score(labels, preds)

def save_results(results: dict, path: str):
    """
    모델의 평가지표와 점수를 파일로 기록하여 저장합니다

    Args:
        results (dict): metric명과 score를 포함합니다
        path (str): 저장할 경로
    """
    with open(path, "w") as f_w:
        for key in sorted(results.keys()):
            f_w.write("{} = {}\n".format(key, str(results[key])))
