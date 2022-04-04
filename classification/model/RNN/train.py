import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from typing import Type

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
    test_x, test_y = df_test['text'].tolist(), df_train['label'].tolist()
    
    return train_x, train_y, test_x, test_y 

def pos_tagging(setences: str):
    """_summary_

    Args:
        setences (str): _description_
    """    
    pass
    return

def identity_tokenizer(text: str):
    """_summary_

    Args:
        text (str): _description_

    Returns:
        _type_: _description_
    """
    return text.split(" ")

def build_model(train_x: list, train_y: list):
    """_summary_

    Args:
        train_x (list): _description_
        train_y (list): _description_
    """
    pass
    return

def evaluate(model: Type[build_model], test_x: list, test_y: list):
    """_summary_

    Args:
        model (Type[build_model]): _description_
        test_x (list): _description_
        test_y (list): _description_
    """
    pass
    return

if __name__ == "__main__":
    pass