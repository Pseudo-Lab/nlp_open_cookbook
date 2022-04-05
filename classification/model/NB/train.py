import pickle
import pandas as pd
from konlpy.tag import Mecab
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from typing import Type


def load_data(train_dir: str, test_dir: str):
    """train and test data load
    Args:
        train_dir (str): _
        test_dir (str): _

    Returns:
        train_x: _
        train_y: _
        test_x: _
        test_y: _
    """
    df_train = pd.read_csv(train_dir)
    df_train = df_train.dropna()
    train_x, train_y = df_train["text"].tolist(), df_train["label"].tolist()

    df_test = pd.read_csv(test_dir)
    df_test = df_test.dropna()
    test_x, test_y = df_test["text"].tolist(), df_test["label"].tolist()

    return train_x, train_y, test_x, test_y


def pos_tagging(sentences: str):
    mecab = Mecab()
    pos_sentences = [" ".join(mecab.nouns(sentence)) for sentence in sentences]
    return pos_sentences


def custom_tokenizer(text: str):
    return text.split(" ")


def build_model(train_x: list, train_y: list):
    clf = MultinomialNB()
    clf.fit(train_x, train_y)
    return clf


def evaluate(model: Type[build_model], test_x: list, test_y: list):
    pred_y = model.predict(test_x)

    f1 = f1_score(test_y, pred_y, average='weighted')
    precision = precision_score(test_y, pred_y, average='weighted')
    recall = recall_score(test_y, pred_y, average='weighted')
    return f1, precision, recall


if __name__ == "__main__":
    # Directory Setting
    train_dir = "../../data/train_binary.csv"
    test_dir = "../../data/test_binary.csv"
    model_dir = "NB.pickle"
    tokenizer_dir = "tokenizer.pickle"

    # Flow
    print("1. load data")
    train_x, train_y, test_x, test_y = load_data(train_dir, test_dir)

    print("2. pre processing")
    train_x = pos_tagging(train_x)
    test_x = pos_tagging(test_x)

    print("3. text to vector")
    # vectorizer = CountVectorizer(tokenizer=custom_tokenizer, lowercase=False)
    vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, lowercase=False)
    train_x = vectorizer.fit_transform(train_x)
    test_x = vectorizer.transform(test_x)

    print("4. build model")
    model = build_model(train_x, train_y)

    print("5. evaluate")
    f1, precision, recall = evaluate(model, test_x, test_y)
    print("F1 Score :", f1)

    print("6. save file")
    pickle.dump(model, open(model_dir, 'wb'))
    pickle.dump(vectorizer, open(tokenizer_dir, 'wb'))