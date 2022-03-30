import mlflow
import pandas as pd
from NB_service import category
from konlpy.tag import Mecab
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from typing import Type


def load_data(train_dir: str, test_dir: str):
    """Returns the sum of two numbers
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
    # 안녕 오늘 날씨가 좋아
    pos_sentences = [" ".join(mecab.nouns(sentence)) for sentence in sentences]
    return pos_sentences


def identity_tokenizer(text: str):
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
    # local_env = "NIPA"

    # MLflow start
    mlflow.set_experiment('category_classification')
    with mlflow.start_run() as run:
        print("1. load data")
        train_x, train_y, test_x, test_y = load_data(train_dir, test_dir)

        # mecab, komoran, kkma 같은 것들을 사용할 수 있게 옵션을 줘야 하나?
        print("2. pos tagging")
        train_x = pos_tagging(train_x)
        test_x = pos_tagging(test_x)

        # tfidf 말고도 countvetor도 고려?
        print("3. text to vector")
        tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)
        train_x = tfidf.fit_transform(train_x)
        test_x = tfidf.transform(test_x)

        print("4. build model")
        model = build_model(train_x, train_y)

        print("5. evaluate")
        f1, precision, recall = evaluate(model, test_x)


        # mlflow.log_param("env", local_env)
        # mlflow.log_param("train", train_dir)
        # mlflow.log_param("train num", len(train_x))
        # mlflow.log_metric("F1", f1)
        # mlflow.log_metric("Precision", precision)
        # mlflow.log_metric("Recall", recall)

        # svc = category()
        # svc.pack('model', clf)
        # svc.pack('tokenizer', tfidf)
        # svc.pack('pos_tagging', pos_tagging)
        # saved_path = svc.save()

        # with open("bentoml_model_dir.txt", "w") as f:
        #     f.write(saved_path)
        # mlflow.log_artifact("result.csv")
        # mlflow.log_artifact("bentoml_model_dir.txt")