import sys
import mlflow
import pandas as pd

from konlpy.tag import Mecab
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data(train_dir, test_dir):
    df_train = pd.read_csv(train_dir)
    df_test = pd.read_csv(test_dir)

    df_train = df_train.dropna()
    df_test = df_test.dropna()

    train_x, train_y = df_train["text"].tolist(), df_train["label"].tolist()
    test_x, test_y = df_test["text"].tolist(), df_test["label"].tolist()

    return train_x, train_y, test_x, test_y


def pos_tagging(sentences):
    mecab = Mecab()
    pos_sentences = [" ".join(mecab.nouns(sentence)) for sentence in sentences]
    return pos_sentences


def identity_tokenizer(text):
    return text


def text_to_vector():
    return


def build_model(train_x, train_y):
    Cs = [1]
    Kernel = ["linear"]
    param_grid = {'C': Cs, 'kernel': Kernel}
    grid_search = GridSearchCV(
        SVC(C=1, probability=True, cache_size=50000, class_weight='balanced'),
        param_grid=param_grid,
        scoring='accuracy',  
        verbose=2, 
        n_jobs=1
    )
    grid_search.fit(train_x, train_y)

    return grid_search, grid_search.best_params_


def evaluate(pos_test_x, test_y):
    tfidf_test_x = tfidf_vector.transform(pos_test_x)
    pred_y = model.predict(tfidf_test_x)

    df_predict = pd.DataFrame({"sentence": test_x, "pred_y": pred_y, "test_y": test_y})
    df_predict.to_csv(output_dir+train_dir[0:-4]+"_"+test_dir, index=False)

    return f1_score(test_y, pred_y, average='weighted')


if __name__ == "__main__":
    # Setting
    train_dir = "../../data/train_binary.csv"
    test_dir = "../../data/test_binary.csv"

    model_name = "SVM"
    pos_tool = "mecab"

    train_x, train_y, test_x, test_y = load_data(train_dir, test_dir)
    
    pos_train_x = pos_tagging(train_x)
    print(pos_train_x[0:3])
    pos_test_x = pos_tagging(test_x)

    print(identity_tokenizer(pos_train_x[0]))
    tfidf = TfidfVectorizer(tokenizer=None, preprocessor=None, lowercase=False)
    tfidf_train_x = tfidf.fit_transform(pos_train_x)
    print(tfidf_train_x[0])
    tfidf_test_x = tfidf.transform(pos_test_x)

    # model, best_params = build_model(tfidf_train_x, train_y)

    # score_100 = evaluate(data_dir, output_dir, train_dir, test_100_dir, tfidf_vector)