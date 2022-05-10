from datetime import datetime
import joblib

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src import model_dir


class Trainer:

    """
    Class to train a tree model
    """

    def __init__(self, model='RandomForest'):
        """
        Initialize the trainer.

        :param model: model to train
        """
        if model == 'RandomForest':
            self.model = RandomForestClassifier()
        if model == 'DecisionTree':
            self.model = DecisionTreeClassifier()
        self.model_name = model

    def train(self, X, y):
        """
        Train the model.
        """
        self.model.fit(X, y)

    def evaluate(self, X, y, report=True):
        """
        Calculate metrics.
        """
        pred = self.model.predict(X)
        accuracy = accuracy_score(y, pred)
        if report:
            f1_score_ = f1_score(y, pred, average='macro')
            confusion_matrix_ = confusion_matrix(y, pred)
            self.eval_report = {'accuracy': accuracy, 'f1_score': f1_score_, 'confusion_matrix': confusion_matrix_}
        return accuracy
        
    def save_model(self, path=model_dir):
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        with (path / f'{self.model_name}_{now}.pkl').open('wb') as f:
            joblib.dump(self.model, f)
