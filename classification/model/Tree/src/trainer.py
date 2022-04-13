class Trainer:

    """
    Class to train a decision tree.
    """

    def __init__(self, model):
        """
        Initialize the trainer.

        :param model: the tree to train
        """
        self.model = model

    def train(self, X, y):
        """
        Train the model.
        """
        self.model.train(X, y)