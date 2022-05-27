class EarlyStopping():
    """
    모델 학습 시 Loss 가 더 이상 감소하지 않을 때, 과적합을 막기 위해 학습을 중단합니다
    """
    def __init__(self, patience:int = 5, min_delta:int = 0):
        """

        Args:
            patience (int, optional): loss 가 더 감소하지 않을 경우, 학습을 중단하기까지 기다릴 epoch 수 . Defaults to 5.
            min_delta (int, optional): loss 감소로 판단할 최소 변화량. Defaults to 0.
            
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True