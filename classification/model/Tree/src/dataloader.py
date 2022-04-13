import numpy as np

class DataLoader:
    def __init__(self, data, batch_size, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.reset()

    def reset(self):
        self.indices = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.indices) < self.batch_size:
            self.reset()
        indices = self.indices[:self.batch_size]
        self.indices = self.indices[self.batch_size:]
        return self.data[indices]