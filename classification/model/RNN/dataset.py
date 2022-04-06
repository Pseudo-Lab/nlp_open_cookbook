from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, sequences, labels):
        self.text = sequences
        self.label = labels
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        return self.text[index], self.label[index]