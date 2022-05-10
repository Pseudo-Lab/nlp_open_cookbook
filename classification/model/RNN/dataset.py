import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from konlpy.tag import Okt, Komoran, Kkma
from torchtext.vocab import build_vocab_from_iterator

class TextDataset(Dataset):
    def __init__(self, dataset_args, vocab=None, is_train=True):
        self.args = dataset_args
        self.is_train = is_train
        self.vocab = vocab
        
        df = pd.read_csv(self.args['file_path'])
        df = df.dropna()
        self.length = len(df)
        self.text = df['text'].tolist()
        self.label = df['label'].tolist()
        
        self.tokenizer = self.args['tokenizer']
        self.vocab_size = self.args['vocab_size']
        
        self.pre_porcess()
        
    def _get_tokenizer(self):
        if self.tokenizer == 'kkma':
            self.tokenizer = Kkma()
        
        elif self.tokenizer == 'okt':
            self.tokenizer = Okt()
        
        elif self.tokenizer == 'komoran':
            self.tokenizer = Komoran()
    
    def _yield_tokens(self):
        for sentence in self.text:
            yield self.tokenizer.nouns(sentence)
    
    def _set_vocab(self):
        self.vocab = build_vocab_from_iterator(self._yield_tokens(), specials=['<unk>'], max_tokens=self.vocab_size)
        self.vocab.set_default_index(self.vocab['<unk>'])
    
    def _integer_encoding(self):
        self.text = [self.vocab(self.tokenizer.nouns(sentence)) for sentence in self.text]
    
    def pre_porcess(self):
        self._get_tokenizer()
        
        if self.is_train:
            self._set_vocab()
        
        self._integer_encoding()
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        return {'text': torch.tensor(self.text[index], dtype=torch.int64), 
                'label':torch.tensor(self.label[index], dtype=torch.int64)}

def create_dataloader(args):
    train_dataset_args = args['dataset']['train']
    val_dataset_args = args['dataset']['val']
    vocab_size = train_dataset_args['vocab_size']
    
    train_dataset = TextDataset(train_dataset_args)
    val_dataset = TextDataset(val_dataset_args, vocab=train_dataset.vocab, is_train=False)
    
    def collate(samples):
        sequences = [sample['text'] for sample in samples]
        labels = [sample['label'] for sample in samples]
        padded_sequences = pad_sequence(sequences, batch_first=True)
        return {'text': padded_sequences, 'label': torch.tensor(labels)}
    
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=train_dataset_args['batch_size'],
                                  shuffle=train_dataset_args['shuffle'],
                                  collate_fn=collate)

    val_dataloader = DataLoader(val_dataset, 
                                batch_size=val_dataset_args['batch_size'],
                                shuffle=val_dataset_args['shuffle'],
                                collate_fn=collate)
    
    return train_dataloader, val_dataloader, vocab_size

if __name__ == '__main__':
    import json
    with open('config.json', 'r') as f:
        args = json.load(f)
    
    train_dataloader, val_dataloader, vocab_size = create_dataloader(args)
    
    for train, val in zip(train_dataloader, val_dataloader):
        print(train['text'].shape, train['label'])
        print(val['text'].shape, val['label'])
        break
