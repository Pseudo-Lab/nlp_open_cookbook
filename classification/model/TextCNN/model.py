import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, n_classes, d_prob):
        super(TextCNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.n_filters = n_filters
        self.n_classes = n_classes
        self.d_prob = d_prob
        
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )
        self.embedding.weight.data.requires_grad = False # pretrained embedding을 사용할 예정이므로
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim, 
                                              out_channels=n_filters, 
                                              kernel_size=(fs, embedding_dim), 
                                              stride=1) for fs in filter_sizes])
        self.dropout = nn.Dropout(d_prob)
        self.fc = nn.Linear(len(filter_sizes) * n_filters, n_classes)

    
    def forward(self, x):
        batch_size, sequence_length = x.shape
        x = self.embedding(x.T).transpose(1, 2)

        x = [F.relu(conv(x)) for conv in self.convs]
        x = [F.max_pool1d(c, c.size(-1)).squeeze(dim=-1) for c in x]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        out = self.fc(x)

        return out
        

