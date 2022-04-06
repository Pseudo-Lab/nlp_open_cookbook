import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, n_classes, vocab_size, embedding_dim, n_layers=2, hidden_size=64):
        super(SimpleRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(input_size=embedding_dim,
                          hidden_size=self.hidden_size,
                          num_layers=self.n_layers,
                          dropout=0.3,
                          batch_first=True)
        self.output = nn.Linear(self.hidden_size, n_classes)
    
    def forward(self, x):
        #initialize hidden state
        h_0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size)
        
        embed = self.embedding(x) # (batch_size, sequence) -> (batch_size, sequence, embedding_dim)
        out, _ = self.rnn(embed, h_0) 
        out = out[:, -1, :] # (batch_size, embedding_dim, hidden_size)
        out = self.output(out)
        
        return out