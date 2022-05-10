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
                          batch_first=True)
        
        self.feedforward = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.output = nn.Linear(self.hidden_size // 2, n_classes)
    
    def forward(self, x):
        #initialize hidden state
        h_0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size)
        
        embed = self.embedding(x) # (batch_size, sequence) -> (batch_size, sequence, embedding_dim)
        out, _ = self.rnn(embed, h_0) 
        out = out[:, -1, :] # (batch_size, embedding_dim, hidden_size)
        out = self.feedforward(out)
        out = self.output(out)
        
        return out
    
if __name__ == '__main__':
    import numpy as np
    
    model = SimpleRNN(n_classes=2, vocab_size=10000, embedding_dim=16, n_layers=2, hidden_size=64)
    x = [torch.tensor(np.random.randint(low=0, high=9999, size=(4, 20))) for i in range(100)]
    y = [torch.tensor(np.random.randint(low=0, high=1, size=(4, 1)), dtype=torch.long) for i in range(100)]
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    
    
    for i in range(100):
        for text, label in zip(x, y):
            output = model(text)
            loss = criterion(output, label.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss.detach())