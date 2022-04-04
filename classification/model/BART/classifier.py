import torch.nn as nn
import torch
from transformers import BartModel


class BartClassifier(nn.Module):
    def __init__(self, num_labels: int):
        super(self, BartClassifier).__init__()
        self.bart = BartModel.from_pretrained("hyunwoongko/kobart")
        self.ffnn = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask, input_length):
        out = self.bart(input_ids=input_ids, attention_mask=attention_mask)
        # torch.gather(out.last_hidden_state, dim=1, )
        gathered_output = out.last_hidden_state  # do some gathering using length
        logit = self.ffnn(gathered_output)
        return logit
