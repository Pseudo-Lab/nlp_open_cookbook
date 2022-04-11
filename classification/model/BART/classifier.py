import torch.nn as nn
import torch
from transformers import BartModel
from typing import List
import torch.tensor as T

class BartClassifier(nn.Module):
    def __init__(self, args, num_labels: int):
        super(BartClassifier, self).__init__()
        self.args = args
        self.device = torch.device(f"cuda:{args.device}" if args.device else "cpu")
        self.bart = BartModel.from_pretrained("hyunwoongko/kobart")
        self.ffnn = nn.Linear(768, num_labels)
        
    def forward(self, input_ids, attention_mask, input_length):
        out = self.bart(input_ids=input_ids, attention_mask=attention_mask)
        gathered_output = torch.gather(out.last_hidden_state, dim=1, index=self._generate_gather_input(input_length))  
        gathered_output = gathered_output.squeeze(1)
        logit = self.ffnn(gathered_output)
        return logit

    def _generate_gather_input(self, length_list:List):
        hidden_dim = self.bart.decoder.embed_tokens.embedding_dim
        ret = [[[_len-1]*hidden_dim] for _len in length_list]
        return T(ret).to(self.device)