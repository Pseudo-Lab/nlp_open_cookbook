import torch 
import torch.nn as nn
from typing import NamedTuple, Dict

class Evaluate:
    '''best score tracking, hold evaluation args, evaluate'''
    def __init__(self, eval_args : NamedTuple, val_data_loader) -> None:
        self.args = eval_args 
        self.val_data_loader = val_data_loader 
        self.best_loss = None

    def evaluate(self, model : nn.Module) -> Dict:
        loss = 0
        for batch in self.val_data_loader:
            pass
        self.cur_loss = loss
        if self.best_loss is None or self.best_loss > loss:
            self.best_loss = loss
        return {"valid_loss" : loss}

    @property
    def is_best(self):
        return self.cur_loss == self.best_loss