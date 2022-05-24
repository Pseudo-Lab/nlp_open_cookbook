from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

class Evaluate:
    def __init__(self, val_dataloader):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.val_dataloader = val_dataloader
        self.best_acc = None
        self.criterion = nn.CrossEntropyLoss()
    
    def _val_step(self, model, batch):
        output = model(batch['text'].to(self.device))
        loss = self.criterion(output, batch['label'])
        val_loss = loss.detach().cpu().item()
        return {'valid_batch_loss': val_loss,
                'answer': batch['label'].detach().cpu(),
                'pred': output.max(1).indices.cpu()}
        
    
    @torch.no_grad() 
    def evaluate(self, model):
        model = model.to(self.device)
        model.eval()
        ret = defaultdict(list)
        ans, pred = [], []
        for batch in self.val_dataloader:
            step_log = self._val_step(model, batch)
            ret['val_loss'].append(step_log['valid_batch_loss'])
            ans.append(step_log['answer'])
            pred.append(step_log['pred'])
        ret['val_loss'] = np.mean(ret['val_loss'])
        ans = torch.cat(ans)
        pred = torch.cat(pred)
        ret['val_acc'] = accuracy_score(ans, pred)
        ret['val_f1'] = f1_score(ans, pred)
        
        if self.best_acc is None or self.best_acc < ret['val_loss']:
            self.best_acc = ret['val_acc']
        self.cur_acc = ret['val_acc']
        return ret

    @property
    def is_best(self):
        if hasattr(self, 'cur_acc'):
            return self.cur_acc == self.best_acc
        return True
            