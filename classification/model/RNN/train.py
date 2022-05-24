import os
import glob
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn

import dataset
import rnn
from eval import Evaluate

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.trainer_args = self.args['trainer']
        
        self.train_dataloader, self.val_dataloader, self.vocab_size = \
            dataset.create_dataloader(self.args)
        
        self.model = rnn.SimpleRNN(n_classes=self.trainer_args['n_classes'],
                                   vocab_size=self.vocab_size,
                                   embedding_dim=self.trainer_args['embedding_size'],
                                   n_layers=self.trainer_args['layers'],
                                   hidden_size=self.trainer_args['hidden_size']).to(self.device)
        
        self.evaluator = Evaluate(self.val_dataloader)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.trainer_args['lr'])
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            **self.trainer_args['scheduler']
        )
    
    def _train_step(self, batch):
        self.model.train()
        out = self.model(batch['text'].to(self.device))
        loss = self.criterion(out, batch['label'].to(self.device))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'train_batch_loss': loss.detach().cpu().item()}
    
    def save_trainer_state(self, ckpt_code):
        state_dict = {}
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['model'] = self.model.state_dict()
        os.makedirs(self.trainer_args['checkpoint_dir'], exist_ok=True)
        torch.save(state_dict, f'{self.trainer_args["checkpoint_dir"]}/{ckpt_code}trainer.pth')
        
    
    def fit(self):
        val_log = {}
        for epoch in range(1, self.trainer_args['epochs'] + 1):
            total_loss = []
            pbar =  tqdm(self.train_dataloader, desc=f'Ep {epoch}')
            for batch in pbar:
                step_log = self._train_step(batch)
                total_loss.append(step_log['train_batch_loss'])
                loss_print = f'tr_loss: {step_log["train_batch_loss"]:.03f}, val_loss: {val_log.get("val_loss", 0):.03f}, val_acc: {val_log.get("val_acc", 0):.03f}'
                pbar.set_description(f'Ep {epoch}, {loss_print}')
                
            if epoch % self.trainer_args['valid_every'] == 0:
                val_log = self.evaluator.evaluate(self.model)
                self.scheduler.step(val_log['val_loss'])
                if self.evaluator.is_best:
                    print(f'reached best val accuracy of {val_log["val_acc"]}, saving states...')
                    ckpt_code = f'{epoch}iter_'
                    self.save_trainer_state(ckpt_code)
                    for ckpt in glob.glob(f'{self.trainer_args["checkpoint_dir"]}/*iter_trainer.pth'):
                        if ckpt_code not in ckpt:
                            os.remove(ckpt)
        
        ckpt_code = f'{epoch}iter_'
        self.save_trainer_state(ckpt_code)
        return self.model

if __name__  == '__main__':
    import json

    with open('config/binary/config.json', 'r') as f:
        args = json.load(f)
        
    trainer = Trainer(args)
    trainer.fit()