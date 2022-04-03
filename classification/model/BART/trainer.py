import torch 
import torch.nn as nn
from typing import NamedTuple, Dict
from tqdm import tqdm

from eval import Evaluate

class Trainer:
    def __init__(self, args , train_data_loader, val_data_loader, model : torch.nn.Module) -> None:
        self.args = args
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.model = model

        self.evaluator = Evaluate(args, self.val_data_loader) # init evaluator
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, betas=self.args.betas, eps=self.args.eps)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',factor=self.args.factor, patience=self.args.patience, min_lr=self.args.min_lr)
        self.loss_fn = nn.CrossEntropyLoss() 

        self.global_step = 0 # initial

    def load_trainer_state(self, trainer_state_path:str) -> None:
        '''load optimizer, scheduler, model states'''
        raise NotImplementedError()

    def save_trainer_state(self) -> None:
        raise NotImplementedError()

    def _train_step(self, batch) -> Dict:
        out = self.model(**batch)
        loss = self.loss_fn(out, batch["label"])
        self.optimizer.zero_grad()
        loss.backward()
        _loss = loss.detach().cpu().item()
        self.optimizer.step()
        return {'train_batch_loss' : _loss}

    def fit(self):
        self.global_step += 1
        for epoch in range(1, self.args.num_epoch+1):
            for batch in tqdm(self.train_data_loader, desc=f'Ep {epoch}'):
                step_log = self._train_step(batch)

                if (self.global_step % self.args.valid_every) == 0:
                    valid_log = self.evaluator.evalute(self.model)
                    print(valid_log) # TODO : logger 활용
                    self.scheduler.step(valid_log['valid_loss']) # update scheduler state

                if self.evaluator.is_best:
                    self.save_trainer_state() # save states when valid score is best






