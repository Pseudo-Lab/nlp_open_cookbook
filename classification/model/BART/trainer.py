import torch
import torch.nn as nn
from typing import Dict
from tqdm import tqdm
import re
import os
import wandb
import glob

from eval import Evaluate


class Trainer:
    def __init__(
        self, args, train_data_loader, val_data_loader, model: torch.nn.Module
    ) -> None:
        self.args = args
        self.device = torch.device(f"cuda:{args.device}" if args.device else "cpu")
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.model = model.to(self.device)

        self.evaluator = Evaluate(args, self.val_data_loader)  # init evaluator
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            betas=self.args.betas,
            eps=self.args.eps,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.args.factor,
            patience=self.args.patience,
            min_lr=self.args.min_lr,
        )
        self.loss_fn = nn.CrossEntropyLoss()

        self.global_step = 0  # initial

    def load_trainer_state(self, trainer_state_path: str) -> None:
        """load optimizer, scheduler, model states"""
        state_dict = torch.load(trainer_state_path)
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.model.load_state_dict(state_dict["model"])
        number = [e for e in re.findall(r'\d+', trainer_state_path) if len(e) == 8][-1]
        self.global_step = int(number) # restore global step number when resuming training  

    def save_trainer_state(self, ckpt_code:str = "") -> None:
        state_dict = {}
        state_dict["optimizer"] = self.optimizer.state_dict()
        state_dict["model"] = self.model.state_dict()
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        torch.save(state_dict, f"{self.args.checkpoint_dir}/{ckpt_code}trainer.ckpt")

    def _train_step(self, batch) -> Dict:
        self.model.train()
        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            input_length=batch["length"],
        )
        loss = self.loss_fn(out, batch["label"])
        loss.backward()
        _loss = loss.detach().cpu().item()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return {"train_batch_loss": _loss}

    def fit(self):
        
        loss_print, val_loss_print,val_acc_print = None, None, None        
        for epoch in range(1, self.args.num_epoch + 1):
            for batch in (pbar := tqdm(self.train_data_loader, desc=f"Ep {epoch}")):
                self.global_step += 1
                step_log = self._train_step(batch)
                loss_print = f"tr_loss={step_log['train_batch_loss'] :.03f}, val_loss={val_loss_print}, val_acc={val_acc_print}"
                pbar.set_description(f"Ep {epoch}, {loss_print}")
                if (self.global_step % self.args.valid_every) == 0:
                    valid_log = self.evaluator.evaluate(self.model)
                    val_loss = step_log["valid_loss"] = valid_log["valid_loss"]
                    val_acc = step_log["valid_acc"] = valid_log["valid_acc"]
                    val_loss_print=f"{val_loss:.03f}"
                    val_acc_print=f"{val_acc:.03f}"
                    self.scheduler.step(val_loss)
                    if self.evaluator.is_best:
                        print(f"reached best val loss of {val_loss_print}, saving states...")
                        ckpt_code = f"{self.global_step:08d}iter_"
                        self.save_trainer_state(ckpt_code=ckpt_code)
                        for ckpt in glob.glob(f"{self.args.checkpoint_dir}/*iter_trainer.ckpt"):
                            if ckpt_code not in ckpt:
                                os.remove(ckpt) # best model이 아닌 모든 이전 ckpt 삭제
        # 마지막 checkpoint 저장
        ckpt_code = f"{self.global_step:08d}iter_"
        self.save_trainer_state(ckpt_code=ckpt_code)
        return self.model # 학습된 모델 반환
    
    def evaluate(self):
        return self.evaluator.evaluate(self.model)
