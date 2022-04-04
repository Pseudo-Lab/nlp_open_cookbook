import torch
import torch.nn as nn
from typing import NamedTuple, Dict
from collections import defaultdict


class Evaluate:
    """best score tracking, hold evaluation args, evaluate"""

    def __init__(self, eval_args: NamedTuple, val_data_loader) -> None:
        self.args = eval_args
        self.val_data_loader = val_data_loader
        self.best_loss = None
        self.loss_fn = nn.CrossEntropyLoss()

    def evaluate(self, model: nn.Module) -> Dict:
        model.eval()
        ret = defaultdict(float)
        for batch in self.val_data_loader:
            step_log = self._valid_step(batch, model)
            ret["n"] += step_log["n"]
            ret["valid_loss"] += step_log["valid_batch_loss"] * step_log["n"]
            ret["n_corr"] += step_log["n_corr"]
        ret["valid_loss"] /= ret["n"]
        ret["valid_acc"] = ret["n_corr"] / ret["n"]
        if self.best_loss is None or self.best_loss > ret["valid_loss"]:
            self.best_loss = ret["valid_loss"]
        self.cur_loss = ret["valid_loss"]
        return ret

    def _valid_step(self, model, batch) -> Dict:
        out = model(**batch)
        bsz = out.size(0)
        loss = self.loss_fn(out, batch["label"])
        n_corr = (out.max(1).indices == batch["label"]).sum()
        _loss = loss.detach().cpu().item()
        return {"n": bsz, "valid_batch_loss": _loss, "n_corr": n_corr}

    @property
    def is_best(self):
        return self.cur_loss == self.best_loss
