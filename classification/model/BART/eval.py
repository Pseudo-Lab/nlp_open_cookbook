import torch
import torch.nn as nn
from typing import NamedTuple, Dict
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score


class Evaluate:
    """best score tracking, hold evaluation args, evaluate"""

    def __init__(self, args: NamedTuple, val_data_loader) -> None:
        self.args = args
        self.device = torch.device(f"cuda:{args.device}" if args.device else "cpu")
        self.val_data_loader = val_data_loader
        self.best_loss = None
        self.loss_fn = nn.CrossEntropyLoss()

    def evaluate(self, model: nn.Module) -> Dict:
        model = model.to(self.device)
        model.eval()
        ret = defaultdict(float)
        ans, pred = [], []
        for batch in self.val_data_loader:
            step_log = self._valid_step(model, batch)
            ret["n"] += step_log["n"]
            ret["valid_loss"] += step_log["valid_batch_loss"] * step_log["n"]
            ans.append(step_log["answer"])
            pred.append(step_log["pred"])
        ret["valid_loss"] /= ret["n"]
        ans = torch.cat(ans)
        pred = torch.cat(pred)
        ret["valid_acc"] = accuracy_score(ans, pred)
        ret["valid_f1"] = f1_score(
            ans, pred, average="macro" if self.args.task == "multi" else "binary"
        )
        if self.best_loss is None or self.best_loss > ret["valid_loss"]:
            self.best_loss = ret["valid_loss"]
        self.cur_loss = ret["valid_loss"]
        return ret

    def _valid_step(self, model, batch) -> Dict:
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            input_length=batch["length"],
        )
        bsz = out.size(0)
        loss = self.loss_fn(out, batch["label"])
        _loss = loss.detach().cpu().item()
        return {
            "n": bsz,
            "valid_batch_loss": _loss,
            "answer": batch["label"].detach().cpu(),
            "pred": out.max(1).indices.cpu(),
        }

    @property
    def is_best(self):
        if hasattr(self, "cur_loss"):
            return self.cur_loss == self.best_loss
        return True
