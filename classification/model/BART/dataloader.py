import pandas as pd
import math
import torch
import numpy as np
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict

import torch.tensor as T


class SeqClassificationDataset:
    def __init__(self, args, data_path: str, tokenizer):
        self.args = args
        self.device = torch.device(f"cuda:{args.device}" if args.device else "cpu")
        self.df = pd.read_csv(data_path, header=0)
        self.tokenizer = tokenizer
        self._preprocess()

    def _preprocess(self):
        sample_cnt = self.df.shape[0]
        self.df = self.df.dropna(how="any")
        print(f"dropping {sample_cnt - self.df.shape[0]} samples due to nan values...")
        tok_text = self.tokenizer.batch_encode_plus(
            self.df.text.values.tolist(),
            max_length=self.args.max_seq_len,
            truncation=True,
            return_length=True,
        )
        self.labels = self.df.label.unique().tolist()
        self.dataset = [
            (txt, attn_mask, self.labels.index(label), length)
            for txt, attn_mask, label, length in zip(
                tok_text["input_ids"],
                tok_text["attention_mask"],
                self.df.label,
                tok_text["length"],
            )
        ]

    def _collator(self, data_list):
        x = pad_sequence(
            [T(e[0]) for e in data_list],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        mask = pad_sequence(
            [T(e[1]) for e in data_list], batch_first=True, padding_value=0
        )
        y = T([e[2] for e in data_list])
        _len = T([e[3] for e in data_list])
        ret = {"input_ids": x, "attention_mask": mask, "label": y, "length": _len}
        return {k:v.to(self.device) for k,v in ret.items()}

    def get_loader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=self._collator
        )
