from transformers import BartModel, PreTrainedTokenizerFast
import torch
import sys

from dataloader import SeqClassificationDataset
from trainer import Trainer
from classifier import BartClassifier
# from config import parse_args
from eval import Evaluate

import json
from utils import dotdict, DATASET_MAP

if __name__ == "__main__":

    # load config
    with open(f"config/{sys.argv[1]}/config.json",'rt') as f:
        args = json.load(f)
    args = dotdict(args)

    # load tokenizer and data
    train_path = DATASET_MAP[args.task].format("train") 
    test_path = DATASET_MAP[args.task].format("test")
    tokenizer = PreTrainedTokenizerFast.from_pretrained("hyunwoongko/kobart")
    train_ds = SeqClassificationDataset(
        args, data_path=f"../../data/{train_path}", tokenizer=tokenizer
    )
    test_ds = SeqClassificationDataset(
        args, data_path=f"../../data/{test_path}", tokenizer=tokenizer
    )

    model = BartClassifier(args=args, num_labels=len(train_ds.labels), hidden_dim=768)
    if args.from_checkpoint:
        print(f"loading from {args.from_checkpoint}")
        states = torch.load(args.from_checkpoint)
        model.load_state_dict(states["model"])

    if args.do_eval:
        evaluator = Evaluate(args, test_ds.get_loader())
        result = evaluator.evaluate(model)
        print(
            f"Evaluation | loss={result['valid_loss']:.03f}, acc={result['valid_acc']:.03f}, f1={result['valid_f1']:.03f}"
        )
        sys.exit()

    # train model
    trainer = Trainer(args, train_ds.get_loader(), test_ds.get_loader(), model)
    trainer.fit()
    trainer.evaluate()
