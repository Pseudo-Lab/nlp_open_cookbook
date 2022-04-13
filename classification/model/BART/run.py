from transformers import BartModel, PreTrainedTokenizerFast
import torch
import sys

from dataloader import SeqClassificationDataset
from trainer import Trainer
from classifier import BartClassifier
from config import parse_args
from eval import Evaluate


if __name__ == "__main__":

    # define args
    # args = parse_args('run.py --task binary --do_eval --from_checkpoint ckpt/00000280iter_trainer.ckpt --device 3'.split())
    # args = parse_args(
    #     "run.py --task binary --device 3 --batch_size 128 --lr 1e-5 --valid_every 20 --checkpoint_dir ckpt-binary".split()
    # )
    args = parse_args(
        "run.py --task binary --device 0 --batch_size 256 --lr 1e-5 --valid_every 10 --checkpoint_dir ckpt-binary-adamw".split()
    )
    # args = parse_args(sys.argv)

    # argument validation
    assert args.task in ("binary", "multi"), "task should be either 'binary' or 'multi'"

    # load tokenizer and data
    tokenizer = PreTrainedTokenizerFast.from_pretrained("hyunwoongko/kobart")
    train_ds = SeqClassificationDataset(
        args, data_path=f"../../data/train_{args.task}.csv", tokenizer=tokenizer
    )
    test_ds = SeqClassificationDataset(
        args, data_path=f"../../data/test_{args.task}.csv", tokenizer=tokenizer
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
