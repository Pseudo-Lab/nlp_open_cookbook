import argparse
from transformers import BartModel, PreTrainedTokenizerFast

from dataloader import SeqClassificationDataset
from trainer import Trainer
from classifier import BartClassifier


if __name__ == "__main__":

    # define args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_epoch", type=int, default=10, help="number of epochs to train"
    )
    parser.add_argument(
        "--valid_every", type=int, default=20, help="do validation every this step"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=128,
        help="maximum number of subwords in a sequence",
    )
    parser.add_argument("--eps", type=float, default=1e-8, help="eps for Adam")
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.99), help="betas")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument(
        "--min_lr", type=float, default=1e-6, help="minimum learning rate by scheduler"
    )
    parser.add_argument(
        "--patience", type=int, default=20, help="reduceLROnPleateau patience"
    )
    parser.add_argument(
        "--factor", type=float, default=0.6, help="reducing factor by scheduler"
    )
    parser.add_argument("--task", type=str, default="binary", help="binary or multi")

    args = parser.parse_args()

    # argument validation
    assert args.task in ("binary", "multi"), "task should be either 'binary' or 'multi'"

    # load model and tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained("hyunwoongko/kobart")
    

    # load_data
    train_ds = SeqClassificationDataset(
        args, data_path=f"../../data/train_{args.task}.csv", tokenizer=tokenizer
    )
    test_ds = SeqClassificationDataset(
        args, data_path=f"../../data/test_{args.task}.csv", tokenizer=tokenizer
    )

    model = BartClassifier(num_labels = len(train_ds.labels))

    # train model
    trainer = Trainer(args, train_ds.get_loader(), test_ds.get_loader(), model)
    trainer.fit()
