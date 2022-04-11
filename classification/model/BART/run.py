from transformers import BartModel, PreTrainedTokenizerFast

from dataloader import SeqClassificationDataset
from trainer import Trainer
from classifier import BartClassifier
from config import parse_args


if __name__ == "__main__":

    # define args
    args = parse_args()

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

    model = BartClassifier(args = args, num_labels = len(train_ds.labels))

    # train model
    trainer = Trainer(args, train_ds.get_loader(), test_ds.get_loader(), model)
    trainer.fit()
    trainer.evaluate()