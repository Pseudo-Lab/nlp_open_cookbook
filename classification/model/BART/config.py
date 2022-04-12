import argparse
from typing import List


def parse_args(argv: List = []):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from_checkpoint",
        type=str,
        default="",
        help="trainer checkpoint path to resume training",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./ckpt-multi",
        help="Checkpoint dir to for saving",
    )
    parser.add_argument(
        "--eval_checkpoint",
        type=str,
        # default="checkpoint-len100-continued/checkpoint-21500/pytorch_model.bin",
        default="",
        help="checkpoint for evaluation",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="problem_data_220316.csv",
        help="dataset file path",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="skt/kogpt-base-v2",
        help="backbone model type",
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="",
        help="project name (if to use) for wandb tracking",
    )
    parser.add_argument(
        "--num_epoch", type=int, default=10, help="number of epochs to train"
    )
    parser.add_argument(
        "--valid_every", type=int, default=40, help="do validation every this step"
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
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
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
    parser.add_argument("--device", type=str, default="3", help="device id")
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="do only evaluation using 'from_checkpoint' argument",
    )
    return parser.parse_args(argv[1:])
