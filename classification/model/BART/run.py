import argparse 
import torch 
from dataloader import SeqClassificationDataset
from transformers import BartModel, PreTrainedTokenizerFast 

if __name__ == "__main__":

	# define args
	parser = argparse.ArgumentParser()
	parser.add_argument("num_epoch",type="int", help="number of epochs to train")
	parser.add_argument("valid_every",type="int", help="do validation every this step")
	parser.add_argument("batch_size",type="int", help="batch size")
	parser.add_argument("eps", type="float", help = "eps for Adam")
	parser.add_argument("betas", type="tuple", help = "betas")
	parser.add_argument("lr", type="float", help = "learning rate")
	parser.add_argument("min_lr", type="float", help = "minimum learning rate by scheduler")
	parser.add_argument("patience", type="int", help = "reduceLROnPleateau patience")
	parser.add_argument("factor", type="float", help = "reducing factor by scheduler")
	parser.add_argument("task", type="str", help = "binary or multi")

	args = parser.parse_args()

	# argument validation
	assert args.task in ("binary","multi"), "task should be either 'binary' or 'multi'"

	# load model and tokenizer
	tokenizer = PreTrainedTokenizerFast.from_pretrained("hyunwoongko/kobart")
	model = BartModel.from_pretrained("hyunwoongko/kobart")

	# load_data
	train_ds = SeqClassificationDataset(args, data_path = f"../../data/train_{args.task}.csv", tokenizer = tokenizer)
	test_ds = SeqClassificationDataset(args, data_path = f"../../data/train_{args.task}.csv", tokenizer = tokenizer)