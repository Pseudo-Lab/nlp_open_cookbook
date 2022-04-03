import pandas as pd 
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from torch import tensor as T

class SeqClassificationDataset:
	def __init__(self, args, data_path:str, tokenizer):
		self.args = args
		self.df = pd.read_csv(data_path, header=0)
		self.tokenizer = tokenizer 
		self._preprocess()

	def _preprocess(self):
		tok_text = self.tokenizer.batch_encode_plus(self.df.text.values)
		self.labels = self.df.label.unique().tolist()
		self.pt_dataset = Dataset([(txt, attn_mask, self.labels.index(label)) for txt, attn_mask, label in zip(tok_text["input_ids"], tok_text["attention_mask"], self.df.label)])
		
	def _collator(self, data_list):
		x = pad_sequence([T(e[0]) for e in data_list], batch_first=True, padding_value = self.tokenizer.pad_token_id)
		mask = pad_sequence([T(e[1]) for e in data_list], batch_first=True, padding_value = 0)
		y = T([e[2] for e in data_list]).unsqueeze(1)

		return {"input_ids": x, "attention_mask": mask, "label": y}

	def get_loader(self):
		return DataLoader(self.pt_dataset, batch_size = self.args.batch_size, shuffle = True, drop_last=False, collate_fn = self._collator)



