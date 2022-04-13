import torch
import os
from dataclasses import dataclass
from torch.utils.data import TensorDataset
import logging
from typing import List

logger = logging.getLogger(__name__)

@dataclass
class SingleData:
    sid:int
    text:str
    label:int

@dataclass
class SingleFeature:
    input_ids: torch.tensor
    attention_mask: torch.tensor
    token_type_ids: torch.tensor
    label: int

class PreprocessorforHF(object):
    def __init__(self, args):
        self.args = args
        self.label_list = None
        self.lb2int = None

    def _reorganize_data(self, df) -> List[SingleData]:
        """ Parse single DataFrame row to SingleData Type : id, text, label """
        examples = []
        for (i, row) in df.iterrows():
            text, label = row['text'], row['label']
            if i % 100000 == 0:
                logger.info(f'{text} ::: {label}')
            examples.append(SingleData(i, text, label))
        return examples

    def get_label_info(self, df = None):
        """ get label list and label_to_int dictionary """

        if df is None:
            return self.label_list, self.lb2int

        self.lb2int = {lb:idx for idx, lb in enumerate(df['label'].unique())}
        self.label_list = df['label'].unique().tolist()
        return self.label_list, self.lb2int

    def convert_data_to_features(self, reorg_data:List, tokenizer, max_length, df, mode:str) -> List[SingleFeature]:
        if mode != 'train':
            df = None

        label_list, lb2int = self.get_label_info(df)

        logger.info("Using label list {} for Classification".format(label_list))

        labels = [lb2int[single_data.label] for single_data in reorg_data]
        
        batch_encoding = tokenizer.batch_encode_plus(
            [single_data.text for single_data in reorg_data],
            max_length=max_length,
            padding="max_length",
            add_special_tokens=True,
            truncation=True,
        )

        features = []
        for i in range(len(reorg_data)):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}
            if "token_type_ids" not in inputs:
                inputs["token_type_ids"] = [0] * len(inputs["input_ids"])  # For xlm-roberta

            feature = SingleFeature(**inputs, label=labels[i])
            features.append(feature)

        for i, example in enumerate(reorg_data[:1]):
            logger.info("*** Example ***")
            logger.info("guid: {}".format(example.sid))
            logger.info("input_ids: {}".format(" ".join([str(x) for x in features[i].input_ids])))
            logger.info("attention_mask: {}".format(" ".join([str(x) for x in features[i].attention_mask])))
            logger.info("token_type_ids: {}".format(" ".join([str(x) for x in features[i].token_type_ids])))
            logger.info("label: {}".format(features[i].label))

        return features

    def load_and_cache(self, args, tokenizer, df, mode):

        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}".format(
                str(args.task), list(filter(None, args.model_name_or_path.split("/"))).pop(), str(args.max_seq_len)
            ),
        )
        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            logger.info("Creating features from dataset file at %s", args.data_dir)
            reorg_data = self._reorganize_data(df)
            features = self.convert_data_to_features(reorg_data, tokenizer, args.max_seq_len, df, mode)
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
        return dataset