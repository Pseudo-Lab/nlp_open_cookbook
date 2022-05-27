"""
License 정보 입력 (TO-DO)
"""

import torch
from torch.utils.data import TensorDataset
from transformers import ElectraTokenizer
from dataclasses import dataclass
from attrdict import AttrDict
import pandas as pd

import os
import logging
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)

@dataclass
class SingleData:
    """
    단일 데이터 정보를 속성별로 포함합니다
    """
    sid:int
    text:str
    label:int

@dataclass
class SingleFeature:
    """
    단일 데이터에 대한 encoding output 을 포함합니다
    """
    input_ids: torch.tensor
    attention_mask: torch.tensor
    token_type_ids: torch.tensor
    label: int

class HFPreprocessor(object):
    def __init__(self, args: AttrDict):
        self.args = args
        self.label_list = None
        self.lb2int = None

    def generate_file_path(self, args: AttrDict, name: str, mode: str) -> str:
        """

        부가정보를 기본 파일명에 추가하여 저장할 파일 경로를 생성합니다

        Args:
            args (AttrDict): argument 정보
            name (str): 기본 파일명
            mode (str): train, validation, test 중 하나

        Returns:
            str: 저장할 파일 경로

        """
        path = os.path.join(
        args.data_dir,
        name.format(
            str(args.task), mode, list(filter(None, args.model_name_or_path.split("/"))).pop(), str(args.max_seq_len)
            ),
        )
        return path

    def _reorganize_data(self, df: pd.DataFrame) -> List[SingleData]:
        """

        데이터의 정보들을 row 별로 속성별로 parsing 하여 SingleData 객체에 담습니다 
        (id, text, label)

        Args:
            df (pd.DataFrame): 데이터프레임 형태의 데이터셋

        Returns:
            List[SingleData]: SingleData 객체들을 포함하는 list

        """
        
        examples = []
        for (i, row) in df.iterrows():
            text, label = row['text'], row['label']
            if i % 100000 == 0:
                logger.info(f'{text} ::: {label}')
            examples.append(SingleData(i, text, label))
        return examples

    def get_label_info(self, df = None) -> Tuple[List[int], Dict[str, int]]:
        """

        unique label list 와 label 별 id 를 맵핑한 dict 를 생성합니다

        Args:
            df (pd.DataFrame, optional): 데이터프레임 형태의 데이터셋. Defaults to None.

        Returns:
            Tuple[List[int], Dict[str, int]]: unique label list 와 label 별로 id 가 맵핑된 dict 반환

        """
        if df is None:
            return self.label_list, self.lb2int

        self.lb2int = {lb:idx for idx, lb in enumerate(df['label'].unique())}
        self.label_list = df['label'].unique().tolist()
        return self.label_list, self.lb2int

    def convert_data_to_features(self, reorg_data:List, tokenizer: ElectraTokenizer, max_length: int) -> List[SingleFeature]:
        """
        각 SingleData 들의 정보를 Tokenizer로 encoding 하여 feature를 생성하고, 이를 다시 취합하여 반환합니다

        Args:
            reorg_data (List): 개별 데이터 정보를 담은 SingleData 객체들을 모두 포함한 list
            tokenizer (ElectraTokenizer): 토크나이징을 수행할 토크나이저 (Huggingface 의 ElectraTokenizer 객체)
            max_length (int): 학습에 사용할 문장 최대 길이 (토큰 개수 기준)

        Returns:
            List[SingleFeature]: encoding output들을 담은 SingleFeature를 모은 List
        """
        labels = [self.lb2int[single_data.label] for single_data in reorg_data]
        
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

    def load_and_cache(self, args: AttrDict, tokenizer: ElectraTokenizer, df: pd.DataFrame, mode: str) -> TensorDataset:
        """
        캐시 데이터가 있을 경우 단순히 불러오고, 없을 경우 생성해서 저장합니다
        cache 활용을 통해, 동일 데이터의 두번째 사용부터 데이터 전처리 과정을 skip할 수 있습니다

        주의할 점은, 동일 데이터를 업데이트 했을 경우, 캐시 데이터를 지우고 실행해야 새로 캐시를 생성할 수 있습니다

        Args:
            args (AttrDict): argument 정보
            tokenizer (ElectraTokenizer): 토크나이징에 활용할 토크나이저
            df (pd.DataFrame): 데이터프레임 형태의 데이터셋
            mode (str): 데이터셋 종류 (train, validation, test)

        Returns:
            TensorDataset: Tensor로 변환된 데이터셋
        """

        # Load data features from cache or dataset file
        feature_name = "cached_{}_{}_{}_{}"
        label_list_name, lb2int_name = feature_name+"_label_list", feature_name+"_lb2int"
        
        cached_features_fname = self.generate_file_path(args, feature_name, mode)
        cached_label_list_fname = self.generate_file_path(args, label_list_name, 'train') # use train labels
        cached_lb2int_fname = self.generate_file_path(args, lb2int_name, 'train')

        if os.path.exists(cached_features_fname):
            logger.info("Loading features from cached file %s", cached_features_fname)
            features = torch.load(cached_features_fname)
            self.label_list, self.lb2int = torch.load(cached_label_list_fname), torch.load(cached_lb2int_fname)
        else:
            logger.info("Creating features from dataset file at %s", args.data_dir)
            reorg_data = self._reorganize_data(df)

            if mode=='train':
                self.label_list, self.lb2int = self.get_label_info(df)
                torch.save(self.label_list, cached_label_list_fname)
                torch.save(self.lb2int, cached_lb2int_fname)

            logger.info("Using label list {} for Classification".format(self.label_list))

            features = self.convert_data_to_features(reorg_data, tokenizer, args.max_seq_len)
            logger.info("Saving features into cached file %s", cached_features_fname)
            torch.save(features, cached_features_fname)
                
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
        return dataset