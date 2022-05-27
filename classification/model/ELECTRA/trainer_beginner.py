"""
License 정보 입력 (TO-DO)
"""

from argparse import ArgumentParser
import json
import os
import numpy as np
import pandas as pd
import logging
from fastprogress.fastprogress import master_bar, progress_bar
from attrdict import AttrDict
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torch.nn.functional as F
from transformers import ElectraConfig, ElectraTokenizer, ElectraForSequenceClassification, AdamW, get_linear_schedule_with_warmup

from utils import init_logger, compute_metrics, load_data
from preprocess import HFPreprocessor

logger = logging.getLogger(__name__)


class Trainer4Beginner:
    """

    [Trainer4Beginner]
    
    Language Model 사용이 익숙하지 않은 beginner 분들을 대상으로, 
    KoELECTRA 모델의 학습, 평가, 예측 기능을 단순하게 구현한 Trainer Class 입니다

    Trainer4Beginner Class 는 총 3가지의 메소드를 가집니다

    - fit 
    - evaluate
    - predict 

    메소드들에 대한 자세한 기능은 각 메소드의 주석에서 확인 가능합니다

    """
    def __init__(self, 
                args:AttrDict, 
                model:ElectraForSequenceClassification, 
                train_dataset:TensorDataset, 
                eval_dataset:TensorDataset) -> None:
        """

        KoELECTRA Trainer 객체를 생성합니다

        Args:
            args (AttrDict): 학습을 위한 여러 config 정보를 포함합니다
            model (ElectraForSequenceClassification): huggingface의 분류 fine-tuning 을 지원하는 모델 객체 입니다
            train_dataset (TensorDataset): 학습에 사용할 훈련 셋입니다
            eval_dataset (TensorDataset): 검증에 사용할 검증 셋입니다

        """
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.fitted = False

    def fit(self) -> None:
        """

        fit() 에서는 self.train_dataset 에 해당하는 훈련 데이터로 모델 학습을 수행합니다

        fit()은 다음 기능을 포함합니다

        - Shuffle 을 위한 RandomSampler 정의
        - DataLoader 정의
        - Optimizer 정의
        - 모델 학습 수행
        
        """
        self.fitted = True
        
        # [Shuffle 을 위한 RandomSampler 정의]
        train_sampler = RandomSampler(self.train_dataset)

        # [Dataloader 정의]
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size, num_workers=4, pin_memory=True)

        t_total = len(train_dataloader) * self.args.num_train_epochs

        # [Optimizer 정의]
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)

        # [Logging]
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)

        global_step = 0
        tr_loss = 0.0

        mb = master_bar(range(int(self.args.num_train_epochs)))

        # [모델 학습 수행]
        for epoch in mb:
            epoch_iterator = progress_bar(train_dataloader, parent=mb)
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.args.device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3]
                }
                
                outputs = self.model(**inputs)
                loss = outputs[0]
                loss.backward()
                tr_loss += loss.item()

                optimizer.step()
                self.model.zero_grad()
                global_step += 1

            mb.write("Epoch {} done".format(epoch + 1))

        logger.info(f"epoch = {epoch}, global_step = {global_step}, average loss = {tr_loss / global_step}")

        return 
    
    def evaluate(self) -> None:
        """

        evaluate() 에서는 self.eval_dataset 에 해당하는 검증 데이터로 
        모델 검증을 수행하고 평가 점수(정확도)를 계산합니다

        evaluate()은 다음 기능을 포함합니다

        - 순차적 sampling 을 위한 SequentialSampler 정의
        - DataLoader 정의
        - 모델 평가 수행
        - 예측 outputs 저장
        - 예측 labels 생성
        - Validation Accuracy 계산

        Raises:
            Exception: 모델 학습 이전에 호출 시 예외가 발생합니다
        
        """
        if not self.fitted:
            raise Exception(" Model should be fitted first ! ")

        # [순차적 sampling 을 위한 SequentialSampler 정의] 
        eval_sampler = SequentialSampler(self.eval_dataset)

        # [DataLoader 정의]
        eval_dataloader = DataLoader(self.eval_dataset, 
                                    sampler=eval_sampler, 
                                    batch_size=self.args.eval_batch_size, 
                                    num_workers=4, pin_memory=True)

        # [Logging]
        logger.info(f"***** Running evaluation on eval dataset *****")
        logger.info(f"  Num examples = {len(self.test_dataset)}")
        logger.info(f"  Eval Batch size = {self.args.eval_batch_size}")

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        # [모델 평가 수행]
        for batch in progress_bar(eval_dataloader):
            self.model.eval()
            batch = tuple(t.to(self.args.device) for t in batch)
            
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids" : batch[2],
                    "labels": batch[3]
                }
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            
            # (예측 outputs 저장)
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps

        # [예측 labels 생성]
        preds = np.argmax(preds, axis=1)

        # [Validation Accuracy 계산]
        acc = compute_metrics('acc', out_label_ids, preds)
        
        print(f"***** Evaluation set Accuracy : {acc:.4f} *****")
    
    def predict(self, test_dataset: TensorDataset) -> pd.DataFrame:
        """

        predict() 에서는 test_dataset 에 해당하는 테스트 데이터로 
        모델 추론을 수행하여 예측결과를 반환합니다

        predict()은 다음 기능을 포함합니다

        - 순차적 sampling 을 위한 SequentialSampler 정의
        - DataLoader 정의
        - 모델 추론 수행
        - 예측 outputs 저장
        - 예측 score 생성
        - 예측 labels 생성
        - Validation Accuracy 계산
        
        Args:
            test_dataset (TensorDataset): 예측을 수행할 테스트 데이터셋

        Raises:
            Exception: _description_

        Returns:
            pd.DataFrame: 
                데이터프레임 형태의 예측결과를 반환합니다. 
                예측 label 인 "prediction" 과 
                예측label 에 대한 probability 인 
                "score" 컬럼을 포함합니다.

        """
        if not self.fitted:
            raise Exception(" Model should be fitted first ! ")

        test_dataset = test_dataset if test_dataset else self.test_dataset

        # [순차적 sampling 을 위한 SequentialSampler 정의]
        test_sampler = SequentialSampler(test_dataset)

        # [Dataloader 정의]
        test_dataloader = DataLoader(test_dataset, 
                                    sampler=test_sampler, 
                                    batch_size=self.args.eval_batch_size, 
                                    num_workers=4, pin_memory=True)

        # [Logging]
        logger.info(f"***** Running evaluation on eval dataset *****")
        logger.info(f"  Num examples = {len(test_dataset)}")
        logger.info(f"  Eval Batch size = {self.args.eval_batch_size}")

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        # [모델 추론 수행]
        for batch in progress_bar(test_dataloader):
            self.model.eval()
            batch = tuple(t.to(self.args.device) for t in batch)
            
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids" : batch[2],
                    "labels": batch[3]
                }
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            
            # (예측 outputs 저장)
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
        
        
        eval_loss = eval_loss / nb_eval_steps

        # [예측 score 생성]
        all_probs=torch.max(
                            F.softmax(
                                    torch.tensor(preds, dtype=torch.float32), dim=-1
                                     ), dim=-1
                            )[0].numpy().tolist()
        
        # [예측 labels 생성]
        preds = np.argmax(preds, axis=1)
        all_preds=preds.tolist()    
        
        # [Prediction Dataframe 생성]
        result_df = pd.DataFrame()
        result_df[f'prediction'] = all_preds
        result_df[f'score'] = all_probs

        # [Validation Accuracy 계산]
        acc = compute_metrics('acc', out_label_ids, preds)

        print(f"***** Test set Accuracy : {acc:.4f} *****")

        return result_df

def main(cli_args:ArgumentParser):
    """

    KoELECTRA 로 학습, 평가, 추론을 수행하는 main 함수 입니다

    다음과 같은 과정으로 이루어집니다

    - 학습을 위한 Arguments 준비
    - logging 준비
    - Dataset 불러오기
    - Tokenizer 불러오기
    - Dataset 전처리
    - Model 준비
    - Model 학습 수행
    - Model 평가 수행
    - Model 추론(예측) 수행

    Args:
        cli_args (ArgumentParser): task와 config의 정보를 가집니다

    """

    # [학습을 위한 Arguments 준비]
    with open(os.path.join(os.path.dirname(__file__), cli_args.config_dir, cli_args.task, cli_args.config_file)) as f:
        args = AttrDict(json.load(f))   
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"


    # [logging 준비]
    init_logger()
    logger.info("Training/evaluation parameters {}".format(args))

    # [Dataset 불러오기]
    train_df, test_df = load_data(args)

    # [Tokenizer 불러오기]
    tokenizer = ElectraTokenizer.from_pretrained(
                                                args.model_name_or_path,
                                                do_lower_case=args.do_lower_case
                                                )

    # [Dataset 전처리]
    processor = HFPreprocessor(args)
    train_dataset = processor.load_and_cache(args, tokenizer, train_df, "train")
    test_dataset = processor.load_and_cache(args, tokenizer, test_df, "test")
    label_list, lb2int = processor.get_label_info()


    # [Model 준비]
    config = ElectraConfig.from_pretrained(
        args.model_name_or_path,
        num_labels = len(label_list),
        id2label = {label_id: str(label) for label, label_id in lb2int.items()},
        label2id = {str(label): label_id for label, label_id in lb2int.items()},
    )

    model = ElectraForSequenceClassification.from_pretrained(
                        args.model_name_or_path,
                        config=config
                    )
    model.to(args.device)


    # [모델 학습 수행]
    trainer = Trainer4Beginner(
                    args=args,
                    model = model,
                    train_dataset=train_dataset, 
                    eval_dataset=test_dataset,
                    )

    trainer.fit()


    # [모델 평가 수행]
    trainer.evaluate()


    # [모델 추론(예측) 수행]
    predictions = trainer.predict(test_dataset)
    print(predictions.head())


if __name__ == '__main__':
    cli_parser = ArgumentParser()
    cli_parser.add_argument('--task', type=str, choices = ['nsmc', 'ynat'])
    cli_parser.add_argument("--config_dir", type=str, default="config")
    cli_parser.add_argument("--config_file", type=str, default = 'koelectra-base-v3.json')
    cli_args = cli_parser.parse_args()

    main(cli_args)
