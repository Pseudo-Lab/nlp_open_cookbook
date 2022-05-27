"""
License 정보 입력 (TO-DO)
"""

import os
import json
import logging
from argparse import ArgumentParser
from datetime import datetime
import numpy as np
import pandas as pd
from attrdict import AttrDict
from fastprogress.fastprogress import master_bar, progress_bar
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import ElectraConfig, ElectraTokenizer, ElectraForSequenceClassification, AdamW, get_linear_schedule_with_warmup

from preprocess import HFPreprocessor
from callbacks import EarlyStopping
from utils import init_logger, set_seed, compute_metrics, load_data, save_results

from typing import Union, List

logger = logging.getLogger(__name__)


class Trainer4Expert:
    def __init__(self, 
                args: AttrDict, 
                model: ElectraForSequenceClassification, 
                train_dataset: TensorDataset, 
                eval_dataset: TensorDataset,
                metrics: Union[List, str],
                ) -> None:
        """

        KoELECTRA Trainer 객체를 생성합니다

        Args:
            args (AttrDict): 학습을 위한 여러 config 정보를 포함합니다
            model (ElectraForSequenceClassification): huggingface의 분류 fine-tuning 을 지원하는 모델 객체 입니다
            train_dataset (TensorDataset): 학습에 사용할 훈련 셋입니다
            metrics (Union[List, str]): 검증에서 평가점수 에 사용할 metric 입니다. 한개일 때는 String, 여러개일 때는 Sequential 객체 입니다

            다음과 같은 metric 이 사용 가능합니다

            - 'acc' : Accuracy
            - 'f1' : F1-score, (Macro, Micro, Weighted f1 모두 계산)

        """

        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.metrics = metrics
        self.args.evaluate_test_during_training = True if self.eval_dataset else False
        self.fitted = False

    def fit(self) -> None:
        """

        fit() 에서는 self.train_dataset 에 해당하는 훈련 데이터로 모델 학습을 수행합니다

        fit()은 다음 기능을 포함합니다

        - Shuffle 을 위한 RandomSampler 정의
        - Dataloader 정의
        - Optimizer 정의
        - Scheduler 정의
            - get_linear_schedule_with_warmup
            - EarlyStopping
        - 저장된 Optimizer, scheduler load
        - 모델 학습 수행
            - gradient accumulation 수행
            - Epoch 마다 검증 수행
            - clip_grad_norm 수행
            - Epoch 마다 검증 수행
            - checkpoint 저장 : 최대 성능 갱신 시
            - Epoch 기준으로 Earlystoping 수행

        """
        self.fitted = True

        # [Shuffle 을 위한 RandomSampler 정의]
        train_sampler = RandomSampler(self.train_dataset)

        # [Dataloader 정의]
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size, num_workers=4, pin_memory=True)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # [Optimizer 정의]
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)

        # [Scheduler 정의] : get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * self.args.warmup_proportion), num_training_steps=t_total)

        # [Scheduler 정의] : EarlyStopping
        early_stopping = EarlyStopping(self.args.es_patience, self.args.es_min_delta)

        # [저장된 Optimizer, scheduler load] : 학습 중단 후 재개 시 사용
        if os.path.isfile(os.path.join(self.args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(self.args.model_name_or_path, "scheduler.pt")
        ):
            optimizer.load_state_dict(torch.load(os.path.join(self.args.model_name_or_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(self.args.model_name_or_path, "scheduler.pt")))

        # [Logging]
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0
        best_val_metric = (0 if 'loss' not in self.args.es_metric else np.inf)

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

                # (gradient accumulation 수행)
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()

                # (clip_grad_norm 수행)
                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    len(train_dataloader) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(train_dataloader)
                ):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

            # (Epoch 마다 검증 수행)
            if self.args.logging_epochs > 0 and epoch % self.args.logging_epochs == 0:
                if self.args.evaluate_test_during_training:
                    results = self.evaluate()
                
                es_val = results[self.args.es_metric] * (-1 if 'loss' not in self.args.es_metric else 1)
                
                if es_val < best_val_metric:
                    best_val_metric = es_val
                    logger.info('Saving best models...')

                    # (checkpoint 저장) : 최대 성능 갱신 시
                    output_dir = os.path.join(self.args.output_dir, f"ckpt-ep{epoch}-gs{global_step}-scr{results[self.args.es_metric]:.4f}-vls{results['val_loss']:.4f}")
                    print('output_dir:', output_dir)

                    if not os.path.exists(output_dir):
                        print('output_dir made:', output_dir)
                        os.makedirs(output_dir)

                    model_to_save = (
                        self.model.module if hasattr(self.model, "module") else self.model
                    )
                    model_to_save.save_pretrained(output_dir)

                    torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to {}".format(output_dir))

                    # (optimizer 저장) : 최대 성능 갱신 시
                    if self.args.save_optimizer:
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to {}".format(output_dir))

                if self.args.max_steps > 0 and global_step > self.args.max_steps:
                    break

            # (Epoch 기준으로 Earlystoping 수행)
            if self.args.earlystopping:
                early_stopping(es_val)
                if early_stopping.early_stop:
                    break

            mb.write("Epoch {} done".format(epoch + 1))

        logger.info(f"epoch = {epoch}, global_step = {global_step}, average loss = {tr_loss / global_step}")

        return 
    
    def evaluate(self, save_scores:bool = False) -> None:
        """

        evaluate() 에서는 self.eval_dataset 에 해당하는 검증 데이터로 
        모델 검증을 수행하고 평가 점수를 계산합니다

        evaluate()은 다음 기능을 포함합니다

        - 순차적 sampling 을 위한 SequentialSampler 정의 
        - DataLoader 정의
        - 모델 평가 수행
            - 예측 outputs 저장
        - 예측 labels 생성
        - Validation Score 계산
        - Validation Score 결과를 파일로 저장

        Args:
            save_scores (bool, optional): 평가 점수를 기록한 파일 저장 여부. Defaults to False.

        Raises:
            Exception: 모델 학습 이전에 호출 시 예외가 발생합니다

        Returns:
            Dict[str, float]: metric명과 score를 기록한 결과를 dict 형태로 반환합니다

        """
        if not self.fitted:
            raise Exception(" Model should be fitted first !")
        
        # [순차적 sampling 을 위한 SequentialSampler 정의] 
        eval_sampler = SequentialSampler(self.eval_dataset)

        # [DataLoader 정의]
        eval_dataloader = DataLoader(self.eval_dataset, 
                                    sampler=eval_sampler, 
                                    batch_size=self.args.eval_batch_size, 
                                    num_workers=4, pin_memory=True)

        # [Logging]
        logger.info(f"***** Running evaluation on eval dataset *****")
        logger.info(f"  Num examples = {len(self.eval_dataset)}")
        logger.info(f"  Eval Batch size = {self.args.eval_batch_size}")

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        results = {}

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

        # [Validation Score 계산]
        if hasattr(self.metrics, "__iter__"):
            for metric in self.metrics:
                score_dict = compute_metrics(metric, out_label_ids, preds)
                results.update(score_dict)
                print(f"***** Evaluation set {metric} : {list(score_dict.values())[0]:.4f} *****")
        else:
            score_dict = compute_metrics(self.metrics, out_label_ids, preds)
            results.update(score_dict)
            print(f"***** Evaluation set {self.metrics} : {list(score_dict.values())[0]:.4f} *****")

        results['val_loss'] = eval_loss

        # [Validation Score 결과를 파일로 저장]
        if self.args.save_results and save_scores:
            results_dir = self.args.save_results
            if not os.path.exists(results_dir):
                print('output_dir made:', results_dir)
                os.makedirs(results_dir)

            now = datetime.now().strftime('%y%m%d%H%M')
            output_eval_trainset_file_path = os.path.join(self.args.save_results, f"{self.args.task}_SD{self.args.seed}_evaluation_scores_{now}.txt")
            save_results(results, output_eval_trainset_file_path)

        return results

    def predict(self, test_dataset: TensorDataset) -> pd.DataFrame:
        """

        predict() 에서는 test_dataset 에 해당하는 테스트 데이터로 
        모델 추론을 수행하여 예측결과를 반환합니다

        predict()은 다음 기능을 포함합니다

        - 순차적 sampling 을 위한 SequentialSampler 정의 
        - DataLoader 정의
        - 모델 추론 수행
            - 예측 outputs 저장
        - 예측 labels 생성
        - 예측 score 생성
        - Prediction Dataframe 생성

        Args:
            test_dataset (TensorDataset): 예측을 수행할 테스트 데이터셋

        Raises:
            Exception: 모델 학습 이전에 호출 시 예외가 발생합니다

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

        # [예측 labels 생성]
        preds = np.argmax(preds, axis=1)
        
        # [예측 score 생성]
        all_preds=preds.tolist()    
        all_probs=torch.max(F.softmax(torch.tensor(preds, dtype=torch.float32), dim=-1),
                            dim=-1)[0].numpy().tolist()
        
        # [Prediction Dataframe 생성]
        result_df = pd.DataFrame()
        result_df[f'prediction'] = all_preds
        result_df[f'score'] = all_probs

        return result_df

def main(cli_args:ArgumentParser):
    """

    KoELECTRA 로 학습, 평가, 추론을 수행하는 main 함수 입니다

    다음과 같은 과정으로 이루어집니다

    - 학습을 위한 Arguments 준비
    - logging 준비
    - SEED 고정
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

    # [SEED 고정]
    set_seed(args)

    now = datetime.now().strftime('%y%m%d%H%M') # 연월일시분
    under_ckpt_dir = f"{args.task}_SD{args.seed}-{args.output_dir}-{now}"
    args.output_dir = os.path.join(args.ckpt_dir, under_ckpt_dir)

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

    # [Model 학습 수행]
    trainer = Trainer4Expert(
                    args=args,
                    model = model,
                    train_dataset=train_dataset, 
                    eval_dataset=test_dataset,
                    metrics = ['acc', 'f1']
                    )

    trainer.fit()

    # [Model 평가 수행]
    results = trainer.evaluate(save_scores=True)

    # [Model 추론(예측) 수행]
    predictions = trainer.predict(test_dataset)

    print("Model Evaluation Scores :", results)
    print("Model Test Predictions :")
    print(predictions.head())

if __name__ == '__main__':
    cli_parser = ArgumentParser()
    cli_parser.add_argument('--task', type=str, choices = ['nsmc', 'ynat'])
    cli_parser.add_argument("--config_dir", type=str, default="config")
    cli_parser.add_argument("--config_file", type=str, default = 'koelectra-base-v3.json')
    cli_args = cli_parser.parse_args()

    main(cli_args)