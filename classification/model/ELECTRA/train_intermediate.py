"""
[intermidiate]

- Tokenizer : unused token 에 단어 추가
- model : freeze layers

- SEED 고정 (v)
- Trainer:
    - save checkpoint (v)
    - earlystopping (v)
    - metrics : list(multiple) or str(single) (v)
    - validation - validation 수행 여부 지정 (test_during_evaluation) (v)

============================================================================

- Trainer
    - fit (v)
    - evaluate (v)
    - predict (v)


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


class Trainer4Intermediate:
    def __init__(self, 
                args:AttrDict, 
                model:ElectraForSequenceClassification, 
                train_dataset:TensorDataset, 
                eval_dataset:TensorDataset,
                metrics: Union[List, str],
                ) -> None:

        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.metrics = metrics
        self.fitted = False
        self.args.evaluate_test_during_training = True if self.eval_dataset else False

    def fit(self) -> None:

        self.fitted = True

        # [Train set Dataloader 정의]
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size, num_workers=4, pin_memory=True)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # [Optimizer, Scheduler 정의]
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        early_stopping = EarlyStopping(self.args.es_patience, self.args.es_min_delta)

        # [학습 재개 시 : 저장된 Optimizer, scheduler load]
        if os.path.isfile(os.path.join(self.args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(self.args.model_name_or_path, "scheduler.pt")
        ):
            optimizer.load_state_dict(torch.load(os.path.join(self.args.model_name_or_path, "optimizer.pt")))

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
                loss.backward()
                tr_loss += loss.item()

                optimizer.step()
                self.model.zero_grad()
                global_step += 1
        
            if self.args.logging_epochs > 0 and epoch % self.args.logging_epochs == 0:
                if self.args.evaluate_test_during_training:
                    results = self.evaluate()
                
                es_val = results[self.args.es_metric] * (-1 if 'loss' not in self.args.es_metric else 1)
                
                if es_val < best_val_metric:
                    best_val_metric = es_val
                    logger.info('Saving best models...')

                    # Save model checkpoint
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

                    if self.args.save_optimizer:
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        logger.info("Saving optimizer and scheduler states to {}".format(output_dir))

                if self.args.max_steps > 0 and global_step > self.args.max_steps:
                    break

            if self.args.earlystopping:
                early_stopping(es_val)
                if early_stopping.early_stop:
                    break

            mb.write("Epoch {} done".format(epoch + 1))

        logger.info(f"epoch = {epoch}, global_step = {global_step}, average loss = {tr_loss / global_step}")

        return 
    
    def evaluate(self, save_scores=False) -> None:

        if not self.fitted:
            raise Exception(" Model should be fitted first ! ")
        
        # [Validation set Dataloader 정의] 
        eval_sampler = SequentialSampler(self.eval_dataset)
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
            
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        # [Validation Scores 계산]
        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)

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

        if not self.fitted:
            raise Exception(" Model should be fitted first ! ")

        # [Test set Dataloader 정의]
        test_dataset = test_dataset if test_dataset else self.test_dataset
        test_sampler = SequentialSampler(test_dataset)
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
            
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        # [Prediction Dataframe 생성]
        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)
        
        all_preds=preds.tolist()    
        all_probs=torch.max(F.softmax(torch.tensor(preds, dtype=torch.float32), dim=-1),
                            dim=-1)[0].numpy().tolist()
        
        result_df = pd.DataFrame()
        result_df[f'prediction'] = all_preds
        result_df[f'score'] = all_probs

        return result_df

def main(cli_args:ArgumentParser):

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

    train_df, test_df = load_data(args)

    tokenizer = ElectraTokenizer.from_pretrained(
                                                args.model_name_or_path,
                                                do_lower_case=args.do_lower_case
                                                )


    # [Dataset 준비]
    processor = HFPreprocessor(args)
    train_dataset = processor.load_and_cache(args, tokenizer, train_df, "train")
    test_dataset = processor.load_and_cache(args, tokenizer, test_df, "test")
    label_list, lb2int = processor.get_label_info()


    # [모델 준비]
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
    trainer = Trainer4Intermediate(
                    args=args,
                    model = model,
                    train_dataset=train_dataset, 
                    eval_dataset=test_dataset,
                    metrics = ['acc', 'f1']
                    )

    trainer.fit()

    # [모델 평가 수행]
    results = trainer.evaluate(save_scores=True)

    # [모델 예측 수행]
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