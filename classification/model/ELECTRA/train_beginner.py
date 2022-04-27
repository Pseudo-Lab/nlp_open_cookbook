"""
[beginner]

- fit : train_set 학습 (self.model)
- evaluate : eval_set 에 대한 score 출력
- predict : test_set 에 대한 prediction 생성, score 출력

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
    def __init__(self, 
                args:AttrDict, 
                model:ElectraForSequenceClassification, 
                train_dataset:TensorDataset, 
                eval_dataset:TensorDataset) -> None:

        self.model = model
        self.args = args,
        self.train_dataset = train_dataset, 
        self.eval_dataset = eval_dataset,

    def fit(self) -> None:

        # [Train set Dataloader]
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size, num_workers=4, pin_memory=True)

        t_total = len(train_dataloader) * self.args.num_train_epochs

        # [Optimizer]
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

        self.fitted = True
        return 
    
    def evaluate(self) -> None:

        if not self.fitted:
            raise Exception(" Model should be fitted first ! ")

        # [Validation set Dataloader] 
        eval_sampler = SequentialSampler(self.eval_dataset)
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
            
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        # [Validation Accuracy 계산]
        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)

        acc = compute_metrics('acc', out_label_ids, preds)
        
        print(f"***** Evaluation set Accuracy : {acc:.4f} *****")
    
    def predict(self, test_dataset: TensorDataset) -> pd.DataFrame:

        if not self.fitted:
            raise Exception(" Model should be fitted first ! ")

        # [Test set Dataloader]
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(self.test_dataset, 
                                    sampler=test_sampler, 
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

        # [Test 수행]
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
        all_probs=torch.max(F.softmax(torch.tensor(preds, dtype=torch.float32), dim=-1), dim=-1)[0].numpy().tolist()
        
        result_df = pd.DataFrame()
        result_df[f'prediction'] = all_preds
        result_df[f'score'] = all_probs

        print(f"***** Test set Accuracy : {acc:.4f} *****")

        return result_df

def main(cli_args:ArgumentParser):

    # [학습을 위한 Arguments 준비]
    with open(os.path.join(os.path.dirname(__file__), cli_args.config_dir, cli_args.task, cli_args.config_file)) as f:
        args = AttrDict(json.load(f))   
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"


    # [logging 준비]
    init_logger()
    logger.info("Training/evaluation parameters {}".format(args))

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


    # [모델 예측 수행]
    predictions = trainer.predict(test_dataset)
    print(predictions.head())

if __name__ == '__main__':
    cli_parser = ArgumentParser()
    cli_parser.add_argument('--task', type=str, choices = ['nsmc', 'ynat'])
    cli_parser.add_argument("--config_dir", type=str, default="config")
    cli_parser.add_argument("--config_file", type=str, default = 'koelectra-base-v3.json')
    cli_args = cli_parser.parse_args()

    main(cli_args)
