import argparse
import json
import logging
import os
import glob
import re
import time

import pandas as pd
import numpy as np
from fastprogress.fastprogress import master_bar, progress_bar
from attrdict import AttrDict
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import ElectraConfig, ElectraTokenizer, ElectraForSequenceClassification, AdamW, get_linear_schedule_with_warmup

from utils import init_logger, set_seed, compute_metrics
from preprocess import PreprocessorforHF
from callbacks import EarlyStopping

from typing import Dict

logger = logging.getLogger(__name__)

def load_data(args):
    train_df = pd.read_csv(args.data_paths["train"])
    test_df = pd.read_csv(args.data_paths["test"])

    train_df, test_df = train_df[train_df['text'].notnull()], test_df[test_df['text'].notnull()]
    return train_df, test_df

def save_results(results, path, len_ckpt=1):
    with open(path, "w") as f_w:
        if len_ckpt > 1:
            for key in sorted(results.keys(), key=lambda key_with_step: (
                    "".join(re.findall(r'[^_]+_', key_with_step)),
                    int(re.findall(r"_\d+", key_with_step)[-1][1:])
            )):
                f_w.write("{} = {}\n".format(key, str(results[key])))
        else:
            for key in sorted(results.keys()):
                f_w.write("{} = {}\n".format(key, str(results[key])))

def train(args,
          model,
          train_dataset,
          dev_dataset=None,
          test_dataset=None, train_sampler=None, val_sampler=None):
    
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4, pin_memory=True)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * args.warmup_proportion), num_training_steps=t_total)
    early_stopping = EarlyStopping()

    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Logging steps = %d", args.logging_steps)
    logger.info("  Save steps = %d", args.save_steps)

    global_step = 0
    tr_loss = 0.0
    best_val_metric = 999999.0

    mb = master_bar(range(int(args.num_train_epochs)))
    for epoch in mb:
        epoch_iterator = progress_bar(train_dataloader, parent=mb)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3]
            }
            if args.model_type not in ["distilkobert", "xlm-roberta"]:
                inputs["token_type_ids"] = batch[2]  # Distilkobert, XLM-Roberta don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    len(train_dataloader) <= args.gradient_accumulation_steps
                    and (step + 1) == len(train_dataloader)
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

        if args.logging_epochs > 0 and epoch % args.logging_epochs == 0:
            if args.evaluate_test_during_training:
                results = evaluate(args, model, test_dataset, "test", global_step)
            else:
                results = evaluate(args, model, train_dataset, "dev", global_step, eval_sampler=val_sampler, epoch=epoch)
            
            val_loss = results["Val_loss"]

            if val_loss < best_val_metric:
                best_val_metric = val_loss
                logger.info('Saving best models...')

                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, f"checkpoint-ep{epoch}-{global_step}-sc{results['micro_f1']:.4f}-vls{val_loss:.4f}")
                print('output_dir:', output_dir)
                if not os.path.exists(output_dir):
                    print('output_dir made:', output_dir)
                    os.makedirs(output_dir)
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )
                model_to_save.save_pretrained(output_dir)

                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving model checkpoint to {}".format(output_dir))

                if args.save_optimizer:
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to {}".format(output_dir))

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        early_stopping(val_loss)
        if early_stopping.early_stop:
            break

        mb.write("Epoch {} done".format(epoch + 1))

    return global_step, tr_loss / global_step, epoch

def evaluate(args, model, eval_dataset, mode, global_step=None, n_fold=None, eval_sampler = None, save_pred = None, epoch=None):
    results = {}

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, 
                                num_workers=4, pin_memory=True)

    if global_step != None:
        logger.info(f"***** Running evaluation on {mode} dataset ({global_step} step) *****")
    else:
        logger.info(f"***** Running evaluation on {mode} dataset *****")
    
    #logger.info("  Num examples = {}".format(len(eval_dataset)))
    logger.info(f"  Num examples = {len(eval_sampler.indices) if eval_sampler else len(eval_dataset)}")
    logger.info(f"  Eval Batch size = {args.eval_batch_size}")
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    for batch in progress_bar(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids" : batch[2],
                "labels": batch[3]
            }
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    # [Evaluation/Test metric 계산]
    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)

    result = compute_metrics(args.task, out_label_ids, preds)
    results.update(result)

    if mode=='dev':
      results['Val_loss'] = eval_loss

    return results

def main(cli_args):
    # Read from config file and make args
    with open(os.path.join(os.path.dirname(__file__), cli_args.config_dir, cli_args.config_file)) as f:
    # with open(cli_args.config_file) as f:
        args = AttrDict(json.load(f))   

    init_logger()

    logger.info("Training/evaluation parameters {}".format(args))
    args.output_dir = f"{cli_args.task}_SD{args.seed}-{args.output_dir}"
    args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)
    
    # GPU or CPU
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    train_df, test_df = load_data(cli_args)

    set_seed(args)

    tokenizer = ElectraTokenizer.from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case
    )

    processor = PreprocessorforHF(args)
    train_dataset = processor.load_and_cache(args, tokenizer, train_df, "train")
    dev_dataset = None
    test_dataset = processor.load_and_cache(args, tokenizer, test_df, "test")
    
    label_list, lb2int = processor.get_label_info()

    config = ElectraConfig.from_pretrained(
        args.model_name_or_path,
        num_labels = len(label_list),
        id2label = {label_id: label for label, label_id in lb2int.item()},
        label2id = lb2int,
    )

    if dev_dataset == None:
        args.evaluate_test_during_training = True  # If there is no dev dataset, only use testset
    
    if args.do_train:
        model = ElectraForSequenceClassification.from_pretrained(
                        args.model_name_or_path,
                        config=config
                    )
        model.to(args.device)
        
        start_time = time.time()
        global_step, tr_loss, epoch = train(args, model, train_dataset)
        train_time = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))
        
        logger.info(f"epoch = {epoch}, global_step = {global_step}, average loss = {tr_loss}")

    if args.do_eval:
        checkpoints = list(os.path.dirname(c) for c in
                        sorted(glob.glob(args.output_dir + "/**/" + "pytorch_model.bin", recursive=True),
                                key=lambda path_with_step: list(map(int, re.findall(r"\d+", path_with_step)))[-1],
                                reverse=True
                                ))
        if not args.eval_all_checkpoints:
            checkpoints = checkpoints[-1:]
        else:
            logging.getLogger("transformers.configuration_utils").setLevel(logging.WARN)  # Reduce logging
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        test_results = {}

        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-2]
            global_epoch = checkpoint.split("-")[-2]
            model = ElectraForSequenceClassification.from_pretrained(checkpoint)
            model.to(args.device)
            
            test_result = evaluate(args, model, test_dataset, "best_val", global_step, epoch=epoch)

            test_result = dict((k + f"_{global_epoch}_{global_step}", v) for k, v in test_result.items())

            test_results.update(test_result)
        
        test_result['train_time'] = train_time

        if args.save_results:
            results_dir = args.save_results
            if not os.path.exists(results_dir):
                print('output_dir made:', results_dir)
                os.makedirs(results_dir)

            output_eval_trainset_file = os.path.join(args.save_results, f"{args.task}_SD{args.seed}_test_results.txt")
            save_results(test_result, output_eval_trainset_file, len(checkpoints))
    

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--task', type=str, choices = ['binary', 'multi'])
    cli_parser.add_argument("--config_dir", type=str, default="config")
    cli_parser.add_argument("--config_file", type=str, default = 'koelectra-base-v3.json')

    cli_args = cli_parser.parse_args()

    cli_args.data_paths = {"train": f"../../data/train_{cli_args.task}.csv",
                            "test": f"../../data/test_{cli_args.task}.csv",}

    main(cli_args)

# python finetune_without_trainer.py --task binary