'''
Sample Run Command:
===================
1. Baseline:
   $ python run.py --task method --data_dir ./datasets/ --output_dir ./outputs/ --do_train

   $ python run.py --task method --data_dir ./datasets/ --output_dir ./outputs/ --do_eval --load_model_path <path-to-model>

2. Hybrid CRL:
   $ python run.py --task method --data_dir ./datasets/ --output_dir ./outputs/ --do_train --hybrid

   $ python run.py --task method --data_dir ./datasets/ --output_dir ./outputs/ --do_eval --hybrid --load_model_path <path-to-model>
'''
from __future__ import absolute_import
import os
import sys
import pickle
import torch
import json
import random
import logging
import argparse
from pathlib import Path

import numpy as np

import pandas as pd

import wandb

from tqdm import tqdm, trange

from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)

import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
from torch.utils.data import (DataLoader, Dataset, SequentialSampler,
                              RandomSampler, TensorDataset)

import transformers
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaTokenizer)

from utils import (get_static_tokens, load_w2v_model,
                   train_hybrid_tokenizer, get_static_ids)
from load_data import DataProcessor
from model import BugDetectionModel

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class MeanCosineLoss(nn.Module):
    def __init__(self):
        super(MeanCosineLoss, self).__init__()

    def forward(self, embeddings_stat, embeddings_con, device):
        '''
        Since the tensor label is 1, cosine_loss = 1 - cos(x1, x2).
        '''
        batch_size = len(embeddings_stat)
        assert len(embeddings_stat) == len(embeddings_con)

        total_cosine_loss = torch.tensor([0.0]).to(device)
        for instance_stat, instance_con in zip(embeddings_stat, embeddings_con):
            if len(instance_stat) != 0:
                instance_stat = np.stack(instance_stat, axis=0)
                instance_stat = torch.tensor(instance_stat).to(device)
                total_cosine_loss += 1 - torch.mean(cosine_similarity(
                    instance_stat, instance_con))
        mean_cosine_loss = total_cosine_loss / batch_size
        return mean_cosine_loss


class MeanSquareLoss(nn.Module):
    def __init__(self):
        super(MeanSquareLoss, self).__init__()

    def forward(self, embeddings_stat, embeddings_con, device):
        '''
        Since the tensor label is 1, cosine_loss = 1 - cos(x1, x2).
        '''
        batch_size = len(embeddings_stat)
        assert len(embeddings_stat) == len(embeddings_con)

        total_loss = torch.tensor([0.0]).to(device)
        for instance_stat, instance_con in zip(embeddings_stat, embeddings_con):
            if len(instance_stat) != 0:
                instance_stat = np.stack(instance_stat, axis=0)
                instance_stat = torch.tensor(instance_stat, dtype=torch.float32).to(device)
                # print(instance_stat.shape)
                # print(instance_con.shape)
                total_loss += nn.MSELoss(reduction='mean')(instance_stat, instance_con)
        mean_square_loss = total_loss / batch_size
        return mean_square_loss


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def make_dataloader(args, examples, tokenizer, stage):
    '''
    '''
    if stage == 'train':
        batch_size = args.train_batch_size
    else:
        batch_size = args.eval_batch_size

    path_to_save = Path(args.data_dir)
    path_to_save.mkdir(exist_ok=True, parents=True)
    path_to_file = path_to_save / f"{args.task}_dataloader_{stage}.pkl"

    try:
        with open(str(path_to_file), 'rb') as handler:
            dataloader = pickle.load(handler)
    except FileNotFoundError:
        labels = tuple([torch.tensor([int(example.label) \
                                      for example in examples])])

        code_examples = [getattr(example, 'code') for example in examples]
        encoded_inputs = tokenizer(code_examples,
                                   padding=True,
                                   truncation=True,
                                   max_length=args.max_source_length,
                                   return_tensors='pt')
        features = (
            torch.tensor(encoded_inputs['input_ids'], dtype=torch.long),
            torch.tensor(encoded_inputs['attention_mask'], dtype=torch.long),
        )
        dataset = TensorDataset(*(features + labels))

        if stage == 'train':
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

        with open(str(path_to_file), 'wb') as handler:
            pickle.dump(dataloader, handler)

    return dataloader


def evaluate(dataloader, model, args, epoch_stats=None):
    '''
    '''
    # Tracking variables
    total_eval_loss = 0
    # Evaluate data for one epoch
    preds, true_labels = [], []

    if not epoch_stats:
        stats = {}

    for batch in dataloader:
        batch = tuple(t.to(args.device) for t in batch)
        # Tell pytorch not to bother with constructing the compute
        # graph during the forward pass, since this is only needed
        # for backpropogation (training).
        with torch.no_grad():
            input_ids, attention_masks, batch_labels = (batch[0], batch[1],
                                                        batch[2])

            batch_outputs, _ = model(input_ids, attention_masks, batch_labels)
            batch_loss = F.cross_entropy(batch_outputs, batch_labels)

        # Accumulate the validation loss.
        total_eval_loss += batch_loss.item()

        # Move labels to CPU
        batch_preds = torch.max(batch_outputs.data, 1)[1].cpu().numpy()
        preds += batch_preds.tolist()
        batch_labels = batch_labels.to('cpu').numpy()
        true_labels += batch_labels.tolist()

    # Calculate the average loss over all of the batches.
    epoch_eval_loss = total_eval_loss / len(dataloader)

    # Record all statistics.
    return {
        **epoch_stats, **{'Epoch evaluation loss': epoch_eval_loss,
                          'Accuracy': accuracy_score(true_labels, preds),
                          'Precision': precision_score(true_labels, preds),
                          'Recall': recall_score(true_labels, preds),
                          'F1-Score': f1_score(true_labels, preds)}
    }


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--task", type=str, required=True,
                        choices=['method', 'stmt'], help="Bug detection task.")
    parser.add_argument("--hybrid", action='store_true',
                        help="If True, train hybrid bug detection model.")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="Path to datasets directory.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Experiment arguments
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    ## Other parameters
    parser.add_argument("--max_source_length", default=256, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--log_interval", default=10, type=int)
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs", default=5, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # Print arguments
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.n_gpu = torch.cuda.device_count()

    logger.warning(f"Device: {args.device}, Number of GPU's: {args.n_gpu}")

    # Set seed
    set_seed(args.seed)

    # Initialize tokenizer
    if not args.hybrid:
        output_dir = Path(args.output_dir) / 'base' / args.task
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        w2v_model = load_w2v_model(args.task)
        # Load static token ids
        static_mapper = get_static_ids(w2v_model, tokenizer)
        static_pos_ids = set(list(static_mapper.keys()))

    else:
        output_dir = Path(args.output_dir) / 'hybrid' / args.task
        path_to_tok = Path('tokenizer')

        if not path_to_tok.exists():
            logger.info('Training hybrid byte-level BPE tokenizer.')
            train_hybrid_tokenizer(args.task)

        tokenizer = RobertaTokenizer(vocab_file=str(path_to_tok / 'vocab.json'),
                                     merges_file=str(path_to_tok / 'merges.txt'))
        # Load Word2Vec model
        w2v_model = load_w2v_model(args.task)
        # Load static token ids
        static_mapper = get_static_ids(w2v_model, tokenizer)
        static_pos_ids = set(list(static_mapper.keys()))

    model = BugDetectionModel()

    if args.load_model_path is not None:
        logger.info(f"Reload model from {args.load_model_path}")
        model.load_state_dict(torch.load(args.load_model_path), strict=False)

    model.to(args.device)

    # Make directory if output_dir does not exist
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    cb_ebd_dict = {}
    if args.hybrid:
        f = open('./cb_ebd_dict.pkl', 'rb')
        cb_ebd_dict = pickle.load(f)
        f.close()
        for ids, key in enumerate(cb_ebd_dict):
            tmp = [i / cb_ebd_dict[key][1] for i in cb_ebd_dict[key][0]]
            cb_ebd_dict[key] = [tmp, cb_ebd_dict[key][1]]
    if args.do_train:
        wandb.init(project="bug-detection", name=str(args.task).upper())
        wandb.config.update(args)

        # Prepare training data loader
        dp = DataProcessor(data_dir=args.data_dir)
        logger.info('Loading training data.')
        train_examples = dp.get_train_examples(args.task)
        logger.info('Constructing data loader for training data.')
        train_dataloader = make_dataloader(args, train_examples, tokenizer, 'train')

        # Prepare validation data loade.
        logger.info('Loading validation data.')
        eval_examples = dp.get_val_examples(args.task)
        logger.info('Constructing data loader for validation data.')
        eval_dataloader = make_dataloader(args, eval_examples, tokenizer, 'val')

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() \
                        if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() \
                        if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate,
                          eps=args.adam_epsilon)
        max_steps = len(train_dataloader) * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=max_steps * 0.1,
                                                    num_training_steps=max_steps)
        # Start training
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_examples)}")
        logger.info(f"  Batch size = {args.train_batch_size}")
        logger.info(f"  Num epoch = {args.num_train_epochs}")

        training_stats = []
        model.zero_grad()

        results = []
        step_cnt = 0

        for epoch in range(args.num_train_epochs):
            tr_loss = 0
            msl_loss = 0
            ce_loss = 0
            num_train_steps = 0
            num_correct = 0
            tr_preds, tr_labels = [], []

            model.train()
            for _, batch in tqdm(enumerate(train_dataloader)):
                batch = tuple(t.to(args.device) for t in batch)
                input_ids, attention_masks, batch_labels = batch[0], batch[1], batch[2]
                # for i in input_ids:
                #     tmp = tokenizer.convert_ids_to_tokens(i)
                #     print(tmp)
                if args.hybrid:
                    static_position_mapper = []
                    for item_input_ids in input_ids:
                        _mapper = {}
                        # print(item_input_ids.tolist())
                        for index, _id in enumerate(item_input_ids.tolist()):
                            if _id in static_pos_ids:
                                # _mapper[index] = static_mapper[_id]
                                _mapper[index] = _id
                                # print('in...')
                                # print(_id)
                                # print(static_mapper[_id])
                        static_position_mapper.append(_mapper)

                    batch_outputs, token_states = model(input_ids, attention_masks, batch_labels)

                    token_outputs = [token_states[b_id,
                                                  list(static_position_mapper[b_id].keys())] \
                                     for b_id in range(len(token_states))]

                    embeddings_stat = []
                    for b_id in range(len(token_states)):
                        _emb = [cb_ebd_dict[static_token][0] for static_token \
                                in list(static_position_mapper[b_id].values())]
                        embeddings_stat.append(_emb)
                    # print(embeddings_stat, token_outputs)
                    # loss_mcl = MeanCosineLoss()(embeddings_stat, token_outputs, args.device)
                    loss_msl = MeanSquareLoss()(embeddings_stat, token_outputs, args.device)
                    loss_ce = CrossEntropyLoss(ignore_index=-1)(batch_outputs, batch_labels)
                    # print(loss_msl, loss_msl.dtype)
                    # print(loss_ce, loss_ce.dtype)
                    batch_loss = torch.mean(loss_msl + loss_ce)
                    # print(batch_loss, batch_loss.dtype)
                    msl_loss += loss_msl.item()
                else:
                    batch_outputs, token_states = model(input_ids, attention_masks, batch_labels)
                    loss_ce = CrossEntropyLoss(ignore_index=-1)(batch_outputs, batch_labels)
                    batch_loss = loss_ce
                    if epoch == args.num_train_epochs - 1:
                        for bid in range(len(input_ids)):
                            item_input_ids = input_ids[bid].tolist()
                            for token_id in range(len(item_input_ids)):
                                if item_input_ids[token_id] in static_pos_ids:
                                    # print(item_input_ids[token_id])
                                    # print(token_states[bid][token_id].tolist()[:5])
                                    if item_input_ids[token_id] not in cb_ebd_dict.keys():
                                        cb_ebd_dict[item_input_ids[token_id]] = [
                                            token_states[bid][token_id].tolist(), 1]
                                    else:
                                        tmp = []
                                        a1 = cb_ebd_dict[item_input_ids[token_id]][0]
                                        a2 = token_states[bid][token_id].tolist()
                                        for bits in range(len(a2)):
                                            tmp.append(a1[bits] + a2[bits])
                                        cb_ebd_dict[item_input_ids[token_id]] = [
                                            tmp, cb_ebd_dict[input_ids[bid][token_id].item()][1] + 1]
                                    # for ids, key in enumerate(cb_ebd_dict):
                                        # pass
                                        # print(key, cb_ebd_dict[key][1], cb_ebd_dict[key][0][:5])
                # a = abcde
                tr_loss += batch_loss.item()
                ce_loss += loss_ce.item()
                num_train_steps += 1

                prob = torch.softmax(batch_outputs, -1)
                batch_preds = torch.max(prob, 1)[1].cpu().numpy()
                tr_preds += batch_preds.tolist()
                batch_labels = batch_labels.to('cpu').numpy()
                tr_labels += batch_labels.tolist()

                if _ % 5000 == 0:
                    step_loss = tr_loss / num_train_steps
                    if args.hybrid:
                        step_msl_loss = msl_loss / num_train_steps
                    else:
                        step_msl_loss = 0.0
                    step_ce_loss = ce_loss / num_train_steps
                    step_accuracy = accuracy_score(tr_labels, tr_preds)
                    logger.info(f"Epoch {epoch}, Training loss per 5000 steps: {step_loss}")
                    logger.info(f"Epoch {epoch}, Training accuracy per 5000 steps: {step_accuracy}")
                    wandb.log({"Training loss per 5000 steps": step_loss,
                               "Training accuracy per 5000 steps": step_accuracy},
                              step=num_train_steps)
                    results.append([step_cnt, step_loss, step_msl_loss, step_ce_loss, step_accuracy])
                    step_cnt += 1

                # print(batch_loss, batch_loss.dtype)
                batch_loss.backward()
                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # Update parameters
                optimizer.step()
                optimizer.zero_grad()
                # Update the learning rate
                scheduler.step()

            # a = abcde
            epoch_tr_loss = tr_loss / len(train_dataloader)
            epoch_accuracy = accuracy_score(tr_labels, tr_preds)

            logger.info(f"Epoch {epoch}, Training loss: {epoch_tr_loss}")
            logger.info(f"Epoch {epoch}, Training accuracy: {epoch_accuracy}")

            # After the completion of one training epoch, measure performance
            # on validation set.
            logger.info('Measuring performance on validation set.')
            # Put the model in evaluation mode--the dropout layers behave
            # differently during evaluation.
            model.eval()
            training_stats = evaluate(eval_dataloader, model, args,
                                      epoch_stats={
                                          'Epoch training loss': epoch_tr_loss,
                                          'Epoch accuracy': epoch_accuracy
                                      }
                                      )
            print(training_stats)

            epoch_output_dir = Path(output_dir) / f'Epoch_{epoch}'
            epoch_output_dir.mkdir(exist_ok=True, parents=True)
            logger.info(f"Saving model to {epoch_output_dir}")
            torch.save(model.state_dict(), str(epoch_output_dir / 'model.ckpt'))

            print(len(results))
            with open('./loss_log.json', 'w') as f:
                f.write(json.dumps(results))

            print(len(cb_ebd_dict))
            f = open('./cb_ebd_dict.pkl', 'wb')
            pickle.dump(cb_ebd_dict, f)
            f.close()

    if args.do_eval:
        # Put the model in evaluation mode--the dropout layers behave
        # differently during evaluation.
        model.eval()

        # Load test data.
        dp = DataProcessor(data_dir=args.data_dir)
        logger.info('Loading training data.')
        test_examples = dp.get_val_examples(args.task)
        test_dataloader = make_dataloader(args, test_examples, model, 'val')
        stats = evaluate(test_dataloader, model, args)
        print("  Test loss: {0:.4f}".format(stats['Epoch evaluation loss']))
        print("  Accuracy: {0:.4f}".format(stats['Accuracy']))
        print("  Precision: {0:.4f}".format(stats['Precision']))
        print("  Recall: {0:.4f}".format(stats['Recall']))
        print("  F1-Score: {0:.4f}".format(stats['F1-Score']))


if __name__ == "__main__":
    main()
