import argparse
import csv
import json
import logging
import os
import random
import sys
import copy
import math
import time
import numpy as np
import torch
import torch.nn.functional as F

from prettytable import PrettyTable
from torch.autograd import Variable
from pytorch_transformers import (WEIGHTS_NAME, AdamW, BertConfig,
                                  BertForTokenClassification, BertTokenizer,
                                  WarmupLinearSchedule)
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from seqeval.metrics import classification_report

from model import Ner


logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s-%(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

def full_train(data_loader=None, test_data_loader=None, model=None, Epochs=5, soft_loader=None, args=None, num_labels=None, device=None, n_gpu=None, label_list=None):
    config = BertConfig.from_pretrained(args.bert_model, num_labels=num_labels, finetuning_task=args.task_name)
    if model==None:
      model = Ner.from_pretrained(args.bert_model, from_tf = False, config = config)
    return_model = Ner.from_pretrained(args.bert_model, from_tf = False, config = config)
    model.to(device)
    return_model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias','LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    num_train_optimization_steps = int(len(data_loader.dataset)/args.train_batch_size/args.gradient_accumulation_steps)*args.num_train_epochs #2190
    warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optimization_steps)
    
    current_train_size = 0
    if soft_loader:
      current_train_size = len(data_loader.dataset) + len(soft_loader.dataset)
    else:
      current_train_size = len(data_loader.dataset)
    print('Training on {} data'.format(current_train_size))
  
    #model.train()
    tr_loss = 2020
    
    test_f1 = []
    best_test_f = -1
    
    for epoch_idx in trange(int(Epochs), desc="Epoch"):
        current_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        
        model.train()
        
        for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, valid_ids,l_mask = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids, valid_ids, l_mask)
            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            current_loss += loss
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
        
        ##eval for each epoch
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        y_true = []
        y_pred = []
        raw_logits = []
        
        label_map = {i : label for i, label in enumerate(label_list,1)}
        for input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask in tqdm(test_data_loader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            valid_ids = valid_ids.to(device)
            label_ids = label_ids.to(device)
            l_mask = l_mask.to(device)
    
            with torch.no_grad():
                logits,_ = model(input_ids, segment_ids, input_mask,valid_ids=valid_ids,attention_mask_label=l_mask)
            
            #raw_logits.append(logits)
            logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()
    
            for i, label in enumerate(label_ids):
                temp_1 = []
                temp_2 = []
                for j,m in enumerate(label):
                    if j == 0:
                        continue
                    elif label_ids[i][j] == len(label_map):
                        y_true.append(temp_1)
                        y_pred.append(temp_2)
                        break
                    else:
                        temp_1.append(label_map[label_ids[i][j]])
                        try:
                          temp_2.append(label_map[logits[i][j]])
                        except:
                          temp_2.append('UKN')
    
        report = classification_report(y_true, y_pred, digits=4)
        logger.info("\n%s", report)
        temp = report.split('\n')[-3]
        f1 = eval(temp.split()[-2])
        test_f1.append(f1)
        
        
        if f1 >= best_test_f:
            best_test_f = f1
        
        
        output_eval_file = os.path.join(args.full_train_output_dir, "test_results.txt")
        with open(output_eval_file, "a") as writer:
            writer.write('*******************epoch*******'+str(epoch_idx)+'\n')
            writer.write(report+'\n')
        
        output_f1_test = os.path.join(args.full_train_output_dir, "f1_score_epoch.txt")   
        with open(output_f1_test, "w") as writer1:
            for i in test_f1:
                writer1.write(str(i)+'\n')
            writer1.write('\n')
            writer1.write(str(best_test_f))
        
        
        if soft_loader:
          for input_ids, input_mask, segment_ids, soft_labels, valid_ids,l_mask in tqdm(soft_loader, desc="Soft Training"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                soft_labels = soft_labels.to(device)
                l_mask = l_mask.to(device)
                #with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask,valid_ids=valid_ids,attention_mask_label=l_mask)
                #logits = F.softmax(logits, dim=2)
                logits = logits.detach().cpu().float()
                soft_labels = soft_labels.detach().cpu().float()
                pos_weight = torch.ones([num_labels])  # All weights are equal to 1
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                loss = 0
                for i in range(len(logits)):
                  turncate_len = np.count_nonzero(l_mask[i].detach().cpu().numpy())
                  logit = logits[i][:turncate_len]
                  soft_label = soft_labels[i][:turncate_len]
                  loss += criterion(logit, soft_label)
                loss = Variable(loss, requires_grad=True)
                current_loss += loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
        if current_loss <= tr_loss:
          return_model.load_state_dict(model.state_dict())
          tr_loss = current_loss
  
    return return_model