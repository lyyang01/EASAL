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
from data_loader2 import readfile, NerProcessor, convert_examples_to_features
from ori_train import full_train



#os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_tr_set(initial_idx=None, train_examples=None, batch_size=32, soft_labels=[], args=None, type_=None, start_end=None, ori_len=None, predicted_label=None, iterative_len=None, do_sampling=None):
    train_features, ori_sent, ori_label = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer, logger, start_end=start_end, ori_len=ori_len, predicted_label=predicted_label, iterative_len=iterative_len, do_sampling=do_sampling)
    if initial_idx: # return part of features
      #select_idx = np.random.choice(range(len(train_features)), size=size, replace=False)
      train_features = list(np.array(train_features)[initial_idx])
  
    logger.info("  Num examples = %d", len(train_examples))
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_valid_ids = torch.tensor([f.valid_ids for f in train_features], dtype=torch.long)
    all_lmask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
    if len(soft_labels):
      all_label_ids = torch.tensor([soft_label for soft_label in soft_labels], dtype=torch.float64)
    else:
      all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids, all_lmask_ids)
    if args.local_rank == -1:
      if type_!=None:
          train_sampler = SequentialSampler(train_data)
      else:
          train_sampler = RandomSampler(train_data)
    else:
      train_sampler = DistributedSampler(train_data)
  
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    if initial_idx:
      return train_dataloader, select_idx
    return train_dataloader, ori_sent, ori_label

def get_eval_set(eval_on, eval_batch_size=8):
    if eval_on == "dev":
      eval_examples = processor.get_dev_examples(args.data_dir, args.dev_data)
    elif eval_on == "test":
      eval_examples = processor.get_test_examples(args.data_dir, args.test_data)
    else:
      raise ValueError("eval on dev or test set only")
    eval_features, _, _ = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, logger, start_end=None)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
    all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_valid_ids,all_lmask_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)
    return eval_dataloader

'''Evaluation'''
def evaluate(prefix=None, model=None, args=None):
    eval_dataloader = get_eval_set(eval_on=args.eval_on, eval_batch_size=args.eval_batch_size)
    model.to(device)
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    y_true = []
    y_pred = []
    raw_logits = []
    label_map = {i : label for i, label in enumerate(label_list,1)}
    for input_ids, input_mask, segment_ids, label_ids,valid_ids,l_mask in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        valid_ids = valid_ids.to(device)
        label_ids = label_ids.to(device)
        l_mask = l_mask.to(device)

        with torch.no_grad():
            logits,_ = model(input_ids, segment_ids, input_mask, valid_ids=valid_ids, attention_mask_label=l_mask)
        
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
    return report

def save_result(prefix='Active', func_paras=None, report=None, table=None, output_dir=None):
    result_path = os.path.join(output_dir, prefix+'.txt')
    with open(result_path,'a') as f:
      if func_paras:
        for para in func_paras:
          if(type(func_paras[para]))==np.ndarray:
            func_paras[para] = func_paras[para].shape
          if(type(func_paras[para]))==list:
            func_paras[para] = np.array(func_paras[para]).shape 
        f.write('\nParameters:\n')
        for item in func_paras.items():
          f.write(str(item)+'\n')
      if report:
        f.write(report)
      if table:
        table = table.get_string()
        f.write(table)

def multi_argmax(values: np.ndarray, n_instances: int = 1) -> np.ndarray:
    """
    Selects the indices of the n_instances highest values.

    Input:
      values: Contains the values to be selected from.
      n_instances: Specifies how many indices to return.
    Output:
      The indices of the n_instances largest values.
    """
    assert n_instances <= values.shape[0], 'n_instances must be less or equal than the size of utility'

    max_idx = np.argpartition(-values, n_instances-1, axis=0)[:n_instances]
    return max_idx


def uncertainty_sampling(model_instance, pool, A=None, f=None, min_len=None, mi=1):
  
    active_eval_loader, ori_sent, ori_label = get_tr_set(train_examples=pool, batch_size=32, args=args, type_='query')
    raw_prediction, turncate_list, predicted_label = active_eval(active_eval_loader, model_instance) # predict, get the softmax output
    #np.shape(raw_prediction) [6232, 128, 6]
    #import pdb
    #pdb.set_trace()
    
    # predicted_label (length:126 | no [CLS] and [SEP] token)
    # turncate_list (length is from [CLS] to [SEP])
    
    word_prob = np.max(raw_prediction,axis=2) # select the max probability prediction as the word tag
    sentence_uncertainty = []
    
    
    label_map = {i : label for i, label in enumerate(label_list,1)}
    predicted_label_str = []
    
    for idx, i in enumerate(predicted_label):
        temp = i[0: turncate_list[idx]-2]
        predict_temp = []
        for j in temp:
            predict_temp.append(label_map[j])
        predicted_label_str.append(predict_temp)
    
    # predicted_label_str (length no [CLS] and [SEP]. it is the true length of these sentences)
    
    #import pdb
    #pdb.set_trace()
    
    ori_uncertainty = []
    for i, sentence in enumerate(word_prob):
      ori_uncertainty.append(-np.log(sentence[1:turncate_list[i]-1])) ##drop [CLS] and [SEP]
    
    #import pdb
    #pdb.set_trace()
    
    for i, sentence in enumerate(word_prob):
      #sentence_uncertainty.append(1/(turncate_list[i])*np.sum(1-sentence[:turncate_list[i]]))
      sentence_uncertainty.append(1/((turncate_list[i]))*np.sum(-np.log(sentence[:turncate_list[i]])))
      #sentence_uncertainty.append(np.sum(-np.log(1-sentence[:turncate_list[i]])))
    
    #import pdb
    #pdb.set_trace()
    
    ##
    new_sentence_uncertainty = []
    temp_uncertainty = []
    
    ##TODO
    min_len = min_len
    
    for idx, i in enumerate(ori_uncertainty):
        start = 0
        end = len(i)
        initial_certain = np.mean(i)
        start_certain = initial_certain
        end_certain = initial_certain
        
        ##TODO
        '''
        start_indicator = 0
        end_indicator = 0
        
        while(1):
            ##start
            if start_indicator != 1:
                start += 1
                current_certain = np.mean(i[start:])
                if current_certain < start_certain:
                    new_start = start - 1
                    start_indicator = 1
                else:
                    #start_certain = current_certain
                    end_certain = current_certain
            ##end
            if end_indicator != 1:
                end -= 1
                current_certain = np.mean(i[start:end])
                if current_certain < end_certain:
                    new_end = end + 1
                    end_indicator = 1
                else:
                    start_certain = current_certain
            
            if (end_indicator == 1 and start_indicator == 1):
                break
        '''     
        
        
        while(1):
            start += 1
            current_certain = np.mean(i[start:])
            if current_certain < start_certain:
                new_start = start - 1
                break
            else:    
                start_certain = current_certain
            if start == len(i):
                new_start = start
                break
    
    
        while(1):
            end -= 1
            current_certain = np.mean(i[0:end])
            if current_certain < end_certain:
                new_end = end + 1
                break
            else:
                end_certain = current_certain
            if end == 0:
                new_end = end
                break
        
        if new_start >= new_end:
            new_certain = 0
            new_start = 0
            new_end = 0
        else:
            if new_start > len(i) or new_end > len(i):
                import pdb
                pdb.set_trace()
            
            #TODO
            key_temp = 0
            start_in = 0
            end_in = 0
            if new_start == 0 and new_end == 0:
                pass
            else:
                while (new_end - new_start) < min_len:
                    if key_temp%2 == 0:
                        if new_start - 1 < 0:
                            start_in = 1
                            pass
                        else:
                            new_start -= 1
                    if key_temp%2 == 1:
                        if new_end + 1 > turncate_list[idx] - 2:
                            end_in = 1
                            pass
                        else:
                            new_end += 1
                    key_temp += 1
                    
                    if start_in == 1 and end_in == 1:
                        break
              
                
            new_certain = np.mean(i[new_start:new_end])
            #new_certain = 1/(new_end - new_start)*np.sum(-np.log(sentence[new_start+1:new_end+1]))
        
        new_sentence_uncertainty.append((new_certain, new_start, new_end))    
        temp_uncertainty.append(new_certain)    
        
    #import pdb
    #pdb.set_trace()
    #query_index = multi_argmax(np.array(sentence_uncertainty), len(sentence_uncertainty))
    sort_index = np.argsort(-np.array(temp_uncertainty))
    sort_index_temp = np.argsort(-np.array(sentence_uncertainty))
    #import pdb
    #pdb.set_trace()
    '''
    query_index_temp = []
    temp_tokenum = 0
    for i in sort_index:
        temp_tokenum += len(pool[i].label)
        if temp_tokenum <= A:
            query_index_temp.append(i)
        else:
            break
    '''
    #import pdb
    #pdb.set_trace()
    
    query_index = []
    temp_tokenum = 0
    start_end = []
    select = []
    predict_label_final = []
    for i in sort_index:
        start = new_sentence_uncertainty[i][1]
        end = new_sentence_uncertainty[i][2]
        temp_tokenum += len((pool[i].label)[start+1:end+1])
        if temp_tokenum <= A:
            query_index.append(i)
            select.append(len((pool[i].label)))
            start_end.append((start+1, end+1))
            predict_label_final.append(predicted_label_str[i])
        else:
            break
    
    
    #import pdb
    #pdb.set_trace()
    
    #import pdb
    #pdb.set_trace()
    
    all_token_num = 0
    token_with_entitylabel = 0
    
    word_prob_all = []
    all_num = 0
    for i in query_index:
        all_token_num += new_sentence_uncertainty[i][2] - new_sentence_uncertainty[i][1]
        
        ori_uncertain = ori_uncertainty[i]
        #word_prob_temp[:turncate_list[i]]
        word_prob_all.extend(ori_uncertain[new_sentence_uncertainty[i][1]:new_sentence_uncertainty[i][2]])
        
        all_num += new_sentence_uncertainty[i][2] - new_sentence_uncertainty[i][1]
        
        for j in range(len(ori_label[i])):
            if j >= new_sentence_uncertainty[i][1] and j < new_sentence_uncertainty[i][2]:
                if ori_label[i][j]!='O':
                    token_with_entitylabel += 1   
    
    lc_score = 1/all_num*np.sum(word_prob_all)
    #str_query_index = [str(i) for i in query_index]
    #if f:
    f.write(str(all_token_num) + ' ' + str(token_with_entitylabel)+ ' ' + str(lc_score) +'\n')
    #import pdb
    #pdb.set_trace()       
    ##
    #query_index = np.sort(query_index).tolist()
    #query_index.reverse()
    ##
    
    #str_query_index = [str(i) for i in query_index]
    #if f:
    #    f.write('*************************************'+' '.join(str_query_index) + '\n')
    #    for j in query_index:
    #        f.write(pool[j].text_a+'\n')
    #        f.write(' '.join(pool[j].label)+'\n')
    #        f.write('\n')
    #    f.write('************************************'+'\n')
    #import pdb
    #pdb.set_trace()
    #import pdb
    #pdb.set_trace()
    
    return query_index, pool[query_index], start_end, predict_label_final


def cal_vote_entropy(mc_pred):
    '''
    Calculate the vote entropy
  
    Input:
      mc_pred: 3d-shape (num_mc_model * num_sentence * max_len * n_tags)
    Output:
      vote_entropy: 2d-shape (num_sentence * max_len)
    '''
    num_mc_model = len(mc_pred)
    num_sentence = mc_pred[0].shape[0]
  
    print('vote_matrix')
    vote_matrix = np.zeros((num_sentence, args.max_seq_length, num_labels))
    for model_idx, pred in enumerate(mc_pred):
      for s_idx, sentence in enumerate(pred):
        for w_idx, word in enumerate(sentence):
          vote_matrix[s_idx][w_idx][word] += 1
    print('vote_prob_matrix')
    vote_prob_matrix = np.zeros((num_sentence, args.max_seq_length, num_labels))
    for s_idx, sentence in enumerate(vote_matrix):
      for w_idx, word in enumerate(sentence):
        for tag_idx in range(num_labels):
          prob_i = np.sum(word==tag_idx) / num_mc_model
          vote_prob_matrix[s_idx][w_idx][tag_idx] = prob_i
    print('vote_entropy')
    vote_entropy = np.zeros(num_sentence)
    for s_idx, sentence in enumerate(vote_prob_matrix):
      sentence_entropy = 0
      for w_idx, word in enumerate(sentence):
        word_entropy = 0
        for tag_prob in word:
          if tag_prob:
            word_entropy -= tag_prob*(math.log(tag_prob)) 
        sentence_entropy += word_entropy
      vote_entropy[s_idx] = sentence_entropy
    
    return vote_entropy

def result2tag(result, turncate):
    '''
    Convert the result with 3-d shape to the tags with 2-d shape. 
    '''
    sentences = []
    for idx, sentence in enumerate(result):
      valid_len = turncate[idx]
      words = []
      for word in sentence[:valid_len]:
        word = word.tolist()
        tag = word.index(max(word))
        words.append(tag)
      sentences.append(words)
    return np.array(sentences)

def random_sampling(model_instance, input_data, A=None, f=None):
    '''
    Random sampling policy.
  
    Input:
      model_instance: model
      input_data: the unobserved data.
      n_instances: the number of instances to be sampled in each round.
    Output:
      query_index: the n_instances index of sampled data.
      input_Data[query_index]: the corresponding data.
    '''
    
    #query_index = np.random.choice(range(len(input_data)), size=n_instances, replace=False)
    
    permutation_list = np.random.permutation(len(input_data))
    query_index = []
    temp_tokenum = 0
    for i in permutation_list:
        temp_tokenum += len(input_data[i].label)
        if temp_tokenum <= A:
            query_index.append(i)
        else:
            break
    
    #query_index = np.sort(query_index).tolist()
    #import pdb
    #pdb.set_trace()
    str_query_index = [str(i) for i in query_index]
    if f:
        f.write('*************************************'+' '.join(str_query_index) + '\n')
        for j in query_index:
            f.write(input_data[i].text_a+'\n')
            f.write(' '.join(input_data[i].label)+'\n')
            f.write('\n')
        f.write('************************************'+'\n')
    
    
    return query_index, input_data[query_index]

def active_train(data_loader=None, model=None, Epochs=5, soft_loader=None, args=None, debug=None):
    
    set_seed(args.seed)
    config = BertConfig.from_pretrained(args.bert_model, num_labels=num_labels, finetuning_task=args.task_name)
    if model==None:
      model = Ner.from_pretrained(args.bert_model, from_tf = False, config = config)
    #return_model = Ner.from_pretrained(args.bert_model, from_tf = False, config = config)
    return_model = model
    model.to(device)
    return_model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias','LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    num_train_optimization_steps = int(len(data_loader.dataset)/args.train_batch_size/args.gradient_accumulation_steps)*Epochs #2190
    warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optimization_steps)
    
    current_train_size = 0
    if soft_loader:
      current_train_size = len(data_loader.dataset) + len(soft_loader.dataset)
    else:
      current_train_size = len(data_loader.dataset)
    print('Training on {} data'.format(current_train_size))
  
    model.train()
    tr_loss = 2020
    for epoch_idx in trange(int(Epochs), desc="Epoch"):
      current_loss = 0
      nb_tr_examples, nb_tr_steps = 0, 0
      for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
          batch = tuple(t.to(device) for t in batch)
          input_ids, input_mask, segment_ids, label_ids, valid_ids,l_mask = batch
          loss = model(input_ids, segment_ids, input_mask, label_ids, valid_ids, l_mask, debug=debug)
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
'''
def active_eval(active_data_loader=None, model=None):
    model.to(device)
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    y_true = []
    y_pred = []
    raw_logits = []
    turncate_list = []
    label_map = {i : label for i, label in enumerate(label_list,1)}
    for input_ids, input_mask, segment_ids, label_ids,valid_ids,l_mask in tqdm(active_data_loader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        valid_ids = valid_ids.to(device)
        label_ids = label_ids.to(device)
        l_mask = l_mask.to(device)

        with torch.no_grad():
          logits = model(input_ids, segment_ids, input_mask,valid_ids=valid_ids,attention_mask_label=l_mask)
        
        logits = F.softmax(logits, dim=2)
        assert logits.shape[0] == 1
        logits = logits.detach().cpu().numpy().reshape((logits.shape[1], logits.shape[2]))
        turncate_len = np.count_nonzero(l_mask.detach().cpu().numpy())
        turncate_list.append(turncate_len)
        raw_logits.append(logits)
    return raw_logits, turncate_list
'''
def active_eval(active_data_loader=None, model=None):
    model.to(device)
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    y_true = []
    y_pred = []
    raw_logits = []
    turncate_list = []
    logits_temp_list = []
    label_map = {i : label for i, label in enumerate(label_list,1)}
    for input_ids, input_mask, segment_ids, label_ids,valid_ids,l_mask in tqdm(active_data_loader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        valid_ids = valid_ids.to(device)
        label_ids = label_ids.to(device)
        l_mask = l_mask.to(device)

        with torch.no_grad():
          logits,_ = model(input_ids, segment_ids, input_mask,valid_ids=valid_ids,attention_mask_label=l_mask)
        
        logits = F.softmax(logits, dim=2)
        logits_temp = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
        
        #import pdb
        #pdb.set_trace()
        
        logits_temp = logits_temp.detach().cpu().numpy()
        #assert logits.shape[0] == 1
        logits = logits.detach().cpu().numpy()#.reshape((logits.shape[1], logits.shape[2]))
        
        for nn in l_mask:
        
        #turncate_len = np.count_nonzero(l_mask.detach().cpu().numpy()) # l_mask [1, 128]
        #turncate_list.append(turncate_len) #turncate_len int
            turncate_len = np.count_nonzero(nn.view(1, -1).detach().cpu().numpy()) # l_mask [1, 128]
            turncate_list.append(turncate_len) #turncate_len int
        
        for nn in logits:
        #raw_logits.append(logits) #logits [128, 6]
            raw_logits.append(nn)
        
        for nn in logits_temp:
        #logits_temp_list.append(logits_temp[0]) # logtis_temp[0] [128,]
            logits_temp_list.append(nn[1:-1])
        
        #import pdb
        #pdb.set_trace()
        
    return raw_logits, turncate_list, logits_temp_list



def active_learn_with_tokenum(init_flag=None, train_data=None, num_initial=0.02, 
                 active_policy=None, dev_data=None, fit_only_new_data=False, Epochs=10, in_Epochs=3, prefix='Active', A=None, args=None):
    '''
    Implement active learning initializaiton and learning loop
    '''
    func_paras = locals()
    # Data Initialization
    pool = copy.deepcopy(train_data)
    pool_two_stage = copy.deepcopy(train_data)
    
    train_data = copy.deepcopy(train_data)
    original_datasize = len(train_data)
    all_tokens = 0
    for i in train_data:
        all_tokens += len(i.label)
    
    initial_tokenum = int(num_initial * all_tokens)
    #import pdb
    #pdb.set_trace()
    
    permutation_list = np.random.permutation(original_datasize)
    initial_idx = []
    temp_tokenum = 0
    for i in permutation_list:
        temp_tokenum += len(train_data[i].label)
        if temp_tokenum <= initial_tokenum:
            initial_idx.append(i)
        else:
            break
    
    #initial_idx = np.random.choice(range(len(train_data)), size=num_initial, replace=False)
    train_data = np.array(train_data)[initial_idx]
    
    fw = open(os.path.join(args.output_dir_ana, args.query_file), 'a')
    #import pdb
    #pdb.set_trace()
  
  
    init_data_loader,_,_ = get_tr_set(train_examples=train_data, args=args, type_='init')
    
    pool = np.delete(pool, initial_idx, axis=0)
    pool_two_stage = np.delete(pool_two_stage, initial_idx, axis=0)
    
    print(np.array(pool).shape)
    
    if init_flag:
      init_dir = 'init_dir'
      model = Ner.from_pretrained(init_dir)
    else:
      model = active_train(init_data_loader, None, Epochs, args=args)
  
    report = evaluate('Intialization', model, args)
    print_table = PrettyTable(['Model', 'Number of Query', 'Data Usage', 'Test_F1'])
    print_table.add_row(['Active Model', 'Model Initialization', len(train_data)/original_datasize, report.split()[-2]])
    print(print_table)
    save_result(prefix=args.prefix, report=report, table=print_table, output_dir=args.output_dir)
  
    print('Learning loop start')
    #for idx in range(num_query):
    idx = 0
    f1_indicator = -1
    num_stop = 0
    
    
    ori_len = len(train_data)
    temp_start_end = []
    #temp_predicted_label = []
    query_all = []
    start_end_all = []
    
    #first stage for querying
    
    iterative_len = len(train_data)
    
    while(1):
        print('\n\n-------One Stage Query no. %d--------\n' % (idx + 1))
        query_idx, query_instance, start_end, predicted_label = active_policy(model, pool, A=A, f=fw, min_len=args.min_len)
        #if my_kk > 3:
        #    break
        query_all.extend(query_idx)
        start_end_all.extend(start_end)
        
        if fit_only_new_data:
          train_data = pool[query_idx]
          ori_len = -1
        else:
          train_data = np.concatenate((train_data, pool[query_idx]))
        
        #if temp_start_end !=[]:
        #    import pdb
        #    pdb.set_trace()
        #    start_end_ = temp_start_end.extend(start_end) 
          ##just copy the selected data double for simple data augmentation
          #TODO
          #pool_copy = copy.deepcopy(pool[query_idx])
          #train_data = np.concatenate((train_data, pool[query_idx]))
        
        #start_end_ = start_end
        
        temp_start_end.extend(start_end)
        
        pool = np.delete(pool, query_idx, axis=0)
        
        
        #import pdb
        #pdb.set_trace()
        
        active_data_loader,_,_ = get_tr_set(train_examples=train_data, args=args, type_=None, start_end=temp_start_end, ori_len=ori_len, predicted_label=predicted_label, iterative_len=iterative_len, do_sampling=args.do_sampling)
        
        #import pdb
        #pdb.set_trace()
        
        model = active_train(active_data_loader, model, in_Epochs, args=args, debug=True)
  
        report = evaluate('Active Learning', model, args)
        f1 = eval(report.split()[-2])
        #import pdb
        #pdb.set_trace()
        if f1 < f1_indicator:
            num_stop += 1
        else:
            f1_indicator = f1
        
        #report.split()[-2]
        print_table.add_row(['Active Model', idx+1, len(train_data)/original_datasize, report.split()[-2]])
        print(print_table)
        
        save_result(prefix=args.prefix, func_paras=func_paras, report=report, table=print_table, output_dir=args.output_dir)
        idx += 1
        
        #break
        #if num_stop == 2:# or num_stop == 2:
        #    break
        if len(pool) == 0 or idx==30:
            break
        
        iterative_len = len(train_data)
        #if idx == 15:
        #    break
        
    #fw.close()
    
    '''
    query_start_end = {}
    for i, j in zip(query_all, start_end_all):
        query_start_end[i] = j
    
    ##two stage
    #idx = 0
    while(1):
        print('\n\n-------Two Stage Query no. %d--------\n' % (idx + 1))
        query_idx, query_instance = uncertainsample_two_stage(model, pool_two_stage, A=A, query_start_end=query_start_end, f=fw)
        #if my_kk > 3:
        #    break
        
        
        if fit_only_new_data:
          train_data = pool_two_stage[query_idx]
        else:
          train_data = np.concatenate((train_data, pool_two_stage[query_idx]))
        
        
        pool_two_stage = np.delete(pool_two_stage, query_idx, axis=0)
        active_data_loader,_,_ = get_tr_set(train_examples=train_data, args=args, type_=None)
        
        #import pdb
        #pdb.set_trace()
        
        model = active_train(active_data_loader, model, in_Epochs, args=args)
  
        report = evaluate('Active Learning', model, args)
        f1 = eval(report.split()[-2])
        #import pdb
        #pdb.set_trace()
        if f1 < f1_indicator:
            num_stop += 1
        else:
            f1_indicator = f1
        
        #report.split()[-2]
        print_table.add_row(['Active Model', idx+1, len(train_data)/original_datasize, report.split()[-2]])
        print(print_table)
        
        save_result(prefix=args.prefix, func_paras=func_paras, report=report, table=print_table, output_dir=args.output_dir)
        idx += 1
        #if len(pool_two_stage) == 0:
        #    break
        if idx == 15:
            break
    f.close()
    '''
    
    return model



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--bert_model", type=str, default='bert-base-cased')
    #parser.add_argument("--bert_model", type=str, default='')
    parser.add_argument("--bert_model", type=str, default='')
    parser.add_argument("--data_dir", type=str, default='./data/')
    #parser.add_argument("--data_dir", type=str, default='data/')
    
    parser.add_argument("--output_dir", type=str, default='result/')
    parser.add_argument("--prefix", type=str, default='temp')
    parser.add_argument("--query_file", type=str, default='entity_recall.txt')
    
    parser.add_argument("--output_dir_ana", type=str, default='result_ana/')
    
    #parser.add_argument("--train_data", type=str, default='train_dev_new.txt')
    #parser.add_argument("--dev_data", type=str, default='test_new.txt')
    #parser.add_argument("--test_data", type=str, default='test_new.txt')
    
    
    parser.add_argument("--train_data", type=str, default='train.txt')
    parser.add_argument("--dev_data", type=str, default='test.txt')
    parser.add_argument("--test_data", type=str, default='test.txt')
    
    parser.add_argument("--do_eval", type=bool, default=True)
    parser.add_argument("--do_train", type=bool, default=True)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--task_name", type=str, default='ner')
    
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    
    parser.add_argument("--active_policy", type=str, default='nte')
    parser.add_argument("--full_train_output_dir", type=str, default='full_data_output/')

    # keep as default
    parser.add_argument("--server_ip", type=str, default='')
    parser.add_argument("--server_port", type=str, default='')
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--no_cuda", type=bool, default=False)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--do_lower_case", type=bool, default=False)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--learning_rate", type=float, default=5e-05)
    parser.add_argument("--fp16", type=bool, default=False)
    parser.add_argument("--fp16_opt_level", type=str, default='O1')
    parser.add_argument("--eval_on", type=str, default='test')
    parser.add_argument("--eval_batch_size", type=int, default=8)
    
    parser.add_argument("--A", type=int, default=3000)
    
    parser.add_argument("--min_len", type=int, default=0)
    
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    ##TODO
    parser.add_argument("--label_list", type=str, default='', nargs='+')
    parser.add_argument("--do_sampling", action='store_true', default=False)

    args = parser.parse_args()
        # parse args
    logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s-%(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger = logging.getLogger(__name__)
    if args.server_ip and args.server_port:
      # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
      print("Waiting for debugger attach")
      ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
      ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
      device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
      #n_gpu = torch.cuda.device_count()
      n_gpu = 1
    else:
      torch.cuda.set_device(args.local_rank)
      device = torch.device("cuda", args.local_rank)
      n_gpu = 1
      # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
      torch.distributed.init_process_group(backend='nccl')

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
      raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(args.gradient_accumulation_steps))
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    
    ##SEED
    set_seed(args.seed)

    if not args.do_train and not args.do_eval:
      raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    if not os.path.exists(args.output_dir):
      os.makedirs(args.output_dir)

    processor = NerProcessor()
    #TODO
    #label_list = processor.get_labels()
    labels = args.label_list
    label_list = []
    label_list.append('O')
    for label in labels:
        label_list.append('B-' + label)
        label_list.append('I-' + label)
    label_list.append('[CLS]')
    label_list.append('[SEP]')
    
    num_labels = len(label_list) + 1
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = 0 
    if args.do_train:
      train_examples = processor.get_train_examples(args.data_dir, args.train_data)
      num_train_optimization_steps = int(len(train_examples)/args.train_batch_size/args.gradient_accumulation_steps)*args.num_train_epochs
      if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    if args.do_eval:
      if args.eval_on == 'dev':
        dev_examples = processor.get_dev_examples(args.data_dir, args.dev_data)
      if args.eval_on == 'test':
        dev_examples = processor.get_test_examples(args.data_dir, args.test_data)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # prepare model
    config = BertConfig.from_pretrained(args.bert_model, num_labels=num_labels, finetuning_task=args.task_name)
    model = Ner.from_pretrained(args.bert_model, from_tf = False, config = config)

    if args.local_rank == 0:
      torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias','LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optimization_steps)

    # For our experiment, the following can be ignored
    if args.fp16:
      try:
        from apex import amp
      except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
      model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
      model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
      model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    if args.active_policy=='random':
      active_policy = random_sampling
    if args.active_policy=='mnlp':
      active_policy = uncertainty_sampling
    if args.active_policy=='nte':
      active_policy = nte_sampling
    if args.active_policy=='entropy':
      active_policy = entropy_base
    if args.active_policy=='margin':
      active_policy = margin_base
    
    ##with supervised model with full training data
    
    model = active_learn_with_tokenum(init_flag=False, train_data=train_examples, dev_data=dev_examples, active_policy=active_policy, prefix=args.prefix, Epochs=args.num_train_epochs, A=args.A, args=args)
