# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

import numpy as np
from numpy import unravel_index
import pandas as pd
import math
from datasets import load_dataset

import random
import sys
from IPython.display import Image
import time

# for text preprocessing
import re
import string

# !CUBLAS_WORKSPACE_CONFIG=:4096:2 # for cuda deterministic behavior

######### BERT ############
# first install transformers from hugging face
# !pip install transformers

# imports
from transformers import BertTokenizer, BertForQuestionAnswering

# dataloaders 
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

def set_seed(seed = 1234):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(False)
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed()

# Set device
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

print('Working on:', device)



train_dataset = load_dataset('squad_v2', split='train')
validation_dataset = load_dataset('squad_v2', split='validation')


df = pd.DataFrame(train_dataset)


def find_end(example):

    if (len(example['answers']['text']) != 0):
        context = example['context']
        text = example['answers']['text'][0]
        start_idx = example['answers']['answer_start'][0]

        end_idx = start_idx + len(text)
        
        temp = example['answers'] # to change the value
        temp['answer_end']=end_idx 
        temp['answer_start'] = start_idx # [num]->num
        temp['text'] = text # ['text']->text
    
    else:
        temp = example['answers']
        temp['answer_end'] = 0 # []->0
        temp['answer_start'] = 0 # []->0
        temp['text'] = "" # []->""
        
    return example

train_dataset = train_dataset.map(find_end)



from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

tokenized_train = tokenizer(train_dataset['context'], train_dataset['question'], truncation=True, padding=True)


def find_token_indexes(tokenized, dataset):
    start_token_list = []
    end_token_list = []
    answers = dataset['answers']
    for i in range(len(answers)):
        if (answers[i]['text'] != ''):
            start_token = tokenized.char_to_token(i, answers[i]['answer_start'])
            end_token = tokenized.char_to_token(i, answers[i]['answer_end'] - 1)
            
            # if start token is None, the answer passage has been truncated
            if start_token is None:
                start_token = tokenizer.model_max_length
            if end_token is None:
                end_token = tokenizer.model_max_length
        else:
            start_token = 0
            end_token = 0
            
        start_token_list.append(start_token)
        end_token_list.append(end_token)

    return start_token_list, start_token_list
    
s, e = find_token_indexes(tokenized_train, train_dataset)
train_dataset = train_dataset.add_column("start_position", s)
train_dataset = train_dataset.add_column("end_position", e)



batch_size = 8
train_data = TensorDataset(torch.tensor(tokenized_train['input_ids'], dtype=torch.int64), 
                           torch.tensor(tokenized_train['token_type_ids'], dtype=torch.int64), 
                           torch.tensor(tokenized_train['attention_mask'], dtype=torch.float), 
                           torch.tensor(train_dataset['start_position'], dtype=torch.int64), 
                           torch.tensor(train_dataset['start_position'], dtype=torch.int64))

train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)


tokenized_validation = tokenizer(validation_dataset['context'], validation_dataset['question'], truncation=True, padding=True, return_offsets_mapping=True)


batch_size = 8
val_data = TensorDataset(torch.tensor(tokenized_validation['input_ids'], dtype=torch.int64), 
                        torch.tensor(tokenized_validation['token_type_ids'], dtype=torch.int64), 
                        torch.tensor(tokenized_validation['attention_mask'], dtype=torch.float))
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)


# fine tuning 
# model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
epochs = 3
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

model.load_state_dict(torch.load("../input/bert-weights/bert2_2.h5"))


from tqdm import tqdm

for epoch in range(epochs):
    epoch_loss = []
    validation_loss = []
    
    total_loss = 0
    model.train()

    count=-1
    progress_bar = tqdm(train_dataloader, leave=True, position=0)
    progress_bar.set_description(f"Epoch {epoch+1}")
    for batch in progress_bar:
        count+=1
        input_ids, segment_ids, mask, start, end  = tuple(t.to(device) for t in batch)

        model.zero_grad()
        loss, start_logits, end_logits = model(input_ids = input_ids, 
                                                token_type_ids = segment_ids, 
                                                attention_mask = mask, 
                                                start_positions = start, 
                                                end_positions = end,
                                                return_dict = False)           

        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if (count % 20 == 0 and count != 0):
            avg = total_loss/count
            progress_bar.set_postfix(Loss=avg)
            
    torch.save(model.state_dict(), "./bert2_" + str(epoch) + ".h5") # save for later use
    avg_train_loss = total_loss / len(train_dataloader)
    epoch_loss.append(avg_train_loss)
    print(f"Epoch {epoch} Loss: {avg_train_loss}\n")



from tqdm import tqdm
# model.load_state_dict(torch.load("../input/bert-weights/bert2_2.h5"))

threshold = 1.0
epoch_i = 0
correct = 0 
pred_dict = {}
na_prob_dict = {}

model.eval()
correct = 0
batch_val_losses = []
row = 0
for test_batch in tqdm(val_dataloader):
    input_ids, segment_ids, masks = tuple(t.to(device) for t in test_batch)

    with torch.no_grad():
        # prediction logits
        start_logits, end_logits = model(input_ids=input_ids,
                                        token_type_ids=segment_ids,
                                        attention_mask=masks,
                                        return_dict=False)

    # to cpu
    start_logits = start_logits.detach().cpu()
    end_logits = end_logits.detach().cpu()

    # for every sequence in batch 
    for bidx in range(len(start_logits)):
        # apply softmax to logits to get scores
        start_scores = np.array(F.softmax(start_logits[bidx], dim = 0))
        end_scores = np.array(F.softmax(end_logits[bidx], dim = 0))

        # find max for start<=end
        size = len(start_scores)
        scores = np.zeros((size, size))

        for j in range(size):
            for i in range(j+1): # include j
                scores[i,j] = start_scores[i] + end_scores[j]

        # find best i and j
        start_pred, end_pred = unravel_index(scores.argmax(), scores.shape)
        answer_pred = ""
        if (scores[start_pred, end_pred] > scores[0,0]+threshold):

            offsets = tokenized_validation.offset_mapping[row]
            pred_char_start = offsets[start_pred][0]

            if end_pred < len(offsets):
                pred_char_end = offsets[end_pred][1]
                answer_pred = validation_dataset[row]['context'][pred_char_start:pred_char_end]
            else:
                answer_pred = validation_dataset[row]['context'][pred_char_start:]

            if answer_pred in validation_dataset[row]['answers']['text']:
                correct += 1

        else:
            if (len(validation_dataset[row]['answers']['text']) ==0):
                correct += 1    

        pred_dict[validation_dataset[row]['id']] = answer_pred
        na_prob_dict[validation_dataset[row]['id']] = scores[0,0]

        row+=1


accuracy = correct/validation_dataset.num_rows
print("accuracy is: ", accuracy)



import json 
with open("pred.json", "w") as outfile:
    json.dump(pred_dict, outfile)


with open("na_prob.json", "w") as outfile:
    json.dump(na_prob_dict, outfile)