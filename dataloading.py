import pandas as pd
import os
import numpy as np
import torch

n_chars = 256
INPUT_DIM = n_chars+4 # unknown, start, end, padding

input_pad_index=INPUT_DIM-1
input_start_index=INPUT_DIM-2
input_end_index=INPUT_DIM-3

input_padding = torch.zeros(1, INPUT_DIM)
input_padding[0][input_pad_index]=1.0

input_ending = torch.zeros(1, INPUT_DIM)
input_ending[0][input_end_index]=1.0

input_starting = torch.zeros(1, INPUT_DIM)
input_starting[0][input_start_index]=1.0

n_digits = 10
OUTPUT_DIM = n_digits + 5
#embedding = torch.nn.Embedding(255,255)

output_pad_index=OUTPUT_DIM-1
output_start_index=OUTPUT_DIM-3
output_end_index=OUTPUT_DIM-2

output_padding = torch.zeros(1, OUTPUT_DIM)
output_padding[0][output_pad_index]=1.0

output_ending = torch.zeros(1, OUTPUT_DIM)
output_ending[0][output_end_index]=1.0

output_starting = torch.zeros(1, OUTPUT_DIM)
output_starting[0][output_start_index]=1.0

### DATA ####
def charToIndex(char):
    if ord(char)<n_chars:
        return ord(char)
    else:
        return n_chars+1 # unknown

# turn a letter into a <1 x n_letters> Tensor
def charToTensor(char):
    tensor = torch.zeros(1, INPUT_DIM)
    tensor[0][charToIndex(char)] = 1
    return tensor

def lineToTensor(line):
    line=str(line)
    tensor = torch.zeros(len(line), INPUT_DIM)
    for li, char in enumerate(line):
        tensor[li][charToIndex(char)] = 1
    return tensor

def pad(values,max_len):
    output = []
    for val in values:
        #print(val.size())
        #print(padding.size())
        o = torch.cat([input_starting]+[val]+[input_ending]+([input_padding] * (max_len-len(val))), dim=0)
        #print(o.size())
        output.append(o)
    return torch.stack(output,dim=1) 

def pad_2(values,max_len):
    output = []
    for val in values:
        print(val.size())
        print(list(val))
        o = torch.cat(list(val)+[torch.tensor(0)] * (max_len-len(val)), dim=0)
        output.append(o)
    return torch.stack(output,dim=1) 

# hot encode - working
def prepare_data(values,padding_len=None):
    embeddings = []
    max_len = 0
    for val in values:
        val=str(val)
        max_len=max(max_len,len(val))
        embeddings.append(lineToTensor(val))
    if padding_len and max_len>padding_len:
        print("WARNING: padding_len {} should be > max_len {}".format(padding_len,max_len))
    if padding_len:
        return pad(embeddings,padding_len)
    return pad(embeddings,max_len)
    
def prepare_data_2(values):
    output = []
    max_len = 0
    for val in values:
        val=str(val)
        max_len=max(max_len,len(val))
        line=[]
        for char in val:
            line.append(charToIndex(char))
        output.append(torch.tensor(line))
        
    return pad(output,max_len)

    #return embeddings   

#### LABELS #####

def digitToIndex(char):
    char=str(char)
    if char.isdigit() and ord(char)<=ord('9') and ord(char)>=ord('0'):
        return int(char)
    elif char=='.':
        return 10 # period
    else:
        return 11
    

def indexToDigit(digit):
    digit = digit.numpy()
    if digit <=9:
        return str(digit)
    elif digit==10:
        return '.' # period
    else:
        return '-'
    
# turn a letter into a <1 x n_letters> Tensor
def digitToTensor(char):
    tensor = torch.zeros(1, OUTPUT_DIM)
    tensor[0][digitToIndex(char)] = 1
    return tensor

def numToTensor(line):
    #print(line)
    line = str(line)
    tensor = torch.zeros(len(line), OUTPUT_DIM)
    for li, char in enumerate(line):
        tensor[li][digitToIndex(char)] = 1
    return tensor

'''
def prepare_targets(values):
    embeddings = []
    for val in values:
        embeddings.append(numToTensor(val))
    return embeddings   
'''
def pad_labels(values,max_len):
    output = []
    for val in values:
        #print(val.size())
        #print(padding.size())
        o = torch.cat([output_starting]+[val]+[output_ending]+([output_padding] * (max_len-len(val))), dim=0)
        #print(o.size())
        output.append(o)
    return torch.stack(output,dim=1) 

def prepare_targets(values,padding_len=None):
    output = []
    max_len = 0
    for val in values:
        val=str(val)
        max_len=max(max_len,len(val))
        output.append(numToTensor(val))
    if padding_len:
        return pad_labels(output,padding_len)
    return pad_labels(output,max_len)