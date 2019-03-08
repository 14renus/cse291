import pandas as pd
import math
import time
import torch
import random
import os

from dataloading import *
from model import *
from utils import *

### SET UP ###
data_dir = 'data/numerical_data_set_simple_torch'
filenames=[]
filenames_by_type = {'A':[], 'B':[], 'C':[], 'D':[], 'E':[]}
for file in os.listdir(data_dir):
    filename, file_extension = os.path.splitext(file)
    
    typ = filename[-1]
    if typ in filenames_by_type:
        filenames.append(file)
        filenames_by_type[typ].append(file)
        
## MODIFY JUST FOR TEST, COMMENT OUT FOR REAL RUNS ###
for typ in filenames_by_type:
    filenames_by_type[typ]=[filenames_by_type[typ][0]]

computing_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_digits =10
n_chars=256

test_types = ['A','B','C','D','E']

config = {
        'epochs':100,
        'N_early_stop':10,
        'batch_size':512,
        'learning_rate':0.001,
        'weight_decay':0,
        'teacher_forcing_ratio':1.0,
        'hidden_dim':512,
        'n_layers':4, 
        'enc': {
            'hid_dropout':0.0,
            'input_dropout':0.0
        },
        'dec': {
            'hid_dropout':0.0,
            'input_dropout':0.0
        },

        'input_dim':n_chars+4,
        'output_dim':n_digits+5,

        'verbose':True
    }

if config['verbose']:
    print(filenames_by_type)
 
### K FOLD
for test_type in test_types:
    if config['verbose']:
        print('Fold {}'.format(test_type))
    
    ### GET DATA ###
    train_inputs, train_targets, val_inputs, val_targets = split_data(filenames_by_type,test_type, BATCH_SIZE=config['batch_size'])
    
    total_min_val_loss=100
    min_config=None
    
    for lr in [.001]:
        for bs in [512]:
            for wd in [0]:
                for tr in [1.0]:
                    for idrop in [0.0]:
                        for hdrop in [0.0]:
                            config['batch_size']=bs
                            config['learning_rate']=lr
                            config['weight_decay']=wd
                            config['teacher_forcing_ratio']=tr
                            config['enc']['hid_dropout']=hdrop
                            config['enc']['input_dropout']=idrop
                            config['dec']['hid_dropout']=hdrop
                            config['dec']['input_dropout']=idrop

                            ### TRAIN AND VALIDATE ###
                            min_val_loss,min_epoch,curr_config = train_and_validate(config,test_type, train_inputs, train_targets, val_inputs, val_targets, N=5)
                            if config['verbose']:
                                print('min val loss: {}, min epoch: {}'.format(min_val_loss,min_epoch))
                            if min_val_loss < total_min_val_loss:
                                min_config=curr_config
                                total_min_val_loss=min_val_loss
    ### TEST ON BEST MODEL ###
    test_inputs, test_targets = get_test_data(filenames_by_type,test_type, BATCH_SIZE=config['batch_size'])  
    model = init_seq2seq(min_config, computing_device)
    output_dir='hd={}_nl={}'.format(config['hidden_dim'],config['n_layers'])
    file = 'bs={}_lr={}_wd={}_tf={}_hd={}_id={}_fold={}'.format(config['batch_size'],config['learning_rate'],config['weight_decay'],config['teacher_forcing_ratio'],config['enc']['hid_dropout'],config['enc']['input_dropout'],test_type)
    PATH = "./output/{}/{}.pt".format(output_dir,file)
    model.load_state_dict(torch.load(PATH))
    optimizer = optim.Adam(model.parameters(), lr=min_config['learning_rate'],weight_decay=min_config['weight_decay'])
    criterion = nn.CrossEntropyLoss(ignore_index=output_pad_index)
    test_loss = validate(model, test_inputs, test_targets, optimizer, criterion, computing_device)

    if config['verbose']:
        print('min test loss: {}'.format(test_loss))
                                

    