import pandas as pd
import math
import time
import torch
import random
import os

from dataloading import *
from model import *

def train(model, inputs, targets, optimizer, criterion, computing_device, config):
    model.train()
    epoch_loss = 0
    
    for i in range(len(targets)):
        
        src = inputs[i].to(computing_device)
        trg = targets[i].to(computing_device)
        
        optimizer.zero_grad()
        
        outputs = model(src, trg, teacher_forcing_ratio=config['teacher_forcing_ratio'])
        
        labels = torch.argmax(trg, dim=2) # grab indices for loss function
        
        #targets = [trg sent len, batch size]
        #outputs = [trg sent len, batch size, output dim]
        
        outputs = outputs.view(-1, outputs.shape[-1]) 
        labels = labels.view(-1)
        
        outputs = outputs.to(computing_device)
        
        #targets = [(trg sent len - 1) * batch size]- trg should be list of indicies
        #outputs = [(trg sent len - 1) * batch size, output dim]
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(targets)

def validate(model, inputs, targets, optimizer, criterion, computing_device):
    model.eval()
    total_loss=0.0
    
    with torch.no_grad():   
        for i in range(len(targets)):
                src = inputs[i].to(computing_device)
                trg = targets[i].to(computing_device)

                outputs = model(src, trg, teacher_forcing_ratio=0.0)

                labels = torch.argmax(trg, dim=2) # grab indices for loss function

                outputs = outputs.view(-1, outputs.shape[-1]) 
                labels = labels.view(-1)

                outputs = outputs.to(computing_device)

                loss = criterion(outputs, labels)
                total_loss+=loss.item()

    return total_loss/len(targets)

def init_seq2seq(config, computing_device):
    enc = Encoder(config['input_dim'], config['hidden_dim'], config['n_layers'], config['enc']['hid_dropout'], config['enc']['input_dropout'])
    dec = Decoder(config['output_dim'], config['hidden_dim'], config['n_layers'], config['dec']['hid_dropout'], config['dec']['input_dropout'])

    model = Seq2Seq(enc, dec,computing_device)#.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'],weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss(ignore_index=output_pad_index)
    
    model = model.to(computing_device)
    return model

def split_data(filenames_by_type,test_type, train_frac=0.75, BATCH_SIZE=512, data_dir='data/numerical_data_set_simple_torch'):
    print('...loading data')
    if test_type != 'A':
        init='A'
    else:
        init='B'
    filename=filenames_by_type[init][0]
    q = torch.load(os.path.join(data_dir,filename))
    inputs,targets = q[0],q[1]

    for typ in filenames_by_type:
        if typ==test_type:
            continue
        if typ==init:
            for filename in filenames_by_type[typ][1:]:
                q = torch.load(os.path.join(data_dir,filename))
                src,trg = q[0],q[1]
                inputs=torch.cat([inputs,src],dim=1)
                targets=torch.cat([targets,trg],dim=1)
        else:
            for filename in filenames_by_type[typ]:
                q = torch.load(os.path.join(data_dir,filename))
                src,trg = q[0],q[1]
                inputs=torch.cat([inputs,src],dim=1)
                targets=torch.cat([targets,trg],dim=1)
    
    #shuffle indices
    indices = list(range(targets.size()[1]))

    random.shuffle(indices)
    
    inputs = inputs[:,indices,:]
    targets = targets[:,indices,:]

    # chunk
    n_chunks = int(math.ceil(inputs.size()[1]/BATCH_SIZE))
    inputs = torch.chunk(inputs, n_chunks, dim=1) 
    targets = torch.chunk(targets, n_chunks, dim=1) 
    
    # split train and val
    i=int(train_frac*len(inputs))
    train_inputs = inputs[:i]
    val_inputs = inputs[i:]
    
    train_targets = targets[:i]
    val_targets = targets[i:]
    
    return train_inputs, train_targets, val_inputs, val_targets

def read_data(data_dir,lim,filename):
    df = pd.read_csv(os.path.join(data_dir,filename))

    values = df['Attribute_value']
    targets = df['Numerical_value']

    #lim = LIM//len(filenames_by_type[typ])

    inputs = prepare_data(values[lim:],padding_len=22)
    outputs = prepare_targets(targets[lim:],padding_len=22)
    
    return inputs,outputs

def encode_and_split_data(filenames_by_type,test_type, LIM=500,train_frac=0.75, BATCH_SIZE=512, data_dir='data/numerical_data_set_simple'):
    print('...loading data')
    if test_type != 'A':
        init='A'
    else:
        init='B'
    filename=filenames_by_type[init][0]
    
    lim = LIM//len(filenames_by_type[init])
    inputs,targets = read_data(data_dir,lim,filename)

    for typ in filenames_by_type:
        lim = LIM//len(filenames_by_type[init])
        if typ==test_type:
            continue
        if typ==init:
            for filename in filenames_by_type[typ][1:]:
                src,trg = read_data(data_dir,lim,filename)
                inputs=torch.cat([inputs,src],dim=1)
                targets=torch.cat([targets,trg],dim=1)
        else:
            for filename in filenames_by_type[typ]:
                src,trg = read_data(data_dir,lim,filename)
                inputs=torch.cat([inputs,src],dim=1)
                targets=torch.cat([targets,trg],dim=1)
    
    #shuffle indices
    indices = list(range(targets.size()[1]))

    random.shuffle(indices)
    
    inputs = inputs[:,indices,:]
    targets = targets[:,indices,:]

    # chunk
    n_chunks = int(math.ceil(inputs.size()[1]/BATCH_SIZE))
    inputs = torch.chunk(inputs, n_chunks, dim=1) 
    targets = torch.chunk(targets, n_chunks, dim=1) 
    
    # split train and val
    i=int(train_frac*len(inputs))
    train_inputs = inputs[:i]
    val_inputs = inputs[i:]
    
    train_targets = targets[:i]
    val_targets = targets[i:]
    
    return train_inputs, train_targets, val_inputs, val_targets
    
def get_test_data(filenames_by_type,test_type, BATCH_SIZE=512,data_dir='data/numerical_data_set_simple_torch'):
    filename=filenames_by_type[test_type][0]
    q = torch.load(os.path.join(data_dir,filename))
    inputs,targets = q[0],q[1]
    
    for filename in filenames_by_type[test_type][1:]:
                q = torch.load(os.path.join(data_dir,filename))
                src,trg = q[0],q[1]
                inputs=torch.cat([inputs,src],dim=1)
                targets=torch.cat([targets,trg],dim=1)   
    # chunk
    n_chunks = math.ceil(inputs.size()[1]/BATCH_SIZE)
    inputs = torch.chunk(inputs, n_chunks, dim=1) 
    targets = torch.chunk(targets, n_chunks, dim=1) 
    
    return inputs, targets

def train_and_validate(config,test_type, train_inputs, train_targets, val_inputs, val_targets, computing_device, N=5):
    output_dir='hd={}_nl={}'.format(config['hidden_dim'],config['n_layers'])
    print(output_dir)
    output_dir = os.path.join('output',output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = 'bs={}_lr={}_wd={}_tf={}_hd={}_id={}_fold={}'.format(config['batch_size'],config['learning_rate'],config['weight_decay'],config['teacher_forcing_ratio'],config['enc']['hid_dropout'],config['enc']['input_dropout'],test_type)
    print(output_file)
    output_filepath = os.path.join(output_dir,output_file+'.csv')   
    
    model = init_seq2seq(config, computing_device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'],weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss(ignore_index=output_pad_index)

    verbose=config['verbose']
    #train_inputs, train_targets, val_inputs, val_targets = split_data(filenames_by_type,test_type, BATCH_SIZE=config['batch_size'])

    avg_val_loss=0.0
    avg_train_loss=0.0
    min_val_loss=100
    min_epoch=0
    best_state_dict=None
    total_start=time.time()
    for epoch in range(config['epochs']):
        # train 
        if verbose:
            print('...training')
        start=time.time()
        train_loss = train(model, train_inputs, train_targets, optimizer, criterion, computing_device,config)
        if verbose:
            print('   epoch {}: train_loss:{}, time:{}'.format(epoch,train_loss,time.time()-start))

        #validate
        if verbose:
            print('...validating')
        start=time.time()
        val_loss = validate(model, val_inputs, val_targets, optimizer, criterion, computing_device)
        if verbose:
            print('   epoch {}: val_loss:{}, time:{}'.format(epoch,val_loss,time.time()-start))
    
        avg_val_loss+=val_loss
        avg_train_loss+=train_loss
        
        if epoch%N==0:
            avg_val_loss/=N
            avg_train_loss/=N
            
            with open(output_filepath, 'a') as file: 
                file.write('{},{},{}\n'.format(avg_val_loss,avg_train_loss,time.time()-total_start))
                
            # update min, state_dict
            if avg_val_loss<min_val_loss:
                min_val_loss=avg_val_loss
                min_epoch=epoch 
                best_state_dict = model.state_dict()
            # if not decreasing for a while
            elif epoch - min_epoch >= config['N_early_stop']:
                if best_state_dict:
                    PATH = "./output/{}.pt".format(output_file)
                    torch.save(best_state_dict, PATH)
                return min_val_loss
            avg_val_loss=0.0
            avg_train_loss=0.0
            total_start=time.time()
            
    with open(os.path.join(output_dir,'final_output.txt'), 'a') as file: 
        file.write('{},{}\n'.format(output_file,avg_val_loss))
                
    if best_state_dict:
        PATH = os.path.join(output_dir,output_file+'_best.pt') 
        torch.save(best_state_dict, PATH)
    else:
        PATH = os.path.join(output_dir,output_file+'_final.pt') 
        torch.save(best_state_dict, PATH)
    
    return min_val_loss, min_epoch, config
