import torch
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