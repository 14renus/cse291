#https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb

import torch
from torch import nn
from torch import optim
import random
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, hid_dropout, input_dropout):
        super().__init__()
        
        self.input_dim = input_dim
        #self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        #self.dropout = dropout
        
        #self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout=hid_dropout)
        
        self.dropout = nn.Dropout(input_dropout)
        
    def forward(self, src):
        
        #src = [src sent len, batch size]
        
        #embedded = self.dropout(self.embedding(src))
        
        #embedded = [src sent len, batch size, emb dim]
        #print('forward encoder')
        #print(src.size())
        #src = src.view(src.shape())
        #print(src.size())
        
        outputs, (hidden, cell) = self.rnn(self.dropout(src))
        
        #outputs = [src sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        
        return hidden, cell
    
class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, hid_dropout, input_dropout):
        super().__init__()

        #self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        #self.dropout = dropout
        
        #self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.LSTM(output_dim, hid_dim, n_layers, dropout=hid_dropout)
        
        self.out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(input_dropout)
        
        
    def forward(self, src, hidden, cell):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        #print('forward decoder')
        
        #print(input.size())
        
        src = src.unsqueeze(0)
        
        #print(input.size())
        
        #input = [1, batch size]
        
        
        
        #embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
        src = self.dropout(src)
        output, (hidden, cell) = self.rnn(src, (hidden, cell))
        
        #output = [sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #sent len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        prediction = self.out(output.squeeze(0))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden, cell
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder,computing_device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = computing_device
        
        assert encoder.hid_dim == decoder.hid_dim, "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        
        #src = [src sent len, batch size]
        #trg = [trg sent len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size)#.to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        
        #first input to the decoder is the <start> token
        input = trg[0,:]
        
        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            
         
            # softmax + find 
            #print('size of output + pred')
            #print(output.size())
            #pred_logits = torch.softmax(output, dim=1)
            pred_logits = F.softmax(output, dim=1)
            #print(pred.size())
            pred_max = torch.argmax(pred_logits, dim=1)
            pred = torch.zeros(pred_logits.size())
            pred[pred_max] = 1.0
            pred = pred.to(self.device)
            #top1 = output.max(1)[1]
            
            input = (trg[t] if teacher_force else pred)
        
        return outputs