# -*- encoding: utf-8 -*-

import sys
sys.path.insert(0, '../../pytorch-i3d')
import torch.nn as nn
import torch
import model_transformer as tt
from pytorch_i3d import InceptionI3d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SiameseTransformerSANet(nn.Module):
    def __init__(self, ntokens, emsize, nhead, nhid, nlayers, dropout):
        super(SiameseTransformerSANet, self).__init__()
        self.transformer = tt.TransformerModelSA(ntokens, emsize, nhead, nhid, nlayers, dropout)
        

    def forward_once(self, x):
        output = self.transformer(x)
#        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class SiameseI3DNet(nn.Module):
    def __init__(self, out_classes, in_channels=3):
        super(SiameseI3DNet, self).__init__()
        self.i3d = InceptionI3d(out_classes, in_channels=in_channels)
        
    def forward_once(self, x):
        output = self.i3d(x)
#        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class GRUBoWSANet(nn.Module):
    '''For Soft Assignment sequences
    '''
    def __init__(self, input_size, hidden_size, n_layers=1, bidirectional=False):
        super(GRUBoWSANet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_directions = int(bidirectional) + 1
        self.n_layers = n_layers
        self.gru1 = nn.GRU(input_size, hidden_size, n_layers, batch_first=True,
                          bidirectional=bidirectional)
#        self.gru2 = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True,
#                          bidirectional=bidirectional)
        self.fc2 = nn.Linear(hidden_size * self.n_directions, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
#        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, inputs, hidden):
        batch_size = inputs.size(0)
        output, hidden = self.gru1(inputs, hidden)
#        output, hidden = self.gru2(output, hidden)
        output = self.dropout(self.fc2(output[:,-1,:]))
        logits = self.fc3(output)
#        probs = self.softmax(logits.view(-1, self.n_classes))
        return logits, hidden
    
    def init_hidden(self, batch_size):
        enc_h = torch.zeros(self.n_directions * self.n_layers, batch_size, self.hidden_size)
        enc_h = enc_h.to(device)
        return enc_h
    

class SiameseGRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, nlayers, bidirectional=False):
        super(SiameseGRUNet, self).__init__()
        self.g1 = GRUBoWSANet(input_size, hidden_size, nlayers, bidirectional)

    def forward_once(self, x, hidden):
        output, h = self.g1(x, hidden)
#        output = output.view(output.size()[0], -1)
        return output, h

    def forward(self, input1, input2, hidden):
        output1, h1 = self.forward_once(input1, hidden)
        output2, h2 = self.forward_once(input2, hidden)
        return output1, output2, h1, h2
    
    def init_hidden(self, batch_size):
        return self.g1.init_hidden(batch_size)
        
    
    