#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 03:44:58 2020

@author: arpan

@Description: Convolutional Attention Model for Cricket Strokes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class C3DGRUv2(nn.Module):
    """
    The C3D network as described in [1]. No pooling in temporal dimension. Added
    GRU layer at the end.
    """

    def __init__(self, hidden_size, n_layers, output_size=2, bidirectional=False, \
                 use_gpu=True):
        super(C3DGRUv2, self).__init__()
        
        self.use_gpu = use_gpu
        self.hidden_size = hidden_size      # 2048 after conv6
        self.n_layers = n_layers
        self.n_directions = int(bidirectional) + 1
        
        self.output_size = output_size
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
#        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
#        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
#        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))
#        self.conv6 = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
#        self.pool6 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        # each vector size is 2048 after pool6 and reshaping
        self.gru = nn.GRU(8192, self.hidden_size, n_layers, batch_first=True,
                          bidirectional=bidirectional)
#        self.gru1 = nn.GRU(self.hidden_size, self.hidden_size, n_layers, batch_first=True,
#                          bidirectional=bidirectional)
        
#        self.fc6 = nn.Linear(8192, 4096)
#        self.fc7 = nn.Linear(4096, 4096)
#        self.fc8 = nn.Linear(4096, output_size)
        self.fc = nn.Linear(self.hidden_size, output_size)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        if torch.__version__.split('.')[0]=='1':
            self.softmax = nn.Softmax(dim = 1)
        else:
            self.softmax = nn.Softmax()     # PyTorch 0.2 Operates on 2D arrays

    def forward(self, x):
        
        batch, ch, seq, ht, wd = x.size()
        batch, ch, seq, ht, wd = int(batch), int(ch), int(seq), int(ht), int(wd)
        hid_vec = self._init_hidden(batch)
        
        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
#        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
#        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
#        h = self.relu(self.conv5b(h))
        h = self.pool5(h)       # output : [batch, 512, 5, 4, 4]

#        h = self.relu(self.conv6(h))
#        h = self.pool6(h)

#        h = h.view(-1, 8192)
#        h = self.relu(self.fc6(h))
#        h = self.dropout(h)
#        #h = self.fc7(h)
#        h = self.relu(self.fc7(h))
#        h = self.dropout(h)
#
#        logits = self.fc8(h)
#        probs = self.softmax(logits)
        # change blob to BATCH x SEQ x CH x H x W ; SEQs to be sent to GRU
        h = h.permute(0, 2, 1, 3, 4)
        vec = h.contiguous().view(batch, seq, -1)   # similar to reshape(batch, seq, -1) in new version
        
        # To compact weights again call flatten_parameters().
        self.gru.flatten_parameters()
        #output, hidden = self.rnn(embedded, hidden)
        output, hid_vec = self.gru(vec, hid_vec)
#        output, hid_vec = self.gru1(output, hid_vec)
        #return self.fc(output.contiguous().view(-1, self.hidden_size))
        
        #out = self.relu(self.fc(output))
        #logits = self.dropout(out)
        out = self.fc(output)
        probs = self.softmax(out.view(-1, self.output_size))
        
        return probs

    
    def _init_hidden(self, batch_size):
         #* self.n_directions
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, \
                             self.hidden_size)
        hidden = hidden.to(device)
        return hidden
    
class C3DGRUv2Orig(nn.Module):
    """
    The C3D network as described in [1]. No pooling in temporal dimension. Added
    GRU layer at the end.
    """

    def __init__(self, hidden_size, n_layers, output_size=2, bidirectional=False, \
                 use_gpu=True):
        super(C3DGRUv2Orig, self).__init__()
        
        self.use_gpu = use_gpu
        self.hidden_size = hidden_size      # 2048 after conv6
        self.n_layers = n_layers
        self.n_directions = int(bidirectional) + 1
        
        self.output_size = output_size
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))
        self.conv6 = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool6 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        # each vector size is 2048 after pool6 and reshaping
        self.gru = nn.GRU(8192, self.hidden_size, n_layers, batch_first=True,
                          bidirectional=bidirectional)
        self.gru1 = nn.GRU(self.hidden_size, self.hidden_size, n_layers, batch_first=True,
                          bidirectional=bidirectional)
        
#        self.fc6 = nn.Linear(8192, 4096)
#        self.fc7 = nn.Linear(4096, 4096)
#        self.fc8 = nn.Linear(4096, output_size)
        self.fc = nn.Linear(self.hidden_size, output_size)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        if torch.__version__.split('.')[0]=='1':
            self.softmax = nn.Softmax(dim = 1)
        else:
            self.softmax = nn.Softmax()     # PyTorch 0.2 Operates on 2D arrays

    def forward(self, x):
        
        batch, ch, seq, ht, wd = x.size()
        batch, ch, seq, ht, wd = int(batch), int(ch), int(seq), int(ht), int(wd)
        hid_vec = self._init_hidden(batch)
        
        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)       # output : [batch, 512, 5, 4, 4]

#        h = self.relu(self.conv6(h))
#        h = self.pool6(h)

#        h = h.view(-1, 8192)
#        h = self.relu(self.fc6(h))
#        h = self.dropout(h)
#        #h = self.fc7(h)
#        h = self.relu(self.fc7(h))
#        h = self.dropout(h)
#
#        logits = self.fc8(h)
#        probs = self.softmax(logits)
        # change blob to BATCH x SEQ x CH x H x W ; SEQs to be sent to GRU
        h = h.permute(0, 2, 1, 3, 4)
        vec = h.contiguous().view(batch, seq, -1)   # similar to reshape(batch, seq, -1) in new version
        
        # To compact weights again call flatten_parameters().
        self.gru.flatten_parameters()
        #output, hidden = self.rnn(embedded, hidden)
        output, hid_vec = self.gru(vec, hid_vec)
        output, hid_vec = self.gru1(output, hid_vec)
        #return self.fc(output.contiguous().view(-1, self.hidden_size))
        
        #out = self.relu(self.fc(output))
        #logits = self.dropout(out)
        out = self.fc(output)
        probs = self.softmax(out.view(-1, self.output_size))
        
        return probs

    
    def _init_hidden(self, batch_size):
         #* self.n_directions
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, \
                             self.hidden_size)
        hidden = hidden.to(device)
        return hidden

class Conv3DEncoder(nn.Module):
    """
    The C3D Layers defined for clips. No pooling in temporal dimension. Spatial pooling
    applied upto some extent.
    GRU layer at the end.
    """

    def __init__(self, hidden_size, n_layers, bidirectional=False):
        super(Conv3DEncoder, self).__init__()
        
        self.hidden_size = hidden_size      # 2048 after conv6
        self.n_layers = n_layers
        self.n_directions = int(bidirectional) + 1
        
#        self.output_size = output_size
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
#        # each vector size is 2048 after pool6 and reshaping
        self.gru = nn.GRU(128, self.hidden_size, n_layers, batch_first=True,
                          bidirectional=bidirectional)
#        self.gru1 = nn.GRU(self.hidden_size, self.hidden_size, n_layers, batch_first=True,
#                          bidirectional=bidirectional)
        
#        self.fc = nn.Linear(self.hidden_size, output_size)

#        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        if torch.__version__.split('.')[0]=='1':
            self.softmax = nn.Softmax(dim = 1)
        else:
            self.softmax = nn.Softmax()     # PyTorch 0.2 Operates on 2D arrays

    def forward(self, x, hid_vec):
        
        batch, ch, seq, ht, wd = x.size()
        batch, ch, seq, ht, wd = int(batch), int(ch), int(seq), int(ht), int(wd)
        
        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3(h))
        h = self.pool3(h)
        
        # (B, 32, 1, 14, 14)
        h = h.squeeze(2).permute(0, 2, 3, 1)
        #h = h.permute(0, 2, 1, 3, 4)
        h = h.view(batch, -1, h.size(-1)) # No of sequences 14*14
#        hid_vec = self._init_hidden(batch)
        output, hid_vec = self.gru(h, hid_vec)        
        return output, hid_vec
    
    def _init_hidden(self, batch_size):
         #* self.n_directions
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, \
                             self.hidden_size)
        return hidden.to(device)

class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, bidirectional=False, 
                 max_length=10, dropout_p=0.1):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.attn = nn.Linear(self.hidden_size + self.output_size, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size + self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)        
  
    def init_hidden(self, batch_size):
        dec_h = torch.zeros(self.n_directions * self.n_layers, batch_size, self.output_size)
        dec_h = dec_h.to(device)
        return dec_h
  
    def forward(self, decoder_hidden, encoder_outputs, input):
        
        # decoder_hidden (n_directions, BATCH, HIDDEN_SIZE)
        # encoder_outputs (BATCH, SEQ_SIZE, HIDDEN_SIZE)
        # input (BATCH, OUTPUT_SIZE)  :- INPUT_SIZE == OUTPUT_SIZE OR use embedding
        
        # find attention weights (BATCH, SEQ_SIZE)
        attn_weights = F.softmax(self.attn(torch.cat((input, decoder_hidden[0]), 1)), dim=1)
        # if BATCH > 1 or BATCH==1, then unsqueeze(1) for attn_weights
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

#        # Either loop over the batch samples of encoder_outputs or broadcast attn_weights
#        attn_applied = []
#        for encoder_output in encoder_outputs:
#            attn_applied.append(torch.bmm(attn_weights.unsqueeze(0), encoder_output.unsqueeze(0)))            
#        torch.cat(attn_applied, 1)
        
        output = torch.cat((input, attn_applied.squeeze(1)), 1) #(BATCH, HIDDEN+OUTPUT_SIZE)
        output = self.attn_combine(output) #.unsqueeze(0)   (BATCH, HIDDEN_SIZE)

#        output = F.relu(output)
        output, decoder_hidden = self.gru(output.unsqueeze(1), decoder_hidden)

        output = self.out(output.squeeze(1)) # F.log_softmax(self.out(output.squeeze(1)), dim=1)
        return output, decoder_hidden, attn_weights

class Conv3DDecoder(nn.Module):
    """
    The convTranspose3D Layers defined for clips. No unpooling in temporal dimension. 
    Spatial unpooling applied upto some extent.
    GRU layer at the end.
    """

    def __init__(self, hidden_size, output_size, n_layers, max_length=196, 
                 bidirectional=False):
        super(Conv3DDecoder, self).__init__()
        
        self.hidden_size = hidden_size      # 2048 after conv6
        self.n_layers = n_layers
        self.n_directions = int(bidirectional) + 1
        self.output_size = output_size
        self.max_length = max_length
        self.dropout_p = 0.5
        
#        self.output_size = output_size
        self.deconv3 = nn.ConvTranspose3d(128, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.deconv2 = nn.ConvTranspose3d(64, 32, kernel_size=(2, 2, 2), stride=(2, 2, 2))
#        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.deconv1 = nn.ConvTranspose3d(32, 3, kernel_size=(2, 2, 2), stride=(2, 2, 2))
#        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

#        self.gru = nn.GRU(8192, self.hidden_size, n_layers, batch_first=True,
#                          bidirectional=bidirectional)
        self.attn_dec = AttentionDecoder(self.hidden_size, self.hidden_size, self.n_layers, 
                                         bidirectional, self.max_length)

#        self.fc = nn.Linear(self.hidden_size, output_size)
#
#        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        if torch.__version__.split('.')[0]=='1':
            self.softmax = nn.Softmax(dim = 1)
        else:
            self.softmax = nn.Softmax()     # PyTorch 0.2 Operates on 2D arrays

    def forward(self, dec_h, enc_outputs):
        
        dec_out_lst, attn_wts_lst = [], []
        for ti in range(enc_outputs.size(1)):
            #start symbol of dim  (batch x output_size) 
            inp = torch.zeros((dec_h.size(1), self.hidden_size)).to(device)  #starting symbol
            dec_out, dec_h, attn_wts = self.attn_dec(dec_h, enc_outputs, inp)
            dec_out_lst.append(dec_out)
            attn_wts_lst.append(attn_wts)
            
        # reconstruct the output
        dec_outputs = torch.stack(dec_out_lst, dim=1)
        batch, seq, ftsize = dec_outputs.size()
        ht = int(math.sqrt(seq))
        dec_outputs = dec_outputs.reshape(batch, ht, ht, ftsize).permute(0, 3, 1, 2)  # shift C to dim1
        dec_outputs = dec_outputs.unsqueeze(2)
        h = self.relu(self.deconv3(dec_outputs))
        h = self.relu(self.deconv2(h))
#        h = self.pool1(h)
        h = torch.sigmoid(self.deconv1(h))
#        h = self.pool2(h)
        
        return h, attn_wts_lst

    def _init_hidden(self, batch_size):
         #* self.n_directions
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, \
                             self.hidden_size)
        return hidden.to(device)
    
    
class Conv3DDecoderClassifier(nn.Module):
    """
    The convTranspose3D Layers defined for clips. No unpooling in temporal dimension. 
    Spatial unpooling applied upto some extent.
    GRU layer at the end.
    """

    def __init__(self, hidden_size, output_size, n_layers, max_length=196, 
                 bidirectional=False):
        super(Conv3DDecoderClassifier, self).__init__()
        
        self.hidden_size = hidden_size      # 2048 after conv6
        self.n_layers = n_layers
        self.n_directions = int(bidirectional) + 1
        self.output_size = output_size
        self.max_length = max_length
        self.dropout_p = 0.5
        
#        self.gru = nn.GRU(8192, self.hidden_size, n_layers, batch_first=True,
#                          bidirectional=bidirectional)
        self.attn_dec = AttentionDecoder(self.hidden_size, self.hidden_size, self.n_layers, 
                                         bidirectional, self.max_length)

        self.fc = nn.Linear(self.hidden_size, output_size)

#        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        if torch.__version__.split('.')[0]=='1':
            self.softmax = nn.Softmax(dim = 1)
        else:
            self.softmax = nn.Softmax()     # PyTorch 0.2 Operates on 2D arrays

    def forward(self, dec_h, enc_outputs):
        
        dec_out_lst, attn_wts_lst = [], []
#        for ti in range(enc_outputs.size(1)):
            #start symbol of dim  (batch x output_size) 
        inp = torch.zeros((dec_h.size(1), self.hidden_size)).to(device)  #starting symbol
        # dec_out : (B, 128), dec_h: (1, B, 128), attn_wts: (B, 196)
        dec_out, dec_h, attn_wts = self.attn_dec(dec_h, enc_outputs, inp)
        dec_out_lst.append(dec_out)
#        attn_wts_lst.append(attn_wts)
            # for ti ends
        out = self.softmax(self.fc(dec_h.squeeze(0)))
#        # reconstruct the output
#        dec_outputs = torch.stack(dec_out_lst, dim=1)
#        batch, seq, ftsize = dec_outputs.size()
#        ht = int(math.sqrt(seq))
#        dec_outputs = dec_outputs.reshape(batch, ht, ht, ftsize).permute(0, 3, 1, 2)  # shift C to dim1
#        dec_outputs = dec_outputs.unsqueeze(2)
#        h = self.relu(self.deconv3(dec_outputs))
#        h = self.relu(self.deconv2(h))
##        h = self.pool1(h)
#        h = torch.sigmoid(self.deconv1(h))
##        h = self.pool2(h)
        
        return out, attn_wts
    
    def _init_hidden(self, batch_size):
         #* self.n_directions
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, \
                             self.hidden_size)
        return hidden.to(device)
    
    
class Conv3DAttention(nn.Module):
    """
    The C3D Layers defined for clips. No pooling in temporal dimension. Spatial pooling
    applied upto some extent.
    GRU layer at the end.
    """

    def __init__(self, hidden_size, n_classes, n_layers, max_length=196, bidirectional=False):
        super(Conv3DAttention, self).__init__()
        
        self.hidden_size = hidden_size      # 2048 after conv6
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.n_directions = int(bidirectional) + 1
        self.dropout_p = 0.1
        self.max_length = max_length
        
#        self.output_size = output_size
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
#        # each vector size is 2048 after pool6 and reshaping
        self.gru = nn.GRU(128, self.hidden_size, n_layers, batch_first=True,
                          bidirectional=bidirectional)
#        self.gru1 = nn.GRU(self.hidden_size, self.hidden_size, n_layers, batch_first=True,
#                          bidirectional=bidirectional)
        
#        self.fc = nn.Linear(self.hidden_size, output_size)

#        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        if torch.__version__.split('.')[0]=='1':
            self.softmax = nn.Softmax(dim = 1)
        else:
            self.softmax = nn.Softmax()     # PyTorch 0.2 Operates on 2D arrays
            
        self.attn = nn.Linear(self.hidden_size, self.max_length)
        self.dropout = nn.Dropout(self.dropout_p)
        self.classifier = nn.Linear(self.hidden_size, self.n_classes)
        

    def forward(self, x, hid_vec):
        
        batch, ch, seq, ht, wd = x.size()
        batch, ch, seq, ht, wd = int(batch), int(ch), int(seq), int(ht), int(wd)
        
        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3(h))
        h = self.pool3(h)
    
        # (B, 32, 1, 14, 14)
        h = h.squeeze(2).permute(0, 2, 3, 1)
        #h = h.permute(0, 2, 1, 3, 4)
        h = h.view(batch, -1, h.size(-1)) # No of sequences 14*14
        
        # convert to BATCH x HIDDEN_SIZE, outputs BATCH x SEQ_SIZE
        attn_weights = self.softmax(self.attn(hid_vec.view(-1, self.hidden_size)))
        # h is BATCH x SEQ_SIZE x HIDDEN_SIZE ; attn
        attn_applied = torch.mul(attn_weights[:,:,None], h)
#        attn_applied = torch.bmm(attn_weights.unsqueeze(1), h)
        
#        hid_vec_new = self._init_hidden(batch)
        output, hid_vec = self.gru(attn_applied, hid_vec)
        
        output = self.softmax(self.classifier(hid_vec.view(-1, self.hidden_size)))

        ###########################################
        # decoder_hidden (n_directions, BATCH, HIDDEN_SIZE)
        # encoder_outputs (BATCH, SEQ_SIZE, HIDDEN_SIZE)
        # input (BATCH, OUTPUT_SIZE)  :- INPUT_SIZE == OUTPUT_SIZE OR use embedding
        
#        # find attention weights (BATCH, SEQ_SIZE)
#        attn_weights = F.softmax(self.attn(torch.cat((input, decoder_hidden[0]), 1)), dim=1)
#        # if BATCH > 1 or BATCH==1, then unsqueeze(1) for attn_weights
#        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
#
##        # Either loop over the batch samples of encoder_outputs or broadcast attn_weights
##        attn_applied = []
##        for encoder_output in encoder_outputs:
##            attn_applied.append(torch.bmm(attn_weights.unsqueeze(0), encoder_output.unsqueeze(0)))            
##        torch.cat(attn_applied, 1)
#        
#        output = torch.cat((input, attn_applied.squeeze(1)), 1) #(BATCH, HIDDEN+OUTPUT_SIZE)
#        output = self.attn_combine(output) #.unsqueeze(0)   (BATCH, HIDDEN_SIZE)
#
##        output = F.relu(output)
#        output, decoder_hidden = self.gru(output.unsqueeze(1), decoder_hidden)
#
#        output = self.out(output.squeeze(1)) # F.log_softmax(self.out(output.squeeze(1)), dim=1)
#        return output, decoder_hidden, attn_weights
    
        ###########################################

        return output, hid_vec, attn_weights
    
    def _init_hidden(self, batch_size):
         #* self.n_directions
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, \
                             self.hidden_size)
        return hidden.to(device)

        
class Conv3DClassifier(nn.Module):
    """
    The C3D Layers defined for clips. No pooling in temporal dimension. Spatial pooling
    applied upto some extent.
    GRU layer at the end.
    """

    def __init__(self, hidden_size, n_layers, n_classes, bidirectional=False):
        super(Conv3DClassifier, self).__init__()
        
        self.hidden_size = hidden_size      # 2048 after conv6
        self.n_layers = n_layers
        self.n_directions = int(bidirectional) + 1
        self.n_classes = n_classes
        
#        self.output_size = output_size
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
#        # each vector size is 2048 after pool6 and reshaping
        self.gru = nn.GRU(128, self.hidden_size, n_layers, batch_first=True,
                          bidirectional=bidirectional)
#        self.gru1 = nn.GRU(self.hidden_size, self.hidden_size, n_layers, batch_first=True,
#                          bidirectional=bidirectional)
        
        self.fc = nn.Linear(self.hidden_size, n_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        if torch.__version__.split('.')[0]=='1':
            self.softmax = nn.Softmax(dim = 1)
        else:
            self.softmax = nn.Softmax()     # PyTorch 0.2 Operates on 2D arrays

    def forward(self, x, hid_vec):
        
        batch, ch, seq, ht, wd = x.size()
        batch, ch, seq, ht, wd = int(batch), int(ch), int(seq), int(ht), int(wd)
        
        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3(h))
        h = self.pool3(h)
        
        # (B, 32, 1, 14, 14)
        h = h.squeeze(2).permute(0, 2, 3, 1)
        #h = h.permute(0, 2, 1, 3, 4)
        h = h.view(batch, -1, h.size(-1)) # No of sequences 14*14
#        hid_vec = self._init_hidden(batch)
        output, hid_vec = self.gru(h, hid_vec)
        return output, hid_vec
    
    def _init_hidden(self, batch_size):
         #* self.n_directions
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, \
                             self.hidden_size)
        return hidden.to(device)