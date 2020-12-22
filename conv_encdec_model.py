#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 03:44:58 2020

@author: arpan

@Description: Convolutional Encoder Decoder Model for Cricket Strokes. Use reconstruction loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        
        self.conv4 = nn.Conv3d(128, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))        
        
#        self.deconv3 = nn.ConvTranspose3d(128, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2))
#        self.deconv2 = nn.ConvTranspose3d(64, 32, kernel_size=(2, 2, 2), stride=(2, 2, 2))
##        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
#
#        self.deconv1 = nn.ConvTranspose3d(32, 3, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
##        # each vector size is 2048 after pool6 and reshaping
#        self.gru = nn.GRU(128, self.hidden_size, n_layers, batch_first=True,
#                          bidirectional=bidirectional)
#        self.gru1 = nn.GRU(self.hidden_size, self.hidden_size, n_layers, batch_first=True,
#                          bidirectional=bidirectional)
        
#        self.fc = nn.Linear(self.hidden_size, output_size)

#        self.dropout = nn.Dropout(p=0.5)

        #self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU(0.1)
#        if torch.__version__.split('.')[0]=='1':
#            self.softmax = nn.Softmax(dim = 1)
#        else:
#            self.softmax = nn.Softmax()     # PyTorch 0.2 Operates on 2D arrays

    def forward(self, x, hid_vec):
        
        batch, ch, seq, ht, wd = x.size()
        batch, ch, seq, ht, wd = int(batch), int(ch), int(seq), int(ht), int(wd)
        
        h = self.conv1(x)
        h = self.relu(h)
        h = self.pool1(h)

        h = self.conv2(h)
        h = self.relu(h)
        h = self.pool2(h)

        h = self.conv3(h)
        h = self.relu(h)
        h = self.pool3(h)
        
        h = self.conv4(h)
        h = self.relu(h)
        h = self.pool4(h)
        
#        # (B, 32, 1, 14, 14)
#        h = h.squeeze(2).permute(0, 2, 3, 1)
#        #h = h.permute(0, 2, 1, 3, 4)
#        h = h.view(batch, -1, h.size(-1)) # No of sequences 14*14
##        hid_vec = self._init_hidden(batch)
#        output, hid_vec = self.gru(h, hid_vec)        
        return h, hid_vec
    
    def _init_hidden(self, batch_size):
         #* self.n_directions
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, \
                             self.hidden_size)
        return hidden.to(device)

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
        self.deconv4 = nn.ConvTranspose3d(128, 128, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.deconv3 = nn.ConvTranspose3d(128, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.deconv3 = nn.ConvTranspose3d(128, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.deconv2 = nn.ConvTranspose3d(64, 32, kernel_size=(2, 2, 2), stride=(2, 2, 2))
#        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.deconv1 = nn.ConvTranspose3d(32, 3, kernel_size=(2, 2, 2), stride=(2, 2, 2))
#        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

#        self.gru = nn.GRU(8192, self.hidden_size, n_layers, batch_first=True,
#                          bidirectional=bidirectional)
#        self.attn_dec = AttentionDecoder(self.hidden_size, self.hidden_size, self.n_layers, 
#                                         bidirectional, self.max_length)

#        self.fc = nn.Linear(self.hidden_size, output_size)
#
#        self.dropout = nn.Dropout(p=0.5)

        #self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU(0.1)
#        if torch.__version__.split('.')[0]=='1':
#            self.softmax = nn.Softmax(dim = 1)
#        else:
#            self.softmax = nn.Softmax()     # PyTorch 0.2 Operates on 2D arrays

    def forward(self, enc_outputs):
        
#        dec_out_lst, attn_wts_lst = [], []
#        for ti in range(enc_outputs.size(1)):
#            #start symbol of dim  (batch x output_size) 
#            inp = torch.zeros((dec_h.size(1), self.hidden_size)).to(device)  #starting symbol
#            dec_out, dec_h, attn_wts = self.attn_dec(dec_h, enc_outputs, inp)
#            dec_out_lst.append(dec_out)
#            attn_wts_lst.append(attn_wts)
#            
#        # reconstruct the output
#        dec_outputs = torch.stack(dec_out_lst, dim=1)
#        batch, seq, ftsize = enc_outputs.size()
#        ht = int(math.sqrt(seq))
#        enc_outputs = enc_outputs.reshape(batch, ht, ht, ftsize).permute(0, 3, 1, 2)  # shift C to dim1
#        enc_outputs = enc_outputs.unsqueeze(2)
        h = self.deconv4(enc_outputs)
        h = self.relu(h)
        h = self.deconv3(h)  #(enc_outputs))
        h = self.relu(h)
        h = self.deconv2(h)
        h = self.relu(h)
#        h = self.pool1(h)
        h = self.deconv1(h)
#        h = self.pool2(h)
        
        return h  #, attn_wts_lst

#    def _init_hidden(self, batch_size):
#         #* self.n_directions
#        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, \
#                             self.hidden_size)
#        return hidden.to(device)
    
    
    
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
    
    
class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()
        
        self.ftsize = 128
        self.enc_out_dim = 32
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4 = nn.Conv3d(128, self.ftsize, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.fc1 = nn.Linear(6272, 1024)
        self.fc21 = nn.Linear(1024, self.enc_out_dim)
        self.fc22 = nn.Linear(1024, self.enc_out_dim)
        self.fc3 = nn.Linear(self.enc_out_dim, 1024)
        self.fc4 = nn.Linear(1024, 6272)
        self.deconv4 = nn.ConvTranspose3d(self.ftsize, 128, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.deconv3 = nn.ConvTranspose3d(128, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.deconv2 = nn.ConvTranspose3d(64, 32, kernel_size=(2, 2, 2), stride=(2, 2, 2))
#        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.deconv1 = nn.ConvTranspose3d(32, 3, kernel_size=(2, 2, 2), stride=(2, 2, 2))        
#        self.relu = nn.ReLU()
#        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        batch = int(x.size(0))
        h = F.leaky_relu(self.conv1(x))
        h = self.pool1(h)

        h = F.leaky_relu(self.conv2(h))
        h = self.pool2(h)

        h = F.leaky_relu(self.conv3(h))
        h = self.pool3(h)
        
        h = F.leaky_relu(self.conv4(h))
        h = self.pool4(h)
        h = h.squeeze(2).permute(0, 2, 3, 1)
        #h = h.permute(0, 2, 1, 3, 4)
        h = h.reshape((batch, -1)) #, h.size(-1)) # No of sequences 14*14

        h1 = F.leaky_relu(self.fc1(h))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.leaky_relu(self.fc3(z))
        h3 = self.fc4(h3)
        batch, ft_seq = h3.size()
        ht = int(math.sqrt(ft_seq/self.ftsize))
        h3 = h3.reshape((batch, ht, ht, self.ftsize)).permute(0, 3, 1, 2)  # shift C to dim1
        h3 = h3.unsqueeze(2)
        h3 = self.deconv4(h3)
        h3 = self.deconv3(h3)
        h3 = self.deconv2(h3)
        h3 = self.deconv1(h3)
        return torch.sigmoid(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
#    def get_vec(self, src):
#        num_clips = src.size(0)
#        out_mu = torch.zeros(num_clips, self.enc_out_dim)
#        out_logvar = torch.zeros(num_clips, self.enc_out_dim)
#
#        def copy_mu(m, i, o):
#            out_mu.copy_(o.data)
#
#        def copy_logvar(m, i, o):
#            out_logvar.copy_(o.data)
#            
#        extraction_mu = self.fc21
#        extraction_logvar = self.fc22
#        h_mu = extraction_mu.register_forward_hook(copy_mu)
#        h_logvar = extraction_logvar.register_forward_hook(copy_logvar)
#        h_x = self.forward(src)
#        h_mu.remove()
#        h_logvar.remove()
#        return extraction_mu, extraction_logvar

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x) #, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD