#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 09:42:33 2020

@author: arpan

@Description: Defining the Attention model for training on video sequence features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, bidirectional = True):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.n_directions = int(bidirectional) + 1
        self.n_layers = n_layers
    
#        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional = bidirectional)
        self.gru = nn.GRU(input_size, hidden_size, n_layers, batch_first=True,
                          bidirectional=bidirectional)
        
    def forward(self, inputs, hidden):
        batch_size = inputs.size(0)
        output, hidden = self.gru(inputs.view(batch_size, -1, self.input_size), hidden)
        return output, hidden
    
    def init_hidden(self, batch_size):
        enc_h = torch.zeros(self.n_directions * self.n_layers, batch_size, self.hidden_size)
        enc_h = enc_h.to(device)        
        return enc_h
    
class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, bidirectional=False, 
                 max_length=10, dropout_p=0.1):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

#        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size + self.output_size, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size + self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)        
        
#        self.n_directions = int(bidirectional) + 1
#        self.n_layers = n_layers
#        # use max_length for attn output and use attn_combine
#        self.attn = nn.Linear(hidden_size + output_size, max_length)
#        #if we are using embedding hidden_size should be added with embedding instead of vocab size
##        self.lstm = nn.LSTM(hidden_size + vocab_size, output_size) 
#        self.gru = nn.GRU(hidden_size + vocab_size, output_size, batch_first=True, 
#                          bidirectional=bidirectional)
#        self.final = nn.Linear(output_size, vocab_size)
  
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
        
#        weights = []
#        for i in range(len(encoder_outputs)):
#            print(decoder_hidden[0][0].shape)
#            print(encoder_outputs[0].shape)
#            weights.append(self.attn(torch.cat((decoder_hidden[0][0], 
#                                                encoder_outputs[i]), dim = 1)))
#        normalized_weights = F.softmax(torch.cat(weights, 1), 1)
#    
#        attn_applied = torch.bmm(normalized_weights.unsqueeze(1),
#                                 encoder_outputs.view(1, -1, self.hidden_size))
#    
#        #if we are using embedding, use embedding of input here instead
#        input_lstm = torch.cat((attn_applied[0], input[0]), dim = 1) 
#    
#        output, hidden = self.gru(input_lstm.unsqueeze(0), decoder_hidden)
#    
#        output = self.final(output[0])
#    
#        return output, hidden, normalized_weights

class AttentionEncoderDecoder(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, seq_len, n_layers=1, 
                 bidirectional=False):
        super(AttentionEncoderDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.encoder = Encoder(input_size, hidden_size, n_layers, bidirectional)
        self.decoder = AttentionDecoder(hidden_size*(1+bidirectional), output_size, 
                                        max_length=seq_len)
        
    def forward(self, inputs, targets):
        
        batch_size = inputs.size(0)
        enc_h = self.encoder.init_hidden(batch_size)
        enc_out, h = self.encoder(inputs, enc_h)
        
#        dec_h = self.decoder.init_hidden(batch_size)
        dec_h = h
#        enc_out = torch.cat((enc_out, enc_out))
#        dec_inp = torch.zeros(1, 1, self.vocab_size)
        dec_out_lst = []
        target_length = targets.size(1)      # assign SEQ_LEN as target length for now
        # run for each word of the sequence (use teacher forcing)
        for ti in range(target_length):
            dec_out, dec_h, dec_attn = self.decoder(dec_h, enc_out, targets[:,ti,:])
            dec_out_lst.append(dec_out)
#            loss += criterion(decoder_output, target_tensor[di])
            #decoder_input = target_tensor[di]  # Teacher forcing
            
        return torch.stack(dec_out_lst, dim=1), dec_h, dec_attn
    
class AttentionEncoderDecoderClassifier(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, seq_len, n_classes=5, 
                 n_layers=1, bidirectional=False):
        super(AttentionEncoderDecoderClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_classes = n_classes
        self.encoder = Encoder(input_size, hidden_size, n_layers, bidirectional)
        self.decoder = AttentionDecoder(hidden_size*(1+bidirectional), output_size, 
                                        max_length=seq_len)
        self.classifier = nn.Linear(output_size, n_classes)
        
    def forward(self, inputs, dec_in):
        
        batch_size = inputs.size(0)
        enc_h = self.encoder.init_hidden(batch_size)
        enc_out, h = self.encoder(inputs, enc_h)
        
#        dec_h = self.decoder.init_hidden(batch_size)
        dec_h = h
#        enc_out = torch.cat((enc_out, enc_out))
#        dec_inp = torch.zeros(1, 1, self.vocab_size)
#        dec_out_lst = []
        # run for entire batch all at once at the time of classification
        dec_out, dec_h, dec_attn = self.decoder(dec_h, enc_out, dec_in)
#        for ti in range(target_length):
#            dec_out, dec_h, dec_attn = self.decoder(dec_h, enc_out, targets[:,ti,:])
        out = self.classifier(dec_out)
#        dec_out_lst.append(dec_out)
        #decoder_input = target_tensor[di]  # Teacher forcing
            
        return out, dec_h, dec_attn

class GRUBoWHAClassifier(nn.Module):
    '''GRU model with Embedding layers for Hard Assignment (OneHot) sequences
    '''
    def __init__(self, input_size, hidden_size, n_classes, n_layers=1, 
                 bidirectional=False):
        super(GRUBoWHAClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.n_directions = int(bidirectional) + 1
        self.n_layers = n_layers
#        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional = bidirectional)
        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        self.gru1 = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True,
                          bidirectional=bidirectional)
#        self.gru2 = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True,
#                          bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * self.n_directions, hidden_size)
        self.classifier = nn.Linear(hidden_size, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, inputs, hidden):
        batch_size = inputs.size(0)
        emb = self.embedding(inputs)
        output, hidden = self.gru1(emb, hidden)
#        output, hidden = self.gru2(output, hidden)
#        hidden = self.dropout(self.relu(self.fc(hidden.squeeze(0))))
#        output = self.dropout(self.relu(self.fc(output.contiguous().view(-1, self.hidden_size))))
        output = self.dropout(self.fc(output[:,-1,:]))
#        logits = self.classifier(output)
#        probs = self.softmax(logits.view(-1, self.n_classes))
#        return probs, hidden
        return output, hidden
    
    def init_hidden(self, batch_size):
        enc_h = torch.zeros(self.n_directions * self.n_layers, batch_size, self.hidden_size)
        enc_h = enc_h.to(device)
        return enc_h
    
#class GRUBoWSAClassifier(nn.Module):
#    '''For Soft Assignment sequences
#    '''
#    def __init__(self, input_size, hidden_size, n_classes, n_layers=1, 
#                 bidirectional=False):
#        super(GRUBoWSAClassifier, self).__init__()
#        self.input_size = input_size
#        self.hidden_size = hidden_size
#        self.n_classes = n_classes
#        self.n_directions = int(bidirectional) + 1
#        self.n_layers = n_layers
#        self.gru1 = nn.GRU(input_size, hidden_size, n_layers, batch_first=True,
#                          bidirectional=bidirectional)
##        self.gru2 = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True,
##                          bidirectional=bidirectional)
#        self.fc = nn.Linear(hidden_size * self.n_directions, hidden_size)
#        self.classifier = nn.Linear(hidden_size, n_classes)
#        self.relu = nn.ReLU()
#        self.dropout = nn.Dropout(p=0.5)
#        self.softmax = nn.Softmax(dim=1)
#        
#    def forward(self, inputs, hidden):
#        batch_size = inputs.size(0)
#        output, hidden = self.gru1(inputs, hidden)
##        output, hidden = self.gru2(output, hidden)
#        output = self.dropout(self.fc(output[:,-1,:]))
##        logits = self.classifier(output)
##        probs = self.softmax(logits.view(-1, self.n_classes))
#        return output, hidden
#    
#    def init_hidden(self, batch_size):
#        enc_h = torch.zeros(self.n_directions * self.n_layers, batch_size, self.hidden_size)
#        enc_h = enc_h.to(device)
#        return enc_h
        
class GRUClassifier(nn.Module):
    
    def __init__(self, input_size, hidden_size, n_classes, n_layers=1, 
                 bidirectional=False):
        super(GRUClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.n_directions = int(bidirectional) + 1
        self.n_layers = n_layers
#        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional = bidirectional)
        self.gru1 = nn.GRU(input_size, hidden_size, n_layers, batch_first=True,
                          bidirectional=bidirectional)
#        self.gru2 = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True,
#                          bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * self.n_directions, hidden_size)
        self.classifier = nn.Linear(hidden_size, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, inputs, hidden):
        batch_size = inputs.size(0)
        output, hidden = self.gru1(inputs, hidden)
#        output, hidden = self.gru2(output, hidden)
#        hidden = self.dropout(self.relu(self.fc(hidden.squeeze(0))))
#        output = self.dropout(self.relu(self.fc(output.contiguous().view(-1, self.hidden_size))))
        output = self.dropout(self.fc(output[:,-1,:]))
        logits = self.classifier(output)
        probs = self.softmax(logits.view(-1, self.n_classes))
        return probs, hidden
    
    def init_hidden(self, batch_size):
        enc_h = torch.zeros(self.n_directions * self.n_layers, batch_size, self.hidden_size)
        enc_h = enc_h.to(device)
        return enc_h
    
class GRUBoWMultiStreamClassifier(nn.Module):
    '''GRU model with Embedding layers for Hard Assignment (OneHot) sequences
    '''
    def __init__(self, input_size1, input_size2, hidden_size1, hidden_size2, n_classes, 
                 n_layers=1, bidirectional=False):
        super(GRUBoWMultiStreamClassifier, self).__init__()
        
        self.stream1_model = GRUBoWSAClassifier(input_size1, hidden_size1, n_classes, n_layers, bidirectional)
        self.stream2_model = GRUBoWSAClassifier(input_size1, hidden_size1, n_classes, n_layers, bidirectional)
#        self.input_size = input_size
#        self.hidden_size = hidden_size
        self.n_classes = n_classes
#        self.n_directions = int(bidirectional) + 1
#        self.n_layers = n_layers
##        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional = bidirectional)
#        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
#        self.gru1 = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True,
#                          bidirectional=bidirectional)
#        self.gru2 = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True,
#                          bidirectional=bidirectional)
#        self.fc = nn.Linear(hidden_size * self.n_directions, hidden_size)
        self.classifier = nn.Linear(hidden_size1 + hidden_size2, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, inputs1, inputs2):
#        batch_size = inputs1.size(0)
        hidden1 = self.stream1_model.init_hidden(inputs1.size(0))
        hidden2 = self.stream2_model.init_hidden(inputs2.size(0))
#        emb = self.embedding(inputs1)
#        output, hidden = self.gru1(emb, hidden)
        out1, hidden1 = self.stream1_model(inputs1, hidden1)
        out2, hidden2 = self.stream2_model(inputs2, hidden2)
        out1 = self.relu(out1)
        out2 = self.relu(out2)
#        hidden = self.dropout(self.relu(self.fc(hidden.squeeze(0))))
#        output = self.dropout(self.relu(self.fc(output.contiguous().view(-1, self.hidden_size))))
#        output = self.dropout(self.fc(output[:,-1,:]))
        logits = self.classifier(torch.cat((out1, out2), dim=1))
        probs = self.softmax(logits.view(-1, self.n_classes))
        return probs #, hidden
#        return out1, out2


if __name__ == '__main__':
    bidirectional = False
    input_size, hidden_size, seq_len = 10, 20, 5
    
    enc_inp = torch.randn((1, seq_len, input_size))
    model = AttentionEncoderDecoder(input_size, hidden_size, input_size, seq_len)
    dec_out, dec_h, wts = model(enc_inp, enc_inp)
    
#    c = Encoder(10, 20, bidirectional)
#    a, b = c.forward(torch.randn(10), c.init_hidden())
#    print(a.shape)
#    print(b[0].shape)
#    print(b[1].shape)
#    
#    x = AttentionDecoder(20 * (1 + bidirectional), 25, 30)
#    y, z, w = x.forward(x.init_hidden(), torch.cat((a,a)), torch.zeros(1,1, 30)) #Assuming <SOS> to be all zeros
#    print(y.shape)
#    print(z[0].shape)
#    print(z[1].shape)
#    print(w)
    
#class AttnDecoderRNN(nn.Module):
#    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=10):
#        super(AttnDecoderRNN, self).__init__()
#        self.hidden_size = hidden_size
#        self.output_size = output_size
#        self.dropout_p = dropout_p
#        self.max_length = max_length
#
#        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
#        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
#        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
#        self.dropout = nn.Dropout(self.dropout_p)
#        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
#        self.out = nn.Linear(self.hidden_size, self.output_size)
#
#    def forward(self, input, hidden, encoder_outputs):
#        embedded = self.embedding(input).view(1, 1, -1)
#        embedded = self.dropout(embedded)
#
#        attn_weights = F.softmax(
#            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
#        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
#                                 encoder_outputs.unsqueeze(0))
#
#        output = torch.cat((embedded[0], attn_applied[0]), 1)
#        output = self.attn_combine(output).unsqueeze(0)
#
#        output = F.relu(output)
#        output, hidden = self.gru(output, hidden)
#
#        output = F.log_softmax(self.out(output[0]), dim=1)
#        return output, hidden, attn_weights
#
#    def initHidden(self):
#        return torch.zeros(1, 1, self.hidden_size)
    
    
###############################################################################
###############################################################################

#class GRUBoWHAClassifier(nn.Module):
#    '''GRU model with Embedding layers for Hard Assignment (OneHot) sequences
#    '''
#    def __init__(self, input_size, hidden_size, n_classes, n_layers=1, 
#                 bidirectional=False):
#        super(GRUBoWHAClassifier, self).__init__()
#        self.input_size = input_size
#        self.hidden_size = hidden_size
#        self.n_classes = n_classes
#        self.n_directions = int(bidirectional) + 1
#        self.n_layers = n_layers
##        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional = bidirectional)
#        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
#        self.gru1 = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True,
#                          bidirectional=bidirectional)
##        self.gru2 = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True,
##                          bidirectional=bidirectional)
#        self.fc = nn.Linear(hidden_size * self.n_directions, hidden_size)
#        self.classifier = nn.Linear(hidden_size, n_classes)
#        self.relu = nn.ReLU()
#        self.dropout = nn.Dropout(p=0.5)
#        self.softmax = nn.Softmax(dim=1)
#        
#    def forward(self, inputs, hidden):
#        batch_size = inputs.size(0)
#        emb = self.embedding(inputs)
#        output, hidden = self.gru1(emb, hidden)
##        output, hidden = self.gru2(output, hidden)
##        hidden = self.dropout(self.relu(self.fc(hidden.squeeze(0))))
##        output = self.dropout(self.relu(self.fc(output.contiguous().view(-1, self.hidden_size))))
#        output = self.dropout(self.fc(output[:,-1,:]))
#        logits = self.classifier(output)
#        probs = self.softmax(logits.view(-1, self.n_classes))
#        return probs, hidden
#    
#    def init_hidden(self, batch_size):
#        enc_h = torch.zeros(self.n_directions * self.n_layers, batch_size, self.hidden_size)
#        enc_h = enc_h.to(device)
#        return enc_h
#    
class GRUBoWSAClassifier(nn.Module):
    '''For Soft Assignment sequences
    '''
    def __init__(self, input_size, hidden_size, n_classes, n_layers=1, 
                 bidirectional=False):
        super(GRUBoWSAClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.n_directions = int(bidirectional) + 1
        self.n_layers = n_layers
        self.gru1 = nn.GRU(input_size, hidden_size, n_layers, batch_first=True,
                          bidirectional=bidirectional)
#        self.gru2 = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True,
#                          bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * self.n_directions, hidden_size)
        self.classifier = nn.Linear(hidden_size, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, inputs, hidden):
        batch_size = inputs.size(0)
        output, hidden = self.gru1(inputs, hidden)
#        output, hidden = self.gru2(output, hidden)
        output = self.dropout(self.fc(output[:,-1,:]))
        logits = self.classifier(output)
        probs = self.softmax(logits.view(-1, self.n_classes))
        return probs, hidden
    
    def init_hidden(self, batch_size):
        enc_h = torch.zeros(self.n_directions * self.n_layers, batch_size, self.hidden_size)
        enc_h = enc_h.to(device)
        return enc_h