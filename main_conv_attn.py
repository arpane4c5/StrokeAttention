#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 23:43:52 2020

@author: arpan

@Description: Train an Attention model on strokes. Problem: Test Acc at training is 0.00
"""

import os
import sys
import numpy as np


sys.path.insert(0, '../cluster_strokes')
sys.path.insert(0, '../cluster_strokes/lib')

##from utils import spectral_utils
#from utils import plot_utils
#from evaluation import eval_of_clusters
import torch
from torch import nn, optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

#from utils.resnet_feature_extracter import Clip2Vec
from utils import autoenc_utils
import datasets.videotransforms as videotransforms
#from datasets.dataset import StrokeFeatureSequenceDataset
from datasets.dataset import CricketStrokesDataset
import copy
import time
import pickle
import conv_attn_model
import attn_utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_path = "/home/arpan/VisionWorkspace/Cricket/StrokeAttention/logs"
#ft_dir = "bow_HL_ofAng_grid20"
#feat, feat_val = "of_feats_grid20.pkl", "of_feats_val_grid20.pkl"
#snames, snames_val = "of_snames_grid20.pkl", "of_snames_val_grid20.pkl"
#INPUT_SIZE, TARGET_SIZE = 576, 576      # OFGRID: 576, 3DCNN: 512, 2DCNN: 2048
HIDDEN_SIZE = 128 #64#1024
bidirectional = False
SHIFT = 4

def copy_pretrained_weights(model_src, model_tar):
    params_src = model_src.named_parameters()
    params_tar = model_tar.named_parameters()
    dict_params_tar = dict(params_tar)
    for name_src, param_src in params_src:
        if name_src in dict_params_tar:
            dict_params_tar[name_src].data.copy_(param_src.data)
            dict_params_tar[name_src].requires_grad = False     # Freeze layer wts
            

def train_model(encoder, dataloaders, criterion, encoder_optimizer, 
                scheduler, labs_keys, labs_values, seq=8, num_epochs=25):
    since = time.time()

#    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                encoder.train()  # Set model to training mode
#                decoder.train()
            else:
                encoder.eval()   # Set model to evaluate mode
#                decoder.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for bno, (inputs, vid_path, stroke, labels) in enumerate(dataloaders[phase]):
                # inputs of shape BATCH x SEQ_LEN x FEATURE_DIM
                labels = attn_utils.get_batch_labels(vid_path, stroke, labs_keys, labs_values, 1)
                # Extract spatio-temporal features from clip using 3D ResNet (For SL >= 16)
                inputs = inputs.permute(0, 2, 1, 3, 4).float()
                
                targets = inputs
#                inputs = inputs.to(device)
#               targets = targets.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                encoder_optimizer.zero_grad()
#                decoder_optimizer.zero_grad()
                loss = 0

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
                    batch_size = inputs.size(0)
                    
                    for si in range(0, inputs.size(2)-seq+1, SHIFT):
                        mod_inp = inputs[:,:,si:(si+seq)]
                        mod_inp = mod_inp.to(device)
                        enc_h = encoder._init_hidden(batch_size)
                        enc_out, enc_h, attn_wts = encoder(mod_inp, enc_h)
                        # dec_out, attn_wts_lst = decoder(h, enc_out)
                        loss += criterion(enc_out, labels)
                        _, preds = torch.max(enc_out, 1)
                    
#                    dec_h = h
#                    dec_out_lst = []
#                    target_length = targets.size(1)      # assign SEQ_LEN as target length for now
#                    # run for each word of the sequence (use teacher forcing)
#                    for ti in range(target_length):
#                        dec_out, dec_h, dec_attn = decoder(dec_h, enc_out, targets[:,ti,:])
#                        dec_out_lst.append(dec_out)
#                        loss += criterion(dec_out, targets[:,ti,:])
#                        #decoder_input = target_tensor[di]  # Teacher forcing
#            
#                    outputs = torch.stack(dec_out_lst, dim=1)
                    
#                    outputs, dec_h, wts = model(inputs, inputs)
#                    _, preds = torch.max(outputs, 1)
#                    loss = criterion(outputs, targets)     #torch.flip(targets, [1])

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        encoder_optimizer.step()
#                        decoder_optimizer.step()

                # statistics
                running_loss += loss.item()
#                print("Iter : {} / {} :: Running Loss : {}".format(bno, 
#                      len(dataloaders[phase]), running_loss))
                running_corrects += torch.sum(preds == labels.data)
                
#                print("Batch No : {} / {}".format(bno, len(dataloaders[phase])))
                if (bno+1) % 10 == 0:
                    break
                    
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss #/ len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() #/ len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} :: Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

#            # deep copy the model
#            if phase == 'test' and epoch_acc > best_acc:
#                best_acc = epoch_acc
#                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, \
          time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

#    # load best model weights
#    model.load_state_dict(best_model_wts)
    return encoder  #, decoder



def predict(encoder, decoder, dataloaders, labs_keys, labs_values, phase="val"):
    assert phase == "val" or phase=="test", "Incorrect Phase."
    encoder = encoder.eval()
    decoder = decoder.eval()
    vid_path_lst, stroke_lst, labs_lst, batch_wts = [], [], [], []
    # Iterate over data.
    for bno, (inputs, vid_path, stroke, labels) in enumerate(dataloaders[phase]):
        # inputs of shape BATCH x SEQ_LEN x FEATURE_DIM
        labels = attn_utils.get_batch_labels(vid_path, stroke, labs_keys, labs_values, 1)
        # Extract spatio-temporal features from clip using 3D ResNet (For SL >= 16)
        inputs = inputs.permute(0, 2, 1, 3, 4).float()
        
        targets = inputs
        inputs = inputs.to(device)
        loss = 0
        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            
            batch_size = inputs.size(0)
            enc_h = encoder._init_hidden(batch_size)
            enc_out, h = encoder(inputs, enc_h)
            dec_out, attn_wts = decoder(h, enc_out)
            
            vid_path_lst.append(vid_path)
            stroke_lst.append(stroke)
            labs_lst.append(labels)
            batch_wts.append(attn_wts)
            #attn_wts = torch.stack(attn_wts_lst)
#            dec_h = h
#            dec_in = torch.zeros(batch_size, targets.size(2)).to(device)
#            dec_out_lst = []
#            target_length = targets.size(1)      # assign SEQ_LEN as target length for now
#            # run for each word of the sequence (use teacher forcing)
#            for ti in range(target_length):
#                dec_out, dec_h, dec_attn = decoder(dec_h, enc_out, dec_in)
#                dec_out_lst.append(dec_out)
##                loss += criterion(dec_out, targets[:,ti,:])
#                dec_in = dec_out
                
#            outputs = torch.stack(dec_out_lst, dim=1)
            
#                    outputs, dec_h, wts = model(inputs, inputs)
#                    _, preds = torch.max(outputs, 1)
#                    loss = criterion(outputs, targets)     #torch.flip(targets, [1])
        # statistics
#        running_loss += loss.item()
#                print("Iter : {} :: Running Loss : {}".format(bno, running_loss))
#                running_corrects += torch.sum(preds == labels.data)
        
        print("Batch No : {} / {}".format(bno, len(dataloaders[phase])))
        if (bno+1) % 20 == 0:
            break
#    epoch_loss = running_loss #/ len(dataloaders[phase].dataset)
#            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
    pred_dict = {"paths": vid_path_lst, 
                 "strokes": stroke_lst, 
                 "labels": labs_lst, 
                 "wts": batch_wts}
    return pred_dict
    

def main(DATASET, LABELS, CLASS_IDS, BATCH_SIZE, ANNOTATION_FILE, SEQ_SIZE=16, 
         STEP=16, nstrokes=-1, N_EPOCHS=25, base_name=""):
    '''
    Extract sequence features from AutoEncoder.
    
    Parameters:
    -----------
    DATASET : str
        path to the video dataset
    LABELS : str
        path containing stroke labels
    CLASS_IDS : str
        path to txt file defining classes, similar to THUMOS
    BATCH_SIZE : int
        size for batch of clips
    SEQ_SIZE : int
        no. of frames in a clip (min. 16 for 3D CNN extraction)
    STEP : int
        stride for next example. If SEQ_SIZE=16, STEP=8, use frames (0, 15), (8, 23) ...
    partition : str
        'all' / 'train' / 'test' / 'val' : Videos to be considered
    nstrokes : int
        partial extraction of features (do not execute for entire dataset)
    
    Returns:
    --------
    trajectories, stroke_names
    
    '''
    ###########################################################################
    # Read the strokes 
    # Divide the highlight dataset files into training, validation and test sets
    train_lst, val_lst, test_lst = autoenc_utils.split_dataset_files(DATASET)
    print("No. of training videos : {}".format(len(train_lst)))
        
    ###########################################################################
    # Create a Dataset    
    # Clip level transform. Use this with framewiseTransform flag turned off
    clip_transform = transforms.Compose([videotransforms.CenterCrop(224),
                                         videotransforms.ToPILClip(), 
                                         videotransforms.Resize((112, 112)),
#                                         videotransforms.RandomCrop(112), 
                                         videotransforms.ToTensor(), 
                                         videotransforms.Normalize(),
                                        #videotransforms.RandomHorizontalFlip(),\
                                        ])
    train_dataset = CricketStrokesDataset(train_lst, DATASET, LABELS, CLASS_IDS, 
                                         frames_per_clip=SEQ_SIZE, step_between_clips=STEP, 
                                         train=True, framewiseTransform=False, 
                                         transform=clip_transform)
    val_dataset = CricketStrokesDataset(val_lst, DATASET, LABELS, CLASS_IDS, 
                                        frames_per_clip=SEQ_SIZE, step_between_clips=STEP, 
                                        train=False, framewiseTransform=False, 
                                        transform=clip_transform)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    data_loaders = {"train": train_loader, "test": val_loader}

    ###########################################################################
    
    labs_keys, labs_values = attn_utils.get_cluster_labels(ANNOTATION_FILE)
    
    num_classes = len(list(set(labs_values)))
    
    ###########################################################################    
    # load model and set loss function    
    encoder = conv_attn_model.Conv3DAttention(HIDDEN_SIZE, num_classes, 1, 196, bidirectional)
#    encoder = conv_attn_model.Conv3DEncoder(HIDDEN_SIZE, 1, bidirectional)
#    decoder = conv_attn_model.Conv3DDecoderClassifier(HIDDEN_SIZE, 5, 1, 196, bidirectional)
#    decoder = conv_attn_model.Conv3DDecoder(HIDDEN_SIZE, HIDDEN_SIZE, 1, 196, bidirectional)
#    model = attn_model.Encoder(10, 20, bidirectional)
    
#    for ft in model.parameters():
#        ft.requires_grad = False
#    inp_feat_size = model.fc.in_features
#    model.fc = nn.Linear(inp_feat_size, num_classes)
#    model = model.to(device)
    encoder = encoder.to(device)
#    decoder = decoder.to(device)
#    # load checkpoint:
    
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
#    criterion = nn.MSELoss()
    
#    # Layers to finetune. Last layer should be displayed
    print("Params to learn:")
    params_to_update = []
    for name, param in encoder.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("Encoder : {}".format(name))
#    for name, param in decoder.named_parameters():
#        if param.requires_grad == True:
#            params_to_update.append(param)
#            print("Decoder : {}".format(name))
    
    # Observe that all parameters are being optimized
#    optimizer_ft = torch.optim.Adam(params_to_update, lr=0.001)
    encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=0.01, momentum=0.9)
#    decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=0.01, momentum=0.9)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    lr_scheduler = StepLR(encoder_optimizer, step_size=10, gamma=0.1)
    
#    # Observe that all parameters are being optimized
#    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
        
#    ###########################################################################
    # Training the model
    start = time.time()
    
#    (encoder, decoder) = train_model(encoder, decoder, data_loaders, criterion, 
#                           encoder_optimizer, decoder_optimizer, lr_scheduler, labs_keys, labs_values,
#                           num_epochs=N_EPOCHS)
    encoder = train_model(encoder, data_loaders, criterion, encoder_optimizer,
                                    lr_scheduler, labs_keys, labs_values, seq=8,
                                    num_epochs=N_EPOCHS)
        
    end = time.time()
    
    # save the best performing model
#    attn_utils.save_attn_model_checkpoint(base_name, (encoder, decoder), N_EPOCHS, "SGD")
    # Load model checkpoints
#    encoder, decoder = attn_utils.load_attn_model_checkpoint(base_name, encoder, decoder, N_EPOCHS, "SGD")
    
    print("Total Execution time for {} epoch : {}".format(N_EPOCHS, (end-start)))

    ###########################################################################    
    
#    features_val, stroke_names_id_val = attn_utils.read_feats(os.path.join(base_name, ft_dir), 
#                                                              feat_val, snames_val)
    
    pred_out_dict = predict(encoder, decoder, data_loaders, labs_keys, 
                            labs_values, phase='test')
    print("Writing prediction dictionary....")
    with open(os.path.join(base_name, "pred_dict.pkl"), "wb") as fp:
        pickle.dump(pred_out_dict, fp)
    
    # save the output wts and related information
    
    print("#Parameters Encoder : {} ".format(autoenc_utils.count_parameters(encoder)))
#    print("#Parameters Decoder : {} ".format(autoenc_utils.count_parameters(decoder)))
    
    
    return encoder, decoder

if __name__ == '__main__':
    # Local Paths
    LABELS = "/home/arpan/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
    DATASET = "/home/arpan/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
    CLASS_IDS = "/home/arpan/VisionWorkspace/Cricket/cluster_strokes/configs/Class Index_Strokes.txt"
    ANNOTATION_FILE = "/home/arpan/VisionWorkspace/Cricket/CricketStrokeLocalizationBOVW/shots_classes.txt"

    SEQ_SIZE = 16
    STEP = 1
    BATCH_SIZE = 8
    N_EPOCHS = 3
    
    encoder, decoder =  main(DATASET, LABELS, CLASS_IDS, BATCH_SIZE, ANNOTATION_FILE, 
                             SEQ_SIZE, STEP, nstrokes=-1, N_EPOCHS=N_EPOCHS, 
                             base_name=base_path)

