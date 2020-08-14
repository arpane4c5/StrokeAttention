#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 10:39:42 2020

@author: arpan

@Description: Training Attention model on downstream task.
"""

import os
import sys
import numpy as np


sys.path.insert(0, '../cluster_strokes')
sys.path.insert(0, '../cluster_strokes/lib')

#from utils.extract_autoenc_feats import extract_3DCNN_feats
#from utils.extract_autoenc_feats import extract_2DCNN_feats
#from utils import trajectory_utils as traj_utils
##from utils import spectral_utils
#from utils import plot_utils
#from evaluation import eval_of_clusters
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

#from utils.resnet_feature_extracter import Clip2Vec
from utils import autoenc_utils
#import datasets.videotransforms as videotransforms
#from datasets.dataset import StrokeFeatureSequenceDataset
from datasets.dataset import StrokeFeaturePairsDataset
import copy
import time
import csv
import pickle
import attn_model
import attn_utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_path = "/home/arpan/VisionWorkspace/Cricket/CricketStrokeLocalizationBOVW/logs"
ft_dir = "bow_HL_ofAng_grid20"
feat, feat_val = "of_feats_grid20.pkl", "of_feats_val_grid20.pkl"
snames, snames_val = "of_snames_grid20.pkl", "of_snames_val_grid20.pkl"
INPUT_SIZE, TARGET_SIZE = 576, 576      # OFGRID: 576, 3DCNN: 512, 2DCNN: 2048
HIDDEN_SIZE = 256 #64#1024
bidirectional = False

        
def save_model_checkpoint(base_name, model, ep, opt):
    """
    Save the optimizer state with epoch no. along with the weights
    """
    # Save only the model params
    model_name = os.path.join(base_name, "downstream_attn_ep"+str(ep)+"_"+opt+".pt")

    torch.save(model.state_dict(), model_name)
    print("Model saved to disk... : {}".format(model_name))

def load_weights(base_name, model, ep, opt):
    """
    Load the pretrained weights to the models' encoder and decoder modules
    """
    # Paths to encoder and decoder files
    enc_name = os.path.join(base_name, "enc_attn_ep"+str(ep)+"_"+opt+".pt")
    dec_name = os.path.join(base_name, "dec_attn_ep"+str(ep)+"_"+opt+".pt")
    if os.path.isfile(enc_name):
        model.encoder.load_state_dict(torch.load(enc_name))
        print("Loading Encoder weights... : {}".format(enc_name))
    if os.path.isfile(dec_name):
        model.decoder.load_state_dict(torch.load(dec_name))
        print("Loading Decoder weights... : {}".format(dec_name))
    return model


def train_model(features, stroke_names_id, model, dataloaders, criterion, 
                optimizer, scheduler, labs_keys, labs_values, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for bno, (inputs, _, vid_path, stroke, labels) in enumerate(dataloaders[phase]):
                # inputs of shape BATCH x SEQ_LEN x FEATURE_DIM
                labels = attn_utils.get_batch_labels(vid_path, stroke, labs_keys, labs_values)
                # Extract spatio-temporal features from clip using 3D ResNet (For SL >= 16)
#                inputs = inputs.permute(0, 2, 1, 3, 4).float()                
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
#                    batch_size = inputs.size(0)
#                    loss = criterion(outputs, labels)
#                    dec_h = h
#                    dec_out_lst = []
#                    target_length = targets.size(1)      # assign SEQ_LEN as target length for now
#                    # run for each word of the sequence (use teacher forcing)
#                    for ti in range(target_length):
#                        dec_out, dec_h, dec_attn = decoder(dec_h, enc_out, targets[:,ti,:])
#                        dec_out_lst.append(dec_out)
#                        
#                        #decoder_input = target_tensor[di]  # Teacher forcing
#            
#                    outputs = torch.stack(dec_out_lst, dim=1)
                    dec_in = torch.zeros(inputs.size(0), inputs.size(2))  # (BATCH, OUTPUT)
                    dec_in = dec_in.to(device)
                    
                    outputs, dec_h, wts = model(inputs, dec_in)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)     #torch.flip(targets, [1])

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
#                print("Iter : {} :: Running Loss : {}".format(bno, running_loss))
                running_corrects += torch.sum(preds == labels.data)
                
#                print("Batch No : {} / {}".format(bno, len(dataloaders[phase])))
#                if (bno+1) % 10 == 0:
#                    break
                    
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

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
    return model

def predict(features, stroke_names_id, encoder, decoder, dataloaders, labs_keys, 
            labs_values, phase="val"):
    assert phase == "val" or phase=="test", "Incorrect Phase."
    encoder = encoder.eval()
    decoder = decoder.eval()
    # Iterate over data.
    for bno, (inputs, targets, vid_path, stroke, labels) in enumerate(dataloaders[phase]):
        # inputs of shape BATCH x SEQ_LEN x FEATURE_DIM
        labels = attn_utils.get_batch_labels(vid_path, stroke, labs_keys, labs_values)
        # Extract spatio-temporal features from clip using 3D ResNet (For SL >= 16)
#                inputs = inputs.permute(0, 2, 1, 3, 4).float()
        
        inputs, targets = inputs.to(device), targets.to(device)
        labels = labels.to(device)
        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            
            batch_size = inputs.size(0)
            enc_h = encoder.init_hidden(batch_size)
            enc_out, h = encoder(inputs, enc_h)
            dec_h = h
            dec_in = torch.zeros(batch_size, targets.size(2)).to(device)
            dec_out_lst = []
            target_length = targets.size(1)      # assign SEQ_LEN as target length for now
            # run for each word of the sequence (use teacher forcing)
            for ti in range(target_length):
                dec_out, dec_h, dec_attn = decoder(dec_h, enc_out, dec_in)
                dec_out_lst.append(dec_out)
#                loss += criterion(dec_out, targets[:,ti,:])
                dec_in = dec_out
    
            outputs = torch.stack(dec_out_lst, dim=1)
            
#                    outputs, dec_h, wts = model(inputs, inputs)
#                    _, preds = torch.max(outputs, 1)
#                    loss = criterion(outputs, targets)     #torch.flip(targets, [1])

        # statistics
#        running_loss += loss.item()
#                print("Iter : {} :: Running Loss : {}".format(bno, running_loss))
#                running_corrects += torch.sum(preds == labels.data)
        
#                print("Batch No : {} / {}".format(bno, len(dataloaders[phase])))
#                if (bno+1) % 10 == 0:
#                    break
            
#            if phase == 'train':
#                scheduler.step()

#    epoch_loss = running_loss #/ len(dataloaders[phase].dataset)
#            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

#    print('{} Loss: {:.4f}'.format(phase, epoch_loss))
    

def train_attn_model(DATASET, LABELS, CLASS_IDS, BATCH_SIZE, ANNOTATION_FILE, 
                   SEQ_SIZE=16, STEP=16, nstrokes=-1, N_EPOCHS=25, base_name=""):
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
#    clip_transform = transforms.Compose([videotransforms.CenterCrop(224),
#                                         videotransforms.ToPILClip(), 
#                                         videotransforms.Resize((112, 112)),
##                                         videotransforms.RandomCrop(112), 
#                                         videotransforms.ToTensor(), 
#                                         videotransforms.Normalize(),
#                                        #videotransforms.RandomHorizontalFlip(),\
#                                        ])
    ft_path = os.path.join(base_name, ft_dir, feat)
    train_dataset = StrokeFeaturePairsDataset(ft_path, train_lst, DATASET, LABELS, CLASS_IDS, 
                                         frames_per_clip=SEQ_SIZE, extracted_frames_per_clip=2,
                                         step_between_clips=STEP, train=True)
    ft_path_val = os.path.join(base_name, ft_dir, feat_val)
    val_dataset = StrokeFeaturePairsDataset(ft_path_val, val_lst, DATASET, LABELS, CLASS_IDS, 
                                         frames_per_clip=SEQ_SIZE, extracted_frames_per_clip=2,
                                         step_between_clips=STEP, train=False)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    data_loaders = {"train": train_loader, "test": val_loader}

    ###########################################################################
    
    labs_keys, labs_values = attn_utils.get_cluster_labels(ANNOTATION_FILE)
    
    num_classes = len(list(set(labs_values)))
    
    ###########################################################################    
    
    # load model and set loss function    
    model = attn_model.AttentionEncoderDecoderClassifier(INPUT_SIZE, HIDDEN_SIZE, 
                                                         TARGET_SIZE, SEQ_SIZE-2+1)
    
    model = load_weights(base_name, model, N_EPOCHS, "Adam")
    
#    for ft in model.parameters():
#        ft.requires_grad = False
        
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    
#    # Layers to finetune. Last layer should be displayed
#    params_to_update = model.parameters()
    print("Fix Weights : ")
    
    model = model.to(device)
    for name, param in model.encoder.named_parameters():
        print("Encoder : {}".format(name))
        param.requires_grad = False
    for name, param in model.decoder.named_parameters():
        print("Decoder : {}".format(name))
        param.requires_grad = False
#    model.decoder.train(False)
    print("Params to learn:")
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
    
    # Observe that all parameters are being optimized
    optimizer_ft = torch.optim.Adam(params_to_update, lr=0.001)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = StepLR(optimizer_ft, step_size=8, gamma=0.1)
    
#    # Observe that all parameters are being optimized
#    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    
    features, stroke_names_id = attn_utils.read_feats(os.path.join(base_name, ft_dir), feat, snames)
    
    ###########################################################################
    # Training the model    
    
    start = time.time()
    
    model = train_model(features, stroke_names_id, model, data_loaders, criterion, 
                        optimizer_ft, exp_lr_scheduler, labs_keys, labs_values,
                        num_epochs=N_EPOCHS)
    
    end = time.time()
    
    # save the best performing model
    save_model_checkpoint(base_name, model, N_EPOCHS, "Adam")
    # Load model checkpoints
    
    print("Total Execution time for {} epoch : {}".format(N_EPOCHS, (end-start)))

#    ###########################################################################
    
    features_val, stroke_names_id_val = attn_utils.read_feats(os.path.join(base_name, ft_dir), 
                                                   feat_val, snames_val)
    
    predict(features_val, stroke_names_id_val, model, data_loaders, labs_keys, 
            labs_values, phase='test')
    
    return None, None

if __name__ == '__main__':
    # Local Paths
    LABELS = "/home/arpan/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
    DATASET = "/home/arpan/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
    CLASS_IDS = "/home/arpan/VisionWorkspace/Cricket/cluster_strokes/configs/Class Index_Strokes.txt"    
    ANNOTATION_FILE = "/home/arpan/VisionWorkspace/Cricket/CricketStrokeLocalizationBOVW/shots_classes.txt"

    SEQ_SIZE = 6
    STEP = 5
    NUM_LAYERS = 2
    BATCH_SIZE = 32
    N_EPOCHS = 25
    
    trajectories, stroke_names = train_attn_model(DATASET, LABELS, CLASS_IDS, 
                                                BATCH_SIZE, ANNOTATION_FILE,
                                                SEQ_SIZE, STEP, nstrokes=-1, 
                                                N_EPOCHS=N_EPOCHS, base_name=base_path)



