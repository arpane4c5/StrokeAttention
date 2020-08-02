#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 23:43:52 2020

@author: arpan

@Description: Train an Attention model on strokes
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_path = "/home/arpan/VisionWorkspace/Cricket/CricketStrokeLocalizationBOVW/logs"
ft_dir = "bow_HL_ofAng_grid20"
feat, feat_val = "of_feats_grid20.pkl", "of_feats_val_grid20.pkl"
snames, snames_val = "of_snames_grid20.pkl", "of_snames_val_grid20.pkl"
INPUT_SIZE, TARGET_SIZE = 576, 576      # OFGRID: 576, 3DCNN: 512, 2DCNN: 2048
HIDDEN_SIZE = 256 #64#1024


def get_cluster_labels(cluster_labels_path):
    labs_keys = []
    labs_values = []
    with open(cluster_labels_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:    
            #print("{} :: Class : {}".format(row[0], row[1]))
            labs_keys.append(row[0])
            labs_values.append(int(row[1]))
            line_count += 1
        print("Read {} ground truth stroke labels from file.".format(line_count))
        
    if min(labs_values) == 1:
        labs_values = [l-1 for l in labs_values]
        labs_keys = [k.replace('.avi', '') for k in labs_keys]
    return labs_keys, labs_values

def get_batch_labels(vid_path, stroke, labs_keys, labs_values):
    
    labels = []
    for i, vid_name in enumerate(vid_path):
        
        vid_name = vid_name.rsplit('/', 1)[1]
        if '.avi' in vid_name:
            vid_name = vid_name.replace('.avi', '')
        elif '.mp4' in vid_name:
            vid_name = vid_name.replace('.mp4', '')
        stroke_name = vid_name+"_"+str(stroke[0][i].item())+"_"+str(stroke[1][i].item())
        assert stroke_name in labs_keys, "Key does not match : {}".format(stroke_name)
        labels.append(labs_values[labs_keys.index(stroke_name)])
        
    return torch.tensor(labels)
        
def save_model_checkpoint(base_name, model, ep, opt):
    """
    TODO: save the optimizer state with epoch no. along with the weights
    """
    # Save only the model params
    name = os.path.join(base_name, "3d_attn_autoenc_ep"+str(ep)+"_"+opt+".pt")
#    if use_gpu and torch.cuda.device_count() > 1:
#        model = model.module    # good idea to unwrap from DataParallel and save

    torch.save(model.state_dict(), name)
    print("Model saved to disk... {}".format(name))
    
def read_feats(base_name, feat, snames):
    with open(os.path.join(base_name, feat), "rb") as fp:
        features = pickle.load(fp)
    with open(os.path.join(base_name, snames), "rb") as fp:
        strokes_name_id = pickle.load(fp)
        
    return features, strokes_name_id

def read_batch_feats(features, stroke_names_id, batch):
    
    return

def train_model(features, stroke_names_id, encoder, decoder, dataloaders, criterion, 
                encoder_optimizer, decoder_optimizer, scheduler, labs_keys, 
                labs_values, num_epochs=25):
    since = time.time()

#    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
#            if phase == 'train':
#                model.train()  # Set model to training mode
#            else:
#                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for bno, (inputs, targets, vid_path, stroke, labels) in enumerate(dataloaders[phase]):
                # inputs of shape BATCH x SEQ_LEN x FEATURE_DIM
                labels = get_batch_labels(vid_path, stroke, labs_keys, labs_values)
                # Extract spatio-temporal features from clip using 3D ResNet (For SL >= 16)
#                inputs = inputs.permute(0, 2, 1, 3, 4).float()
                
                inputs, targets = inputs.to(device), targets.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                loss = 0

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
                    batch_size = inputs.size(0)
                    enc_h = encoder.init_hidden(batch_size)
                    enc_out, h = encoder(inputs, enc_h)
                    dec_h = h
                    dec_out_lst = []
                    target_length = targets.size(1)      # assign SEQ_LEN as target length for now
                    # run for each word of the sequence (use teacher forcing)
                    for ti in range(target_length):
                        dec_out, dec_h, dec_attn = decoder(dec_h, enc_out, targets[:,ti,:])
                        dec_out_lst.append(dec_out)
                        loss += criterion(dec_out, targets[:,ti,:])
                        #decoder_input = target_tensor[di]  # Teacher forcing
            
                    outputs = torch.stack(dec_out_lst, dim=1)
                    
#                    outputs, dec_h, wts = model(inputs, inputs)
#                    _, preds = torch.max(outputs, 1)
#                    loss = criterion(outputs, targets)     #torch.flip(targets, [1])

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        encoder_optimizer.step()
                        decoder_optimizer.step()

                # statistics
                running_loss += loss.item()
#                print("Iter : {} :: Running Loss : {}".format(bno, running_loss))
#                running_corrects += torch.sum(preds == labels.data)
                
#                print("Batch No : {} / {}".format(bno, len(dataloaders[phase])))
#                if (bno+1) % 10 == 0:
#                    break
                    
#            if phase == 'train':
#                scheduler.step()

            epoch_loss = running_loss #/ len(dataloaders[phase].dataset)
#            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

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
    return encoder, decoder


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
    
    labs_keys, labs_values = get_cluster_labels(ANNOTATION_FILE)
    
    num_classes = len(list(set(labs_values)))
    
    ###########################################################################    
    # load model and set loss function
#    model = torchvision.models.video.r3d_18(pretrained=True, progress=True)
    
    bidirectional = False
    encoder = attn_model.Encoder(INPUT_SIZE, HIDDEN_SIZE, 1, bidirectional)
    decoder = attn_model.AttentionDecoder(HIDDEN_SIZE*(1+bidirectional), TARGET_SIZE, 
                               max_length=SEQ_SIZE-2+1)
#    model = attn_model.AttentionEncoderDecoder(INPUT_SIZE, HIDDEN_SIZE, TARGET_SIZE, SEQ_SIZE-16+1)
#    model = attn_model.Encoder(10, 20, bidirectional)
    
#    for ft in model.parameters():
#        ft.requires_grad = False
#    
#    inp_feat_size = model.fc.in_features
#    model.fc = nn.Linear(inp_feat_size, num_classes)
#    model = model.to(device)
    encoder, decoder = encoder.to(device), decoder.to(device)
    
#    # load checkpoint:
#    if os.path.isfile(os.path.join(base_name, "3dresnet18_ep"+str(N_EPOCHS)+"_Adam.pt")):
#        model.load_state_dict(torch.load(os.path.join(base_name, "3dresnet18_ep"+str(N_EPOCHS)+"_Adam.pt")))
    
    # Setup the loss fxn
#    criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    
#    # Layers to finetune. Last layer should be displayed
#    params_to_update = model.parameters()
#    print("Params to learn:")
#    
#    params_to_update = []
#    for name, param in model.named_parameters():
#        if param.requires_grad == True:
#            params_to_update.append(param)
#            print("\t",name)
    
    # Observe that all parameters are being optimized
#    optimizer_ft = torch.optim.Adam(params_to_update, lr=0.001)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = StepLR(encoder_optimizer, step_size=8, gamma=0.1)
    
#    # Observe that all parameters are being optimized
#    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    
    features, stroke_names_id = read_feats(os.path.join(base_name, ft_dir), feat, snames)
    
#    ###########################################################################
    # Training the model    
    
    start = time.time()
    
    model_ft = train_model(features, stroke_names_id, encoder, decoder, data_loaders, criterion, 
                           encoder_optimizer, decoder_optimizer, exp_lr_scheduler, labs_keys, labs_values,
                           num_epochs=N_EPOCHS)
        
    end = time.time()
    
    # save the best performing model
#    save_model_checkpoint(base_name, model_ft, N_EPOCHS, "Adam")
    
    print("Total Execution time for {} epoch : {}".format(N_EPOCHS, (end-start)))
    
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
    base_path = "/home/arpan/VisionWorkspace/Cricket/CricketStrokeLocalizationBOVW/logs"
    
    trajectories, stroke_names = train_attn_model(DATASET, LABELS, CLASS_IDS, 
                                                BATCH_SIZE, ANNOTATION_FILE,
                                                SEQ_SIZE, STEP, nstrokes=-1, 
                                                N_EPOCHS=N_EPOCHS, base_name=base_path)



