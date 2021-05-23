#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 10:39:42 2021

@author: arpan

@Description: Training Siamese I3D net in a self supervision approach.
"""
import os
import sys
import argparse
sys.path.insert(0, '../../pytorch-i3d')
sys.path.insert(0, '../cluster_strokes')
sys.path.insert(0, '../cluster_strokes/lib')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from datasets.dataset import CricketStrokesDataset
from datasets.dataset_selfsupervised import CricketStrokeClipsDataset
from models.contrastive import ContrastiveLoss
#import datasets.videotransforms as videotransforms

import datasets.videotransforms as T
#import videotransforms as T
from torchvision import transforms
import numpy as np
import time
#import random

from utils import autoenc_utils
import pickle
#import attn_model
#import conv_selfexp_model as mod
#import attn_model
import siamese_net
import attn_utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

log_path = "logs/bov_i3d_selfsup"
#feat_path = "/home/arpan/VisionWorkspace/Cricket/CricketStrokeLocalizationBOVW/logs/bow_HL_ofAng_grid20"

def save_model_checkpoint(base_name, model, ep, opt):
    """
    Save the optimizer state with epoch no. along with the weights
    """
    # Save only the model params
    model_name = os.path.join(base_name, "i3d_ep"+str(ep)+"_"+opt+".pt")

    torch.save(model.state_dict(), model_name)
    print("Model saved to disk... : {}".format(model_name))

def load_weights(base_name, model, ep, opt):
    """
    Load the pretrained weights to the models' encoder and decoder modules
    """
    # Paths to encoder and decoder files
    model_name = os.path.join(base_name, "i3d_ep"+str(ep)+"_"+opt+".pt")
    if os.path.isfile(model_name):
        model.load_state_dict(torch.load(model_name))
        print("Loading I3D weights... : {}".format(model_name))
    return model


def train_model(model, dataloaders, criterion, optimizer, scheduler, labs_keys,
                labs_values, num_epochs=25):
    since = time.time()

#    best_model_wts = copy.deepcopy(model.state_dict())
#    best_acc = 0.0

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
#            running_corrects = 0.0
            # Iterate over data.
            for bno, (x1, x2, vid_path1, stroke1, vid_path2, stroke2, labels) in enumerate(dataloaders[phase]):
                # inputs of shape BATCH x SEQ_LEN x FEATURE_DIM
#                labels = attn_utils.get_batch_labels(vid_path, stroke, labs_keys, labs_values, 1)
                labels = [lab for lab in labels.tolist() for i in range(1)]
                # Extract spatio-temporal features from clip using 3D ResNet (For SL >= 16)
                
                x1, x2 = x1.float(), x2.float()
                x1 = x1.to(device)
                x2 = x2.to(device)
                labels = torch.FloatTensor(labels).to(device)
                x1 = x1.permute(0, 2, 1, 3, 4).float().to(device)      # for Raw Crops
                x2 = x2.permute(0, 2, 1, 3, 4).float().to(device)      # for Raw Crops
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                output1, output2 = model(x1, x2)  # output size (BATCH, hidden_size)
#                output1 = output1.view(-1, output1.shape[-1])
##                output2 = F.softmax(output2.view(-1, output2.shape[-1]), dim=1)
#                output2 = output2.view(-1, output2.shape[-1])
                loss = criterion(output1.squeeze(axis=2), output2.squeeze(axis=2), labels)
                
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
#                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                
#                # track history if only in train
#                _, preds = torch.max(output, 1)

#                # statistics
                running_loss += loss.item() #* inputs.size(0)
                print("Iter : {} :: Running Loss : {}".format(bno, running_loss))
#                running_corrects += torch.sum(preds == targets.data)
                if bno==100:
                    break
                
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
#            epoch_acc = running_corrects.double() #/ len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} LR: {}'.format(phase, epoch_loss,  
                  scheduler.get_lr()[0]))
##            # deep copy the model for best test accuracy
#            if phase == 'test' and epoch_acc > best_acc:
#                best_acc = epoch_acc
#                best_model_wts = copy.deepcopy(model.state_dict())

        print()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, \
          time_elapsed % 60))
#    print('Best val Acc: {:4f}'.format(best_acc))

#    # load best model weights
#    model.load_state_dict(best_model_wts)
    return model
    


def main(DATASET, LABELS, CLASS_IDS, BATCH_SIZE, ANNOTATION_FILE, SEQ_SIZE=16, STEP=16, 
         nstrokes=-1, N_EPOCHS=25, base_name=''):
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
    
    attn_utils.seed_everything(1234)
    
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    
    # Read the strokes 
    # Divide the highlight dataset files into training, validation and test sets
    train_lst, val_lst, test_lst = autoenc_utils.split_dataset_files(DATASET)
    print("No. of training videos : {}".format(len(train_lst)))
    
#    features, stroke_names_id = attn_utils.read_feats(feat_path, feat, snames)

#    ###########################################################################
#    
#    features_val, stroke_names_id_val = attn_utils.read_feats(feat_path, feat_val, 
#                                                              snames_val)
    ###########################################################################
    # Create a Dataset    
    train_transforms = transforms.Compose([T.CenterCrop(300),
                                           T.ToPILClip(),
                                           T.Resize((224, 224)),
                                           T.ToTensor(),
                                           T.Normalize(),
    ])
    test_transforms = transforms.Compose([T.CenterCrop(300),
                                           T.ToPILClip(),
                                           T.Resize((224, 224)),
                                           T.ToTensor(),
                                           T.Normalize(),
                                          ])
    
#    ft_path = os.path.join(base_name, feat_path, feat)
    train_dataset = CricketStrokeClipsDataset(train_lst, DATASET, LABELS, CLASS_IDS, 
                                         frames_per_clip=SEQ_SIZE, extracted_frames_per_clip=16, 
                                         step_between_clips=STEP, 
                                         train=True, framewiseTransform=False,
                                         transform=train_transforms)
#    ft_path_val = os.path.join(base_name, feat_path, feat_val)
    val_dataset = CricketStrokeClipsDataset(val_lst, DATASET, LABELS, CLASS_IDS, 
                                         frames_per_clip=SEQ_SIZE, extracted_frames_per_clip=16, 
                                         step_between_clips=STEP, 
                                         train=False, framewiseTransform=False,
                                         transform=test_transforms)
    
    # get labels
    labs_keys, labs_values = attn_utils.get_cluster_labels(ANNOTATION_FILE)
#    # created weighted Sampler for class imbalance
#    samples_weight = attn_utils.get_sample_weights(train_dataset, labs_keys, labs_values, 
#                                                   train_lst)
#    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True) 
#                              sampler=sampler, worker_init_fn=np.random.seed(12))
    
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    data_loaders = {"train": train_loader, "test": val_loader}

    num_classes = len(list(set(labs_values)))
        
    ###########################################################################    

    # load model and set loss function
    model = siamese_net.SiameseI3DNet(400, in_channels=3)
    model.i3d.load_state_dict(torch.load('/home/arpan/VisionWorkspace/pytorch-i3d/models/rgb_imagenet.pt'))
#    model.i3d.replace_logits(2)
#    model = load_weights(log_path, model, N_EPOCHS, 
#                                    "S"+str(SEQ_SIZE)+"C"+str(cluster_size)+"_SGD")
    
    lr = 0.1
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
#    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [10, 25]) # [300, 1000])
    # Decay LR by a factor of 0.1 every 7 epochs
    lr_sched = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # Setup the loss fxn
    criterion = ContrastiveLoss()
    model = model.to(device)
#    print("Params to learn:")
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
    
#    # Observe that all parameters are being optimized
##    optimizer_ft = torch.optim.Adam(model.parameters(), lr=0.001)
#    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#    
#    # Decay LR by a factor of 0.1 every 7 epochs
#    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    
#    lr = 5.0 # learning rate
#    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
#    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)    
    ###########################################################################
    # Training the model    
    
    start = time.time()
    
    model = train_model(model, data_loaders, criterion, optimizer, lr_sched, 
                        labs_keys, labs_values, num_epochs=N_EPOCHS)
    
    end = time.time()
    
#    # save the best performing model
    save_model_checkpoint(log_path, model, N_EPOCHS, 
                                     "S"+str(SEQ_SIZE)+"_SGD")
    # Load model checkpoints
    model = load_weights(log_path, model, N_EPOCHS, 
                                    "S"+str(SEQ_SIZE)+"_SGD")
    
    print("Total Execution time for {} epoch : {}".format(N_EPOCHS, (end-start)))

#    ###########################################################################
    
#    acc = predict(features_val, stroke_names_id_val, model, data_loaders, labs_keys, 
#                  labs_values, SEQ_SIZE, phase='test')
    
    ###########################################################################
    
#    # Extract attention model features 
#    if not os.path.isfile(os.path.join(log_path, "siamgru_feats.pkl")):
#        if not os.path.exists(log_path):
#            os.makedirs(log_path)
#        #    # Extract Grid OF / HOOF features {mth = 2, and vary nbins}
#        print("Training extraction ... ")
#        feats_dict, stroke_names = extract_trans_feats(model, DATASET, LABELS, 
#                                                      CLASS_IDS, BATCH_SIZE, SEQ_SIZE, 
#                                                      SEQ_SIZE-1, partition='train', nstrokes=nstrokes, 
#                                                      base_name=log_path)
#
#        with open(os.path.join(log_path, "siamgru_feats.pkl"), "wb") as fp:
#            pickle.dump(feats_dict, fp)
#        with open(os.path.join(log_path, "siamgru_snames.pkl"), "wb") as fp:
#            pickle.dump(stroke_names, fp)
#                
#    if not os.path.isfile(os.path.join(log_path, "siamgru_feats_val.pkl")):
#        print("Validation extraction ....")
#        feats_dict_val, stroke_names_val = extract_trans_feats(model, DATASET, LABELS, 
#                                                      CLASS_IDS, BATCH_SIZE, SEQ_SIZE, 
#                                                      SEQ_SIZE-1, partition='val', nstrokes=nstrokes, 
#                                                      base_name=log_path)
#
#        with open(os.path.join(log_path, "siamgru_feats_val.pkl"), "wb") as fp:
#            pickle.dump(feats_dict_val, fp)
#        with open(os.path.join(log_path, "siamgru_snames_val.pkl"), "wb") as fp:
#            pickle.dump(stroke_names_val, fp)
#            
#    if not os.path.isfile(os.path.join(log_path, "siamgru_feats_test.pkl")):
#        print("Testing extraction ....")
#        feats_dict_val, stroke_names_val = extract_trans_feats(model, DATASET, LABELS, 
#                                                      CLASS_IDS, BATCH_SIZE, SEQ_SIZE, 
#                                                      SEQ_SIZE-1, partition='test', nstrokes=nstrokes, 
#                                                      base_name=log_path)
#
#        with open(os.path.join(log_path, "siamgru_feats_test.pkl"), "wb") as fp:
#            pickle.dump(feats_dict_val, fp)
#        with open(os.path.join(log_path, "siamgru_snames_test.pkl"), "wb") as fp:
#            pickle.dump(stroke_names_val, fp)
    
    # call count_paramters(model)  for displaying total no. of parameters
    print("#Parameters : {} ".format(autoenc_utils.count_parameters(model)))
    return 0


if __name__ == '__main__':
    # Local Paths
    LABELS = "/home/arpan/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
    DATASET = "/home/arpan/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
    CLASS_IDS = "/home/arpan/VisionWorkspace/Cricket/cluster_strokes/configs/Class Index_Strokes.txt"    
    ANNOTATION_FILE = "/home/arpan/VisionWorkspace/Cricket/CricketStrokeLocalizationBOVW/shots_classes.txt"
    
    base_path = "/home/arpan/VisionWorkspace/Cricket/StrokeAttention/logs"
    
    seq_sizes = range(16, 17, 1)
    STEP = 1
    BATCH_SIZE = 2
    N_EPOCHS = 30
    
    attn_utils.seed_everything(1234)
    acc = []

    print("Raw Frame Sequence Learning using self-supervision...")
    print("EPOCHS = {} ".format(N_EPOCHS))
    for SEQ_SIZE in seq_sizes:
        print("SEQ_SIZE : {}".format(SEQ_SIZE))
        acc.append(main(DATASET, LABELS, CLASS_IDS, BATCH_SIZE, ANNOTATION_FILE, 
                        SEQ_SIZE, STEP, nstrokes=-1, N_EPOCHS=N_EPOCHS, 
                        base_name=base_path))
        
    print("*"*60)
    print("SEQ_SIZES : {}".format(seq_sizes))
    print("Accuracy values : {}".format(acc))



#
#def extract_trans_feats(model, DATASET, LABELS, CLASS_IDS, BATCH_SIZE, 
#                       SEQ_SIZE=16, STEP=16, partition='train', nstrokes=-1, base_name=""):
#    '''
#    Extract sequence features from AutoEncoder.
#    
#    Parameters:
#    -----------
#    encoder, decoder : attn_model.Encoder 
#        relative path to the checkpoint file for Autoencoder
#    DATASET : str
#        path to the video dataset
#    LABELS : str
#        path containing stroke labels
#    BATCH_SIZE : int
#        size for batch of clips
#    SEQ_SIZE : int
#        no. of frames in a clip
#    STEP : int
#        stride for next example. If SEQ_SIZE=16, STEP=8, use frames (0, 15), (8, 23) ...
#    partition : str
#        'train' / 'test' / 'val' : Videos to be considered
#    nstrokes : int
#        partial extraction of features (do not execute for entire dataset)
#    base_name : str
#        path containing the pickled feature dumps
#    
#    Returns:
#    --------
#    features_dictionary, stroke_names
#    
#    '''
#    
#    ###########################################################################
#    # Read the strokes 
#    # Divide the highlight dataset files into training, validation and test sets
#    train_lst, val_lst, test_lst = autoenc_utils.split_dataset_files(DATASET)
#    print("No. of training videos : {}".format(len(train_lst)))
#    
#    #####################################################################
#    
#    if partition == 'train':
#        partition_lst = train_lst
#        ft_path = os.path.join(base_name, feat_path, feat)
#    elif partition == 'val':
#        partition_lst = val_lst
#        ft_path = os.path.join(base_name, feat_path, feat_val)
#    elif partition == 'test':
#        partition_lst = test_lst
#        ft_path = os.path.join(base_name, feat_path, feat_test)
#    else:
#        print("Partition should be : train / val / test")
#        return
#    
#    ###########################################################################
#    # Create a Dataset
#    
#    part_dataset = StrokeFeatureSequenceDataset(ft_path, partition_lst, DATASET, LABELS, CLASS_IDS, 
#                                         frames_per_clip=SEQ_SIZE, extracted_frames_per_clip=2,
#                                         step_between_clips=STEP, train=True)
#    
#    data_loader = DataLoader(dataset=part_dataset, batch_size=BATCH_SIZE, shuffle=False)
#
#    ###########################################################################
#    # Validate / Evaluate
#    model.eval()
#    stroke_names = []
#    trajectories, stroke_traj = [], []
#    num_strokes = 0
#    prev_stroke = None
#    print("Total Batches : {} :: BATCH_SIZE : {}".format(data_loader.__len__(), BATCH_SIZE))
#    ###########################################################################
#    for bno, (inputs, vid_path, stroke, _) in enumerate(data_loader):
#        # inputs of shape BATCH x SEQ_LEN x FEATURE_DIM
#        inputs = inputs.float()
##        inp_emb = attn_utils.get_long_tensor(inputs)    # comment out for SA
##        inputs = inp_emb.to(device)                     # comment out for SA
#        inputs = inputs.to(device)
#        
#        # forward
#        # track history if only in train
#        with torch.set_grad_enabled(False):
#            
#            hid = model.init_hidden(inputs.shape[0])
#            outputs, hidden = model.forward_once(inputs, hid)  # output size (BATCH, SEQ_SIZE, HIDDEN_SIZE)
#        
#        if len(outputs.size()) == 2:
#            outputs = outputs[:, None, :]
#
#        # convert to start frames and end frames from tensors to lists
#        stroke = [s.tolist() for s in stroke]
#        # outputs are the reconstructed features. Use compressed enc_out values(maybe wtd.).
#        inputs_lst, batch_stroke_names = autoenc_utils.separate_stroke_tensors(outputs, \
#                                                                    vid_path, stroke)
#        
#        # for sequence of features from batch segregated extracted features.
#        if bno == 0:
#            prev_stroke = batch_stroke_names[0]
#        
#        for enc_idx, enc_input in enumerate(inputs_lst):
#            # get no of sequences that can be extracted from enc_input tensor
#            nSeqs = enc_input.size(0)
#            if prev_stroke != batch_stroke_names[enc_idx]:
#                # append old stroke to trajectories
#                if len(stroke_traj) > 0:
#                    num_strokes += 1
#                    trajectories.append(stroke_traj)
#                    stroke_names.append(prev_stroke)
#                    stroke_traj = []
#            
##            enc_output = model.encoder(enc_input.to(device))
##            enc_output = enc_output.squeeze(axis=1).cpu().data.numpy()
#            enc_output = enc_input.cpu().data.numpy()
#            
#            # convert to [[[stroke1(size 32 each) ... ], [], ...], [ [], ... ]]
#            stroke_traj.extend([enc_output[i,j,:] for i in range(enc_output.shape[0]) \
#                                                for j in range(enc_output.shape[1])])
#            prev_stroke = batch_stroke_names[enc_idx]
#            
#        if nstrokes > -1 and num_strokes >= nstrokes:
#            break
#       
#    # for last batch only if extracted for full dataset
#    if len(stroke_traj) > 0 and nstrokes < 0:
#        trajectories.append(stroke_traj)
#        stroke_names.append(batch_stroke_names[-1])
#    
#    # convert to dictionary of features with keys as stroke names(with ext). 
#    features = {}
#    for i, t in enumerate(trajectories):
#        features[stroke_names[i]] = np.array(t)
#    
##    trajectories, stroke_names = autoenc_utils.group_strokewise(trajectories, stroke_names)
#    
#    return features, stroke_names