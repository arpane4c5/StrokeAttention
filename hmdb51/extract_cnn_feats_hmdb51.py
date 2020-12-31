#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 08:27:02 2020

@author: arpan
"""
#import _init_paths

import sys
# path to datasets, features, models
sys.path.insert(0, '../../../pytorch-i3d')
sys.path.insert(0, '../../cluster_strokes')
sys.path.insert(0, '../../cluster_strokes/lib')

import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import dataset_hmdb as hmdb

from features.resnet_feature_extracter import Img2Vec, Clip2Vec
from utils import autoenc_utils
import datasets.videotransforms as videotransforms
from models import autoenc
#from datasets.dataset import CricketStrokesDataset
import datasets.videotransforms as T #videotransforms as T
import pickle
from pytorch_i3d import InceptionI3d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def separate_video_tensors(inputs, vid_path):
    '''
    Separate the features of different strokes in a lists. Similar to 
    separate_stroke_tensors() for Cricket dataset extraction but for trimmed videos.
    
    Parameters:
    -----------
    inputs : Torch.tensor (size = batch x feature_width)
        tensor for a batch of sequence inputs, extracted from feature extractor
    vid_path : tuple of str (len = batch)
        tuples of stroke paths, got from data_loader
        
    Returns:
    --------
    list of Torch.tensors for different videos
    
    '''
    from collections import Counter
    
    n_frms_dict = Counter(vid_path)
    n_frms_keys = [x for i, x in enumerate(vid_path) if i == vid_path.index(x)]
    stroke_vectors, stroke_names = [], []
    
    in_idx = 0
    for i, vid_key in enumerate(n_frms_keys):
        stroke_vectors.append(inputs[in_idx:(in_idx + n_frms_dict[vid_key]), ...])
        stroke_names.append(vid_path[in_idx])
        in_idx += n_frms_dict[vid_key]
    return stroke_vectors, stroke_names

def extract_2DCNN_feats(DATASET, LABELS, BATCH_SIZE, STEP=1, foldno=1, train=True, 
                        nclasses=51, model_path=None, nstrokes=-1):
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
        Autoencoder.
    partition : str
        'all' / 'train' / 'test' / 'val' : Videos to be considered
    nstrokes : int
        partial extraction of features (if don't want to execute for entire dataset)
    
    Returns:
    --------
    trajectories, stroke_names
    
    '''
    
    ###########################################################################
    
    ###########################################################################
    # Create a Dataset    
    # Frame-wise transform
    clip_transform = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                              std=[0.229, 0.224, 0.225]),]) 
    # For using Frame level transform, the framewiseTransform flag turned on
    part_dataset = hmdb.HMDB51(DATASET, LABELS, 1, step_between_clips = STEP, 
                               fold=foldno, train=train, framewiseTransform=True,
                               transform=clip_transform)
    
    data_loader = DataLoader(dataset=part_dataset, batch_size=BATCH_SIZE, shuffle=False)

    ###########################################################################
    # Extract using the data_loader
    stroke_names = []
    trajectories, stroke_traj = [], []
    num_strokes = 0
    extractor = Img2Vec()
    #INPUT_SIZE = extractor.layer_output_size
    prev_stroke = None
    
    print("Total Batches : {} :: BATCH_SIZE : {}".format(data_loader.__len__(), BATCH_SIZE))
    for bno, (inputs, vid_path, start_pts, end_pts, _) in enumerate(data_loader):
        # get video clips (B, SL, C, H, W)
        print("Batch No : {}".format(bno))
        # Extract spatial features using 2D ResNet
        inputs = torch.stack([extractor.get_vec(x) for x in inputs])
            
        # convert to start frames and end frames from tensors to lists
#        stroke = [s.tolist() for s in stroke]
        inputs_lst, batch_stroke_names = separate_video_tensors(inputs, vid_path)
        
        if bno == 0:
            prev_stroke = batch_stroke_names[0]
        
        for enc_idx, enc_input in enumerate(inputs_lst):
            # get no of sequences that can be extracted from enc_input tensor
            if prev_stroke != batch_stroke_names[enc_idx]:
                # append old stroke to trajectories
                if len(stroke_traj) > 0:
                    num_strokes += 1
                    trajectories.append(stroke_traj)
                    stroke_names.append(prev_stroke)
                    stroke_traj = []
            
            # enc_input is same as enc_output while extraction of features.
            enc_output = enc_input
            enc_output = enc_output.squeeze(axis=1).cpu().data.numpy()
            # convert to [[[stroke1(size 32 each) ... ], [], ...], [ [], ... ]]
            stroke_traj.extend([enc_output[i, :] for i in range(enc_output.shape[0])])
            prev_stroke = batch_stroke_names[enc_idx]
            
        if nstrokes >= -1 and num_strokes == nstrokes:
            break
       
    # for last batch only if extracted for full dataset
    if len(stroke_traj) > 0 and nstrokes < 0:
        trajectories.append(stroke_traj)
        stroke_names.append(batch_stroke_names[-1])
        
    traj_dict = {}
    for i, vid in enumerate(stroke_names):
        traj_dict[vid] = np.array(trajectories[i])
    #stroke_vecs, stroke_names =  aggregate_outputs(sequence_outputs, seq_stroke_names)    
    return traj_dict, stroke_names



def extract_3DCNN_feats(DATASET, LABELS, BATCH_SIZE, SEQ_SIZE=16, STEP=16, \
                        foldno=1, train=True, nclasses=51, model_path=None, \
                        nstrokes=-1):
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
    ###########################################################################
    # Create a Dataset    
    # Clip level transform. Use this with framewiseTransform flag turned off
    clip_transform = transforms.Compose([T.CenterCrop(224),
                                         T.ToPILClip(), 
                                         T.Resize((112, 112)),
#                                         T.RandomCrop(112), 
                                         T.ToHMDBTensor(), 
#                                         T.Normalize(),
                                        #T.RandomHorizontalFlip(),\
                                        ])
    part_dataset = hmdb.HMDB51(DATASET, LABELS, SEQ_SIZE, step_between_clips = STEP, 
                               fold=foldno, train=train, transform=clip_transform)
    
    data_loader = DataLoader(dataset=part_dataset, batch_size=BATCH_SIZE, shuffle=False)

    ###########################################################################
    # Validate / Evaluate
    stroke_names = []
    trajectories, stroke_traj = [], []
    num_strokes = 0
    extractor = Clip2Vec(model_path, nclasses)
    #INPUT_SIZE = extractor.layer_output_size
    prev_stroke = None
    
    print("Total Batches : {} :: BATCH_SIZE : {}".format(data_loader.__len__(), BATCH_SIZE))
    assert SEQ_SIZE>=16, "SEQ_SIZE should be >= 16"
    for bno, (inputs, vid_path, start_pts, end_pts, _) in enumerate(data_loader):
        # get video clips (B, SL, C, H, W)
        print("Batch No : {}".format(bno))
        # Extract spatio-temporal features from clip using 3D ResNet (For SL >= 16)
        inputs = inputs.permute(0, 2, 1, 3, 4).float()
        inputs = extractor.get_vec(inputs)
        
        # convert to start frames and end frames from tensors to lists
        inputs_lst, batch_stroke_names = separate_video_tensors(inputs, vid_path)
        
        if bno == 0:
            prev_stroke = batch_stroke_names[0]
        
        for enc_idx, enc_input in enumerate(inputs_lst):
            # get no of sequences that can be extracted from enc_input tensor
            if prev_stroke != batch_stroke_names[enc_idx]:
                # append old stroke to trajectories
                if len(stroke_traj) > 0:
                    num_strokes += 1
                    trajectories.append(stroke_traj)
                    stroke_names.append(prev_stroke)
                    stroke_traj = []
            
            enc_output = enc_input
            enc_output = enc_output.squeeze(axis=1).cpu().data.numpy()
            # convert to [[[stroke1(size 32 each) ... ], [], ...], [ [], ... ]]
            stroke_traj.extend([enc_output[i, :] for i in range(enc_output.shape[0])])
            prev_stroke = batch_stroke_names[enc_idx]
            
                
        if nstrokes >=-1 and num_strokes == nstrokes:
            break
       
    # for last batch only if extracted for full dataset
    if len(stroke_traj) > 0 and nstrokes < 0:
        trajectories.append(stroke_traj)
        stroke_names.append(batch_stroke_names[-1])
        
    # group_strokewise not needed here as videos are trimmed. Was needed for strokes
    # to generate a lists of stroke feature lists for an untrimmed video. 
#    trajectories, stroke_names = group_strokewise(trajectories, stroke_names)
    
    traj_dict = {}
    for i, vid in enumerate(stroke_names):
        traj_dict[vid] = np.array(trajectories[i])
    return traj_dict, stroke_names

def extract_I3D_feats(DATASET, LABELS, BATCH_SIZE, SEQ_SIZE=16, STEP=16, \
                      foldno=1, train=True, nclasses=51, model_path=None, \
                      nstrokes=-1):
    
    clip_transform = transforms.Compose([T.CenterCrop(224),
#                                         T.ToPILClip(), 
#                                         T.Resize((224, 224)),
#                                         T.ToHMDBTensor(), 
#                                         T.Normalize(),
                                        ])
    part_dataset = hmdb.HMDB51(DATASET, LABELS, SEQ_SIZE, step_between_clips = STEP, 
                               fold=foldno, train=train, transform=clip_transform)
    
    data_loader = DataLoader(dataset=part_dataset, batch_size=BATCH_SIZE, shuffle=False)

    i3d = InceptionI3d(51, in_channels=3)
#    i3d.replace_logits(157) # for charades
    if model_path is not None:
        i3d.load_state_dict(torch.load(model_path))
    
    i3d = i3d.cuda()

    # Validate / Evaluate
    stroke_names = []
    trajectories, stroke_traj = [], []
    num_strokes = 0
    
    prev_stroke = None
    
    print("Total Batches : {} :: BATCH_SIZE : {}".format(data_loader.__len__(), BATCH_SIZE))
    assert SEQ_SIZE>=16, "SEQ_SIZE should be >= 16"
    for bno, (inputs, vid_path, start_pts, end_pts, _) in enumerate(data_loader):
        # get video clips (B, SL, C, H, W)
        print("Batch No : {}".format(bno))
        # Extract spatio-temporal features from clip using I3D (For SL >= 16)
        inputs = inputs.permute(0, 4, 1, 2, 3).float().cuda()
        inputs = i3d.extract_features(inputs)  # returned (B, 1024, 1, 1, 1) tensor
        
        # convert to start frames and end frames from tensors to lists
        inputs_lst, batch_stroke_names = separate_video_tensors(inputs.cpu(), vid_path)
        
        if bno == 0:
            prev_stroke = batch_stroke_names[0]
        
        for enc_idx, enc_input in enumerate(inputs_lst):
            # get no of sequences that can be extracted from enc_input tensor
            if prev_stroke != batch_stroke_names[enc_idx]:
                # append old stroke to trajectories
                if len(stroke_traj) > 0:
                    num_strokes += 1
                    trajectories.append(stroke_traj)
                    stroke_names.append(prev_stroke)
                    stroke_traj = []
            
            enc_output = enc_input
            enc_output = enc_output.squeeze(4).squeeze(3).squeeze(2).cpu().data.numpy()
            # convert to [[[stroke1(size 32 each) ... ], [], ...], [ [], ... ]]
            stroke_traj.extend([enc_output[i] for i in range(enc_output.shape[0])])
            prev_stroke = batch_stroke_names[enc_idx]
            
                
        if nstrokes >=-1 and num_strokes == nstrokes:
            break
       
    # for last batch only if extracted for full dataset
    if len(stroke_traj) > 0 and nstrokes < 0:
        trajectories.append(stroke_traj)
        stroke_names.append(batch_stroke_names[-1])
        
    # group_strokewise not needed here as videos are trimmed. Was needed for strokes
    # to generate a lists of stroke feature lists for an untrimmed video. 
#    trajectories, stroke_names = group_strokewise(trajectories, stroke_names)
    
    traj_dict = {}
    for i, vid in enumerate(stroke_names):
        traj_dict[vid] = np.array(trajectories[i])
    return traj_dict, stroke_names


if __name__ == '__main__':
    
    LABELS = "/home/arpan/VisionWorkspace/VideoData/hmdb51/train_test_splits"
    DATASET = "/home/arpan/VisionWorkspace/VideoData/hmdb51/videos"
    
    ###########################################################################
#    # 3DCNN Extraction
#    STEP = 2
#    SEQ_SIZE = 16
#    BATCH_SIZE = 32
#    fold = 1
#    log_path = "feats/3dcnn_hmdb51"
#    if not os.path.isdir(log_path):
#        os.makedirs(log_path)
#    trajectories, vid_names = extract_3DCNN_feats(DATASET, LABELS, BATCH_SIZE, SEQ_SIZE=16, 
#                                                  STEP=STEP, foldno=fold, train=True, 
#                                                  nstrokes=-1)
#    
##    if not os.path.isfile(os.path.join(log_path, "hmdb_3dcnn_feats_train_seq"+str(SEQ_SIZE)+".pkl")):
#    with open(os.path.join(log_path, "hmdb_3dcnn_featsF"+str(fold)+"_train_seq"+str(SEQ_SIZE)+".pkl"), "wb") as fp:
#        pickle.dump(trajectories, fp)
#    with open(os.path.join(log_path, "hmdb_3dcnn_snamesF"+str(fold)+"_train_seq"+str(SEQ_SIZE)+".pkl"), "wb") as fp:
#        pickle.dump(vid_names, fp)
##    else:
##        with open(os.path.join(log_path, "hmdb_3dcnn_feats_train_seq"+str(SEQ_SIZE)+".pkl"), "rb") as fp:
##            trajectories = pickle.load(fp)
##        with open(os.path.join(log_path, "hmdb_3dcnn_snames_train_seq"+str(SEQ_SIZE)+".pkl"), "rb") as fp:
##            vid_names = pickle.load(fp)
#
#    trajectories, vid_names = extract_3DCNN_feats(DATASET, LABELS, BATCH_SIZE, SEQ_SIZE=16, 
#                                                  STEP=STEP, foldno=fold, train=False, 
#                                                  nstrokes=-1)
#    
##    if not os.path.isfile(os.path.join(log_path, "hmdb_3dcnn_feats_train_seq"+str(SEQ_SIZE)+".pkl")):
#    with open(os.path.join(log_path, "hmdb_3dcnn_featsF"+str(fold)+"_test_seq"+str(SEQ_SIZE)+".pkl"), "wb") as fp:
#        pickle.dump(trajectories, fp)
#    with open(os.path.join(log_path, "hmdb_3dcnn_snamesF"+str(fold)+"_test_seq"+str(SEQ_SIZE)+".pkl"), "wb") as fp:
#        pickle.dump(vid_names, fp)


    ###########################################################################
#    # 2DCNN Features
#    STEP = 8
#    SEQ_SIZE = 16
#    BATCH_SIZE = 64
#    fold = 1
#    log_path = "feats/2dcnn_hmdb51_step"+str(STEP)
#    if not os.path.isdir(log_path):
#        os.makedirs(log_path)
#    trajectories, vid_names = extract_2DCNN_feats(DATASET, LABELS, BATCH_SIZE, STEP,
#                                                  foldno=fold, train=True, 
#                                                  nstrokes=-1)
#    
#    with open(os.path.join(log_path, "hmdb_2dcnn_featsF"+str(fold)+"_train.pkl"), "wb") as fp:
#        pickle.dump(trajectories, fp)
#    with open(os.path.join(log_path, "hmdb_2dcnn_snamesF"+str(fold)+"_train.pkl"), "wb") as fp:
#        pickle.dump(vid_names, fp)
#    
#    trajectories, vid_names = extract_2DCNN_feats(DATASET, LABELS, BATCH_SIZE, STEP,
#                                                  foldno=fold, train=False, 
#                                                  nstrokes=-1)
#    
#    with open(os.path.join(log_path, "hmdb_2dcnn_featsF"+str(fold)+"_test.pkl"), "wb") as fp:
#        pickle.dump(trajectories, fp)
#    with open(os.path.join(log_path, "hmdb_2dcnn_snamesF"+str(fold)+"_test.pkl"), "wb") as fp:
#        pickle.dump(vid_names, fp)    
    
    
    ###########################################################################
    # I3D Features
    STEP = 8
    SEQ_SIZE = 16
    BATCH_SIZE = 2
    fold = 1
    log_path = "feats/I3DFine_hmdb51_SEQ16_STEP"+str(STEP)
    model_path = "/home/arpan/VisionWorkspace/Cricket/StrokeAttention/hmdb51/i3d_000030.pt"
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    trajectories, vid_names = extract_I3D_feats(DATASET, LABELS, BATCH_SIZE, SEQ_SIZE, 
                                                STEP, foldno=fold, train=True, 
                                                model_path=model_path, nstrokes=-1)
    
    with open(os.path.join(log_path, "hmdb_i3d_featsF"+str(fold)+"_train.pkl"), "wb") as fp:
        pickle.dump(trajectories, fp)
    with open(os.path.join(log_path, "hmdb_i3d_snamesF"+str(fold)+"_train.pkl"), "wb") as fp:
        pickle.dump(vid_names, fp)
    
    trajectories, vid_names = extract_I3D_feats(DATASET, LABELS, BATCH_SIZE, SEQ_SIZE, 
                                                STEP, foldno=fold, train=False, 
                                                model_path=model_path, nstrokes=-1)
    
    with open(os.path.join(log_path, "hmdb_i3d_featsF"+str(fold)+"_test.pkl"), "wb") as fp:
        pickle.dump(trajectories, fp)
    with open(os.path.join(log_path, "hmdb_i3d_snamesF"+str(fold)+"_test.pkl"), "wb") as fp:
        pickle.dump(vid_names, fp)
        