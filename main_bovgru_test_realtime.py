#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 5 10:39:42 2020

@author: arpan

@Description: Training GRU model on BoV sequence classification.
"""

import os
import sys
import numpy as np

sys.path.insert(0, '../cluster_strokes')
sys.path.insert(0, '../cluster_strokes/lib')

import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader

from utils import autoenc_utils
from utils import trajectory_utils as traj_utils
#import datasets.videotransforms as videotransforms
from datasets.dataset import StrokeFeatureSequenceDataset
#from datasets.dataset import StrokeFeaturePairsDataset
import time
import pickle
import attn_model
import attn_utils
from create_bovw import create_bovw_SA  #create_bovw_OMP   #
import json

sys.path.insert(0, '../CricketStrokeLocalizationBOVW')
from extract_hoof_feats import extract_flow_grid, extract_flow_angles

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cluster_size = 1000
INPUT_SIZE = cluster_size      # OFGRID: 576, 3DCNN: 512, 2DCNN: 2048
HIDDEN_SIZE = 256
N_LAYERS = 2
bidirectional = True

km_filename = "km_onehot"
log_path = "logs/bovgru_SA_of20_Hidden256"


def predict(features, stroke_names_id, model, dataloaders, labs_keys, labs_values, 
            seq, phase="val"):
    assert phase == "val" or phase=="test", "Incorrect Phase."
    model = model.eval()
    gt_list, pred_list, stroke_ids  = [], [], []
    # Iterate over data.
    for bno, (inputs, vid_path, stroke, labels) in enumerate(dataloaders[phase]):
        # inputs of shape BATCH x SEQ_LEN x FEATURE_DIM
        labels = attn_utils.get_batch_labels(vid_path, stroke, labs_keys, labs_values, 1)
        inputs = inputs.float()
#        inp_emb = attn_utils.get_long_tensor(inputs)    # comment out for SA
#        inputs = inp_emb.to(device)                     # comment out for SA
        inputs = inputs.to(device)
        labels = labels.to(device)
        # forward
        with torch.set_grad_enabled(phase == 'train'):
            batch_size = inputs.size(0)
            hidden = model.init_hidden(batch_size)
            outputs, hidden = model(inputs, hidden)
            gt_list.append(labels.tolist())
            pred_list.append((torch.max(outputs, 1)[1]).tolist())
            for i, vid in enumerate(vid_path):
                stroke_ids.extend([vid+"_"+str(stroke[0][i].item())+"_"+str(stroke[1][i].item())] * 1)
                
#    epoch_loss = running_loss #/ len(dataloaders[phase].dataset)
#            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
#    print('{} Loss: {:.4f}'.format(phase, epoch_loss))
    
    ###########################################################################
    
    confusion_mat = np.zeros((model.n_classes, model.n_classes))
    gt_list = [g for batch_list in gt_list for g in batch_list]
    pred_list = [p for batch_list in pred_list for p in batch_list]
    
    predictions = {"gt": gt_list, "pred": pred_list}
    
    # Save prediction and ground truth labels
#    with open(os.path.join(log_path, "preds_Seq"+str(seq)+"_C"+str(cluster_size)+".pkl"), "wb") as fp:
#        pickle.dump(predictions, fp)
#    with open(os.path.join(log_path, "preds_Seq"+str(seq)+"_C"+str(cluster_size)+".pkl"), "rb") as fp:
#        predictions = pickle.load(fp)
    gt_list = predictions['gt']
    pred_list = predictions['pred']
    
    
    prev_gt = stroke_ids[0]
    val_labels, pred_labels, vid_preds = [], [], []
    for i, pr in enumerate(pred_list):
        if prev_gt != stroke_ids[i]:
            # find max category predicted in pred_labels
            val_labels.append(gt_list[i-1])
            pred_labels.append(max(set(vid_preds), key = vid_preds.count))
            vid_preds = []
            prev_gt = stroke_ids[i]
        vid_preds.append(pr)
        
    val_labels.append(gt_list[-1])
    pred_labels.append(max(set(vid_preds), key = vid_preds.count))
    
    ###########################################################################
    
    correct = 0
    for i,true_val in enumerate(val_labels):
        if pred_labels[i] == true_val:
            correct+=1
        confusion_mat[pred_labels[i], true_val]+=1
    print('#'*30)
    print("GRU Sequence Classification Results:")
    print("%d/%d Correct" % (correct, len(pred_labels)))
    print("Accuracy = {} ".format( float(correct) / len(pred_labels)))
    print("Confusion matrix")
    print(confusion_mat)
    return (float(correct) / len(pred_labels))
    

def main(DATASET, LABELS, CLASS_IDS, BATCH_SIZE, ANNOTATION_FILE, SEQ_SIZE=16, 
         STEP=16, nstrokes=-1, N_EPOCHS=25):
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
    nstrokes : int
        partial extraction of features (do not execute for entire dataset)
    
    Returns:
    --------
    acc, time for extraction and prediction
    
    '''
    ###########################################################################
    s1 = time.time()

    grid_size = 20
    mag_thresh, bins, density = 2, 20, True
    attn_utils.seed_everything(1234)
    
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    
    # Read the strokes 
    # Divide the highlight dataset files into training, validation and test sets
    train_lst, val_lst, test_lst = autoenc_utils.split_dataset_files(DATASET)
    print("No. of training videos : {}".format(len(train_lst)))
    
    km_filepath = os.path.join(log_path, km_filename)
    if not os.path.isfile(km_filepath+"_C"+str(cluster_size)+".pkl"):
        print("KMeans file not found...")
        sys.exit()
    else:
        # Load from disk, for validation and test sets.
        km_model = pickle.load(open(km_filepath+"_C"+str(cluster_size)+".pkl", 'rb'))
        
    ###########################################################################
    
    nFrames = 0
    partition_lst = val_lst
    strokes_name_id = []
    all_feats = {}
    # extract feats and run on one video at a time
    for i, v_file in enumerate(partition_lst):
        print('-'*60)
        print(str(i+1)+". v_file :: ", v_file)
        if '.avi' in v_file or '.mp4' in v_file:
            v_file = v_file.rsplit('.', 1)[0]
        json_file = v_file + '.json'
        
        # read labels from JSON file
        assert os.path.exists(os.path.join(LABELS, json_file)), "{} doesn't exist!".format(json_file)
            
        with open(os.path.join(LABELS, json_file), 'r') as fr:
            frame_dict = json.load(fr)
        frame_indx = list(frame_dict.values())[0]
        for m,n in frame_indx:
            k = v_file+"_"+str(m)+"_"+str(n)
            print("Stroke {} - {}".format(m,n))
            strokes_name_id.append(k)
            # Extract the stroke features
            if grid_size is None:
                all_feats[k] = extract_flow_angles(os.path.join(DATASET, v_file+".avi"), \
                                                 m, n, bins, mag_thresh, density)
            else:
                all_feats[k] = extract_flow_grid(os.path.join(DATASET, v_file+".avi"), \
                                                 m, n, grid_size)
            nFrames += (all_feats[k].shape[0] + 1)
    
    print("Create numpy one hot representation for val features...")
    onehot_feats_val = create_bovw_SA(all_feats, strokes_name_id, km_model)
    
    ft_path_partition = os.path.join(log_path, "C"+str(cluster_size)+"_partition.pkl")
    with open(ft_path_partition, "wb") as fp:
        pickle.dump(onehot_feats_val, fp)
    ###########################################################################
    
    ###########################################################################
    s2 = time.time()
    # Create a Dataset
    partition_dataset = StrokeFeatureSequenceDataset(ft_path_partition, partition_lst, DATASET, LABELS, CLASS_IDS, 
                                         frames_per_clip=SEQ_SIZE, extracted_frames_per_clip=2,
                                         step_between_clips=STEP, train=False)
    
    # get labels
    labs_keys, labs_values = attn_utils.get_cluster_labels(ANNOTATION_FILE)
    
    partition_loader = DataLoader(dataset=partition_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    data_loaders = {"test": partition_loader}

    num_classes = len(list(set(labs_values)))
    
#    vis_clusters(features, onehot_feats, stroke_names_id, 2, DATASET, log_path)
    
    ###########################################################################    
    
    # load model and set loss function
    model = attn_model.GRUBoWSAClassifier(INPUT_SIZE, HIDDEN_SIZE, num_classes, 
                                     N_LAYERS, bidirectional)
    
    model = model.to(device)

    ###########################################################################
    # Training the model    
    
    model = attn_utils.load_weights(log_path, model, N_EPOCHS, 
                                    "S"+str(SEQ_SIZE)+"C"+str(cluster_size)+"_SGD")
    
#    ###########################################################################
    s3 = time.time()
    acc = predict(all_feats, strokes_name_id, model, data_loaders, labs_keys, 
                  labs_values, SEQ_SIZE, phase='test')
    
    # call count_paramters(model)  for displaying total no. of parameters
    print("#Parameters : {} ".format(autoenc_utils.count_parameters(model)))
    print("Total Frames : {}".format(nFrames))
    s4 = time.time()
    return acc, [s1, s2, s3, s4]


if __name__ == '__main__':
    # Local Paths
    LABELS = "/home/arpan/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
    DATASET = "/home/arpan/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
    CLASS_IDS = "/home/arpan/VisionWorkspace/Cricket/cluster_strokes/configs/Class Index_Strokes.txt"    
    ANNOTATION_FILE = "/home/arpan/VisionWorkspace/Cricket/CricketStrokeLocalizationBOVW/shots_classes.txt"

    seq_sizes = range(20, 21, 2)
    STEP = 1
    BATCH_SIZE = 32
    N_EPOCHS = 30
    
    attn_utils.seed_everything(1234)
    acc = []

    print("OF20 BOV GRU HA without Embedding...")
    print("EPOCHS = {} : HIDDEN_SIZE = {} : GRU LAYERS = {}".format(N_EPOCHS, 
          HIDDEN_SIZE, N_LAYERS))
    start = time.time()
    for SEQ_SIZE in seq_sizes:
        print("SEQ_SIZE : {} :: CLUSTER_SIZE : {}".format(SEQ_SIZE, cluster_size))
        acc, tm = main(DATASET, LABELS, CLASS_IDS, BATCH_SIZE, ANNOTATION_FILE,
                          SEQ_SIZE, STEP, nstrokes=-1, N_EPOCHS=N_EPOCHS)
        
    print("*"*60)
    end = time.time()
    print("SEQ_SIZES : {}".format(seq_sizes))
    print("Accuracy values : {}".format(acc))
    print("Feature extraction : {} :: Model Prediction : {}".format(tm[1]-tm[0], tm[3]-tm[2]))
    print("Total Time : {}".format(end-start))
    