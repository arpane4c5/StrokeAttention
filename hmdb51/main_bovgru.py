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

sys.path.insert(0, '../../cluster_strokes')
sys.path.insert(0, '../../cluster_strokes/lib')
sys.path.insert(0, '../')

import torch
from torch import nn, optim
import dataset_hmdb as hmdb
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from utils import autoenc_utils

import copy
import time
import pickle
import attn_model
import attn_utils
from collections import Counter
from create_bovw import make_codebook
from create_bovw import create_bovw_onehot
from create_bovw import vis_cluster
#from sklearn.externals import joblib


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# "of_fold1_feats_grid20.pkl","of_fold1_test_feats_grid20.pkl"
# "unhoof_fold1_feats_bins20.pkl","unhoof_fold1_test_feats_bins20.pkl"
# "hmdb_3dcnn_featsF1_train_seq16.pkl","hmdb_3dcnn_featsF1_test_seq16.pkl"
# "hmdb_2dcnn_featsF1_train.pkl", "hmdb_2dcnn_featsF1_test.pkl"
# "hmdb_i3d_featsF1_train.pkl", "hmdb_i3d_featsF1_test.pkl"
feat, feat_test = "hmdb_i3d_featsF1_train.pkl", "hmdb_i3d_featsF1_test.pkl"
# "of_fold1_snames_grid20.pkl", "of_fold1_test_snames_grid20.pkl"
# "unhoof_fold1_snames_bins20.pkl", "unhoof_fold1_test_snames_bins20.pkl"
# "hmdb_3dcnn_snamesF1_train_seq16.pkl", "hmdb_3dcnn_snamesF1_test_seq16.pkl"
# "hmdb_2dcnn_snamesF1_train.pkl", "hmdb_2dcnn_snamesF1_test.pkl"
# "hmdb_i3d_snamesF1_train.pkl", "hmdb_i3d_snamesF1_test.pkl"
snames, snames_test = "hmdb_i3d_snamesF1_train.pkl", "hmdb_i3d_snamesF1_test.pkl"
cluster_size = 100
INPUT_SIZE = cluster_size      # OFGRID: 576, 3DCNN: 512, 2DCNN: 2048
#INPUT_SIZE = 1024
HIDDEN_SIZE = 256
N_LAYERS = 2
bidirectional = True

km_filename = "km_onehot"
#bovgru_HA_unhoof_bins20_Hidden256_tmp ; bovgru_2dcnn_hmdb51 ; bovgru_HA_i3d_hmdb51
log_path = "logs/bovgru_HA_i3dFine_hmdb51"
# ofMagAng_grid20" ; 2dcnn_hmdb51 ; 2dcnn_hmdb51_step8; I3D_hmdb51_SEQ16_STEP8 ; 
feat_path = "feats/I3DFine_hmdb51_SEQ16_STEP8" 
            

def train_model(model, dataloaders, criterion, optimizer, scheduler, 
                num_epochs=25):
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
            
            count = [0.] * 51

            # Iterate over data.
            for bno, (inputs, vid_path, start_pts, labels) in enumerate(dataloaders[phase]):
                # inputs of shape BATCH x SEQ_LEN x FEATURE_DIM
                # Extract spatio-temporal features from clip using 3D ResNet (For SL >= 16)
                inputs = inputs.float()
                inp_emb = attn_utils.get_long_tensor(inputs)    # comment out for SA
                inputs = inp_emb.to(device)                     # comment out for SA
                inputs = inputs.to(device)
                labels = labels.to(device)
                iter_counts = Counter(labels.tolist())
                for k,v in iter_counts.items():
                    count[k]+=v
                
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    hidden = model.init_hidden(inputs.size(0))
                    
                    outputs, hidden = model(inputs, hidden)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)     #torch.flip(targets, [1])

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() #* inputs.size(0)
#                print("Iter : {}/{} :: Running Loss : {}".format(bno, \
#                      len(dataloaders[phase]), running_loss))
                running_corrects += torch.sum(preds == labels.data)
#                if bno == 5:
#                    print("")
#                if bno == 200:
#                    break
                                    
            if phase == 'train':
                scheduler.step()
                print("Category Weights : {}".format(count))

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
#            # deep copy the model for best test accuracy
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, \
          time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

#    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def predict(model, dataloaders, seq, phase="val"):
    assert phase == "val" or phase=="test", "Incorrect Phase."
    model = model.eval()
    gt_list, pred_list, stroke_ids  = [], [], []
    count = [0.] * cluster_size
    # Iterate over data.
    for bno, (inputs, vid_path, start_pts, labels) in enumerate(dataloaders[phase]):
        # inputs of shape BATCH x SEQ_LEN x FEATURE_DIM
        inputs = inputs.float()
        inp_emb = attn_utils.get_long_tensor(inputs)    # comment out for SA
        inputs = inp_emb.to(device)                     # comment out for SA
        inputs = inputs.to(device)
        labels = labels.to(device)
        iter_counts = Counter(inp_emb.flatten().tolist())
        for k,v in iter_counts.items():
            count[k]+=v
        # forward
        with torch.set_grad_enabled(phase == 'train'):
            batch_size = inputs.size(0)
            hidden = model.init_hidden(batch_size)
            outputs, hidden = model(inputs, hidden)
            gt_list.append(labels.tolist())
            pred_list.append((torch.max(outputs, 1)[1]).tolist())
            for i, vid in enumerate(vid_path):
                stroke_ids.extend([vid] * 1)
    
    ###########################################################################
    print("Clusters : ")
    print(count)
    confusion_mat = np.zeros((model.n_classes, model.n_classes))
    gt_list = [g for batch_list in gt_list for g in batch_list]
    pred_list = [p for batch_list in pred_list for p in batch_list]
    
    predictions = {"gt": gt_list, "pred": pred_list}
    
    # Save prediction and ground truth labels
    with open(os.path.join(log_path, "preds_Seq"+str(seq)+"_C"+str(cluster_size)+".pkl"), "wb") as fp:
        pickle.dump(predictions, fp)
    with open(os.path.join(log_path, "preds_Seq"+str(seq)+"_C"+str(cluster_size)+".pkl"), "rb") as fp:
        predictions = pickle.load(fp)
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
    
def get_hmdb_sample_weights(train_dataset):
    # get list of labs_keys and labs_values from samples
    labs_keys = [train_dataset.samples[i][0] for i in train_dataset.indices]
    labs_values = [train_dataset.samples[i][1] for i in train_dataset.indices]
    n_classes = len(list(set(labs_values)))        
    sample_counts = [0.] * n_classes
            
    train_set_keys = []
    # count the number of samples for each class
    for i in range(train_dataset.__len__()):
#        print(i, len(train_dataset))
#        if isinstance(train_dataset, hmdb.HMDB51):
#            _, vpath, start_pts, *_ = train_dataset.video_clips.get_clip(i)
#        else:
#        seq, vpath, start_pts, *_ = train_dataset.__getitem__(i)
        video_idx, _ = train_dataset.video_clips.get_clip_location(i)
        vpath = train_dataset.video_clips.video_paths[video_idx]
        key = vpath
        label = labs_values[labs_keys.index(key)]
        train_set_keys.append(key)
        sample_counts[label] +=1
    # calculate the weights for each class
    weights = 1./np.array(sample_counts)
    sample_weights = [0.] * train_dataset.__len__()
    # assign the weights for each sample for creation of a sampler
    for i, key in enumerate(train_set_keys):
        label = labs_values[labs_keys.index(key)]
        sample_weights[i] = weights[label]
    
    sample_weights = torch.from_numpy(np.array(sample_weights))
    sample_weights = sample_weights.double()
    return sample_weights

def display_sizes(vid_list):
    
    import cv2
    for v in vid_list:
        cap = cv2.VideoCapture(v)
        if cap.isOpened():
            h, w = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
            if w < 240:
                print(h, w)
        cap.release()

def main(DATASET, LABELS, BATCH_SIZE, SEQ_SIZE=16, STEP=16, nstrokes=-1, 
         N_EPOCHS=25):
    '''
    Extract sequence features from AutoEncoder.
    
    Parameters:
    -----------
    DATASET : str
        path to the video dataset
    LABELS : str
        path containing stroke labels
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
    
    features, stroke_names_id = attn_utils.read_feats(feat_path, feat, snames)
    
    # get matrix of features from dictionary (N, vec_size)
    vecs = []
    for key in sorted(list(features.keys())):
        vecs.append(features[key])
    vecs = np.vstack(vecs)
    
    vecs[np.isnan(vecs)] = 0
    vecs[np.isinf(vecs)] = 0
    
    #fc7 layer output size (4096) 
    INP_VEC_SIZE = vecs.shape[-1]
    print("INP_VEC_SIZE = ", INP_VEC_SIZE)
    
    km_filepath = os.path.join(log_path, km_filename)
#    # Uncomment only while training.
    if not os.path.isfile(km_filepath+"_C"+str(cluster_size)+".pkl"):
        km_model = make_codebook(vecs, cluster_size)  #, model_type='gmm') 
        ##    # Save to disk, if training is performed
        print("Writing the KMeans models to disk...")
        pickle.dump(km_model, open(km_filepath+"_C"+str(cluster_size)+".pkl", "wb"))
    else:
        # Load from disk, for validation and test sets.
        km_model = pickle.load(open(km_filepath+"_C"+str(cluster_size)+".pkl", 'rb'))
        
    print("Create numpy one hot representation for train features...")
    onehot_feats = create_bovw_onehot(features, stroke_names_id, km_model)
    
    ft_path = os.path.join(log_path, "C"+str(cluster_size)+"_train.pkl")
#    ft_path = os.path.join(feat_path, feat)
    with open(ft_path, "wb") as fp:
        pickle.dump(onehot_feats, fp)
    with open(os.path.join(log_path, "C"+str(cluster_size)+"_snames_train.pkl"), "wb") as fp:
        pickle.dump(stroke_names_id, fp)    
    #########################################################################
    #########################################################################
    features_test, stroke_names_id_test = attn_utils.read_feats(feat_path, feat_test, 
                                                              snames_test)
    print("Create numpy one hot representation for val features...")
    onehot_feats_test = create_bovw_onehot(features_test, stroke_names_id_test, km_model)
    
    ft_path_test = os.path.join(log_path, "C"+str(cluster_size)+"_test.pkl")
#    ft_path_test = os.path.join(feat_path, feat_test)
    with open(ft_path_test, "wb") as fp:
        pickle.dump(onehot_feats_test, fp)    
    with open(os.path.join(log_path, "C"+str(cluster_size)+"_snames_test.pkl"), "wb") as fp:
        pickle.dump(stroke_names_id_test, fp)
    
    ###########################################################################
    # Create a Dataset    

    train_dataset = hmdb.HMDB51FeatureSequenceDataset(ft_path, DATASET, LABELS, 
                                         frames_per_clip=SEQ_SIZE, extracted_frames_per_clip=16,
                                         step_between_clips=STEP, train=True)

    test_dataset = hmdb.HMDB51FeatureSequenceDataset(ft_path_test, DATASET, LABELS,  
                                         frames_per_clip=SEQ_SIZE, extracted_frames_per_clip=16,
                                         step_between_clips=STEP, train=False)
    
#    display_sizes(train_dataset.video_list)
#    display_sizes(test_dataset.video_list)
#    # created weighted Sampler for class imbalance
    samples_weight = get_hmdb_sample_weights(train_dataset)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, #shuffle=True)
                              sampler=sampler, worker_init_fn=np.random.seed(12))
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    data_loaders = {"train": train_loader, "test": test_loader}

#    num_classes = len(list(set(labs_values)))
    num_classes = 51
        
    ###########################################################################    
    
    # load model and set loss function
    model = attn_model.GRUBoWHAClassifier(INPUT_SIZE, HIDDEN_SIZE, num_classes, 
                                     N_LAYERS, bidirectional)
    
#    model = load_weights(base_name, model, N_EPOCHS, "Adam")
    
#    for ft in model.parameters():
#        ft.requires_grad = False
        
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()    
    model = model.to(device)
#    print("Params to learn:")
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
#            print("\t",name)
    
    # Observe that all parameters are being optimized
    optimizer_ft = torch.optim.Adam(model.parameters(), lr=0.001)
#    optimizer_ft = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = StepLR(optimizer_ft, step_size=10, gamma=0.1)
    
    ###########################################################################
    # Training the model    
    
    start = time.time()
    
    model = train_model(model, data_loaders, criterion, optimizer_ft, 
                        exp_lr_scheduler, num_epochs=N_EPOCHS)
    
    end = time.time()
    
    # save the best performing model
    attn_utils.save_model_checkpoint(log_path, model, N_EPOCHS, 
                                     "S"+str(SEQ_SIZE)+"C"+str(cluster_size)+"_SGD")
    # Load model checkpoints
    model = attn_utils.load_weights(log_path, model, N_EPOCHS, 
                                    "S"+str(SEQ_SIZE)+"C"+str(cluster_size)+"_SGD")
    
    print("Total Execution time for {} epoch : {}".format(N_EPOCHS, (end-start)))

    ###########################################################################
    
    acc = predict(model, data_loaders, SEQ_SIZE, phase='test')
    
    # call count_paramters(model)  for displaying total no. of parameters
    print("#Parameters : {} ".format(autoenc_utils.count_parameters(model)))
    return acc


if __name__ == '__main__':
    # Local Paths
    LABELS = "/home/arpan/VisionWorkspace/VideoData/hmdb51/train_test_splits"
    DATASET = "/home/arpan/VisionWorkspace/VideoData/hmdb51/videos"

    seq_sizes = range(30, 31, 2)
    STEP = 8
    BATCH_SIZE = 32
    N_EPOCHS = 30
    
    attn_utils.seed_everything(1234)
    acc = []

    print("3DCNN BOV GRU HA without Embedding...")
    print("EPOCHS = {} : HIDDEN_SIZE = {} : GRU LAYERS = {}".format(N_EPOCHS, 
          HIDDEN_SIZE, N_LAYERS))
    for SEQ_SIZE in seq_sizes:
        print("SEQ_SIZE : {} :: CLUSTER_SIZE : {}".format(SEQ_SIZE, cluster_size))
        acc.append(main(DATASET, LABELS, BATCH_SIZE, SEQ_SIZE, STEP, nstrokes=-1, 
                        N_EPOCHS=N_EPOCHS))
        
    print("*"*60)
    print("SEQ_SIZES : {}".format(seq_sizes))
    print("Accuracy values : {}".format(acc))
