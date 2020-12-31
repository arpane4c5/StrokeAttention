#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 10:39:42 2020

@author: arpan

@Description: Training SVM model on BoVW representations.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.svm import LinearSVC

sys.path.insert(0, '../../cluster_strokes')
sys.path.insert(0, '../../cluster_strokes/lib')
sys.path.insert(0, '../../CricketStrokeLocalizationBOVW')

import torch
import dataset_hmdb as hmdb

#from utils import autoenc_utils
import pickle
from collections import Counter
from create_bovw import make_codebook
from create_bovw import create_bovw_df
import warnings

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore")



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# "of_fold1_feats_grid20.pkl","of_fold1_test_feats_grid20.pkl"
# "2dcnn_feats_train.pkl" ; "3dcnn_feats_train.pkl" ; "hoof_feats_val_b20.pkl"
# "hmdb_i3d_featsF1_train.pkl", "hmdb_i3d_featsF1_test.pkl"
# "unhoof_fold1_feats_bins20.pkl","unhoof_fold1_test_feats_bins20.pkl"
feat, feat_test = "hmdb_i3d_featsF1_train.pkl", "hmdb_i3d_featsF1_test.pkl"
# "of_fold1_snames_grid20.pkl", "of_fold1_test_snames_grid20.pkl"
# "hoof_snames_b20.pkl"
# "unhoof_fold1_snames_bins20.pkl", "unhoof_fold1_test_snames_bins20.pkl"
snames, snames_test = "hmdb_i3d_snamesF1_train.pkl", "hmdb_i3d_snamesF1_test.pkl" 
cluster_size = 100
INPUT_SIZE = cluster_size      # OFGRID: 576, 3DCNN: 512, 2DCNN: 2048
bidirectional = True

km_filename = "km_onehot"
log_path = "logs/bovgru_HA_i3dFine_hmdb51_tmp"
# feats/ofMagAng_grid20" ; "feats/unhoof_bins20_tmp"
feat_path = "feats/I3DFine_hmdb51_SEQ16_STEP8"


#def extract_of_features(feat_path, video_list, fold=1, isTrain=True):
#    
#    nbins, mth, grid = 20, 0.5, None   # grid should be None for extracting HOOF
#    if isTrain:
#        ft_prefix='unhoof_fold'+str(fold)
#    else:
#        ft_prefix='unhoof_fold'+str(fold)+'_test'
#    if not os.path.isfile(os.path.join(feat_path, ft_prefix+"_feats_bins"+str(nbins)+".pkl")):
#        if not os.path.exists(feat_path):
#            os.makedirs(feat_path)
#        if isTrain:
#            print("Fold {} : Training extraction ... ".format(fold))
#        else:
#            print("Fold {} : Test extraction ... ".format(fold))
#            
#        #    # Extract Grid OF / HOOF features {mth = 2, and vary nbins}
#        features, strokes_name_id = extract_feats(video_list, nbins, mth, False, grid)
#        with open(os.path.join(feat_path, ft_prefix+"_feats_bins"+str(nbins)+".pkl"), "wb") as fp:
#            pickle.dump(features, fp)
#        with open(os.path.join(feat_path, ft_prefix+"_snames_bins"+str(nbins)+".pkl"), "wb") as fp:
#            pickle.dump(strokes_name_id, fp)
#    else:
#        print("Already extracted : {}".format(os.path.join(feat_path, ft_prefix+"_feats_bins"+str(nbins)+".pkl")))
            
def read_feats(base_name, feat, snames):
    with open(os.path.join(base_name, feat), "rb") as fp:
        features = pickle.load(fp)
    with open(os.path.join(base_name, snames), "rb") as fp:
        strokes_name_id = pickle.load(fp)
    return features, strokes_name_id


#def predict(model, dataloaders, seq, phase="val"):
#    assert phase == "val" or phase=="test", "Incorrect Phase."
#    model = model.eval()
#    gt_list, pred_list, stroke_ids  = [], [], []
#    count = [0.] * cluster_size
#    # Iterate over data.
#    for bno, (inputs, vid_path, start_pts, labels) in enumerate(dataloaders[phase]):
#        # inputs of shape BATCH x SEQ_LEN x FEATURE_DIM
##        labels = attn_utils.get_batch_labels(vid_path, stroke, labs_keys, labs_values, 1)
#        inputs = inputs.float()
#        inp_emb = attn_utils.get_long_tensor(inputs)    # comment out for SA
#        inputs = inp_emb.to(device)                     # comment out for SA
#        inputs = inputs.to(device)
#        labels = labels.to(device)
#        iter_counts = Counter(inp_emb.flatten().tolist())
#        for k,v in iter_counts.items():
#            count[k]+=v
#        # forward
#        with torch.set_grad_enabled(phase == 'train'):
#            batch_size = inputs.size(0)
#            hidden = model.init_hidden(batch_size)
#            outputs, hidden = model(inputs, hidden)
#            gt_list.append(labels.tolist())
#            pred_list.append((torch.max(outputs, 1)[1]).tolist())
#            for i, vid in enumerate(vid_path):
#                stroke_ids.extend([vid] * 1)
#                
##    epoch_loss = running_loss #/ len(dataloaders[phase].dataset)
##            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
##    print('{} Loss: {:.4f}'.format(phase, epoch_loss))
#    
#    ###########################################################################
#    print("Clusters : ")
#    print(count)
#    confusion_mat = np.zeros((model.n_classes, model.n_classes))
#    gt_list = [g for batch_list in gt_list for g in batch_list]
#    pred_list = [p for batch_list in pred_list for p in batch_list]
#    
#    predictions = {"gt": gt_list, "pred": pred_list}
#    
#    # Save prediction and ground truth labels
#    with open(os.path.join(log_path, "preds_Seq"+str(seq)+"_C"+str(cluster_size)+".pkl"), "wb") as fp:
#        pickle.dump(predictions, fp)
#    with open(os.path.join(log_path, "preds_Seq"+str(seq)+"_C"+str(cluster_size)+".pkl"), "rb") as fp:
#        predictions = pickle.load(fp)
#    gt_list = predictions['gt']
#    pred_list = predictions['pred']
#    
#    
#    prev_gt = stroke_ids[0]
#    val_labels, pred_labels, vid_preds = [], [], []
#    for i, pr in enumerate(pred_list):
#        if prev_gt != stroke_ids[i]:
#            # find max category predicted in pred_labels
#            val_labels.append(gt_list[i-1])
#            pred_labels.append(max(set(vid_preds), key = vid_preds.count))
#            vid_preds = []
#            prev_gt = stroke_ids[i]
#        vid_preds.append(pr)
#        
#    val_labels.append(gt_list[-1])
#    pred_labels.append(max(set(vid_preds), key = vid_preds.count))
#    
#    ###########################################################################
#    
#    correct = 0
#    for i,true_val in enumerate(val_labels):
#        if pred_labels[i] == true_val:
#            correct+=1
#        confusion_mat[pred_labels[i], true_val]+=1
#    print('#'*30)
#    print("GRU Sequence Classification Results:")
#    print("%d/%d Correct" % (correct, len(pred_labels)))
#    print("Accuracy = {} ".format( float(correct) / len(pred_labels)))
#    print("Confusion matrix")
#    print(confusion_mat)
#    return (float(correct) / len(pred_labels))
    

def main(DATASET, LABELS, SEQ_SIZE=16, STEP=16, nstrokes=-1):
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
    
#    attn_utils.seed_everything(1234)
    
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    
    hmdb51_train = hmdb.HMDB51(DATASET, LABELS, SEQ_SIZE, step_between_clips = STEP, 
                               fold=1, train=True, transform=None)

    hmdb51_test = hmdb.HMDB51(DATASET, LABELS, SEQ_SIZE, step_between_clips = STEP, 
                              fold=1, train=False, transform=None)

#    # Extracting training features
#    extract_of_features(feat_path, hmdb51_train.video_list, hmdb51_train.fold, hmdb51_train.train)
#    # Extracting testing features
#    extract_of_features(feat_path, hmdb51_test.video_list, hmdb51_test.fold, hmdb51_test.train)
    
    features, stroke_names_id = read_feats(feat_path, feat, snames)
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
        
    df_train, words_train = create_bovw_df(features, stroke_names_id, km_model, 
                                           log_path, 'train')
    
    # read the stroke annotation labels from text file.
    vids_list = list(df_train.index)
    
    labs_keys = hmdb51_train.video_list
    labs_indx = hmdb51_train.indices
    labs_values = [hmdb51_train.samples[i][1] for i in labs_indx]
    train_labels = np.array([labs_values[labs_keys.index(v)] for v in vids_list])
    
    num_classes = len(list(set(labs_values)))
    
    print("Training dataframe formed.")
    
    ###########################################################################
    # Train a classifier on the features.
    ###########################################################################
    # Train SVM
    clf = LinearSVC(verbose=False, random_state=124, max_iter=3000)
    clf.fit(df_train, train_labels)
    
    print("Training Complete.")
    ###########################################################################
    
    #print("Training complete. Saving to disk.")
    # Save model to disk
    joblib.dump(clf, os.path.join(log_path, "clf.pkl"))
    # Load trained model from disk
    clf = joblib.load(os.path.join(log_path, "clf.pkl"))

    # Train a classifier on both the features.
    #print("Training with SVM")
    #df_train = pd.concat([df_train_mag, df_train_ang], axis=1)
    #clf_both = SVC(kernel="linear",verbose=True)
    #clf_both = LinearSVC(verbose=True, random_state=123, max_iter=2000)
    #clf_both.fit(df_train, labels)
    #print("Training with SVM (ang)")
    #clf_ang = SVC(kernel="linear",verbose=True)
    #clf_ang.fit(df_train_ang, labels)
    

    ##########################################################################
    features_test, stroke_names_id_test = read_feats(feat_path, feat_test, 
                                                              snames_test)
    
    print("Create dataframe BOVW validation set...")
    df_test_hoof, words_test = create_bovw_df(features_test, stroke_names_id_test, \
                                            km_model, log_path, "test")
    
    vids_list_test = list(df_test_hoof.index)
        
    labs_keys = hmdb51_test.video_list
    labs_indx = hmdb51_test.indices
    labs_values = [hmdb51_test.samples[i][1] for i in labs_indx]

    test_labels = np.array([labs_values[labs_keys.index(v)] for v in vids_list_test])
    
    ###########################################################################
    # Evaluate the BOW classifier (SVM)
    confusion_mat = np.zeros((num_classes, num_classes))
    pred = clf.predict(df_test_hoof)
    correct = 0
    for i,true_val in enumerate(test_labels):
        if pred[i] == true_val:
            correct+=1
        confusion_mat[pred[i], true_val]+=1
    print('#'*30)
    print("BOW Classification Results:")
    print("%d/%d Correct" % (correct, len(pred)))
    print("Accuracy = {} ".format( float(correct) / len(pred)))
    print("Confusion matrix")
    print(confusion_mat)
    return (float(correct) / len(pred))



if __name__ == '__main__':
    # Local Paths
    LABELS = "/home/arpan/VisionWorkspace/VideoData/hmdb51/train_test_splits"
    DATASET = "/home/arpan/VisionWorkspace/VideoData/hmdb51/videos"

    seq_sizes = range(30, 31, 2)
    
    acc = []

    print("OF20 BOV SVM ...")
#    print("EPOCHS = {} : HIDDEN_SIZE = {} : GRU LAYERS = {}".format(N_EPOCHS, 
#          HIDDEN_SIZE, N_LAYERS))
    for SEQ_SIZE in seq_sizes:
        print("SEQ_SIZE : {} :: CLUSTER_SIZE : {}".format(SEQ_SIZE, cluster_size))
        acc.append(main(DATASET, LABELS, nstrokes=-1))
        
    print("*"*60)
    print("SEQ_SIZES : {}".format(seq_sizes))
    print("Accuracy values : {}".format(acc))
