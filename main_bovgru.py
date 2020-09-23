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
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from utils import autoenc_utils
#import datasets.videotransforms as videotransforms
from datasets.dataset import StrokeFeatureSequenceDataset
#from datasets.dataset import StrokeFeaturePairsDataset
import copy
import time
import pickle
import attn_model
import attn_utils
from collections import Counter
from create_bovw import make_codebook
from create_bovw import create_bovw_SA
from create_bovw import vis_cluster
from sklearn.externals import joblib
import warnings

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore")

sys.path.insert(0, '../CricketStrokeLocalizationBOVW')
from extract_hoof_feats import extract_stroke_feats



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# "of_feats_grid20.pkl", "of_feats_val_grid20.pkl" ; "hoof_feats_b20.pkl"
# "2dcnn_feats_train.pkl" ; "3dcnn_feats_train.pkl" ; "hoof_feats_val_b20.pkl"
feat, feat_val = "of_feats_grid20.pkl", "of_feats_test_grid20.pkl"
# "of_snames_grid20.pkl" ; "2dcnn_snames_train.pkl" ; "3dcnn_snames_train.pkl";
# "hoof_snames_b20.pkl"
snames, snames_val = "of_snames_grid20.pkl", "of_snames_test_grid20.pkl"
cluster_size = 1000
INPUT_SIZE = cluster_size      # OFGRID: 576, 3DCNN: 512, 2DCNN: 2048
HIDDEN_SIZE = 256
N_LAYERS = 2
bidirectional = True

km_filename = "km_onehot"
log_path = "logs/bovgru_SA_of20_Hidden256"
# bow_HL_ofAng_grid20 ; bow_HL_2dres ; bow_HL_3dres_seq16; bow_HL_hoof_b20_mth2
feat_path = "/home/arpan/VisionWorkspace/Cricket/CricketStrokeLocalizationBOVW/logs/bow_HL_ofAng_grid20"

#HIST_DIFFS = "/home/arpan/VisionWorkspace/Cricket/localization_gru/hist_diff_highlights_gray"
#SBD_MODEL = "/home/arpan/VisionWorkspace/Cricket/scripts/supporting_files/sbd_model_RF_histDiffs_gray.pkl"

def read_boundaries(vkeys, hist_diffs_path, sbd_model):
    # load the model
    model = joblib.load(sbd_model)
    boundaries = {}
    for k in vkeys:
        vidfile = k.rsplit('/', 1)[1]
        feature_file = os.path.join(hist_diffs_path, vidfile.rsplit('.', 1)[0]+'.npy')
        vfeat = np.load(feature_file)       # load feature
        vpreds = model.predict(vfeat)       # make prediction
        idx_preds = np.argwhere(vpreds)     # get indices where 1
        # convert to list of indices where +ve predictions
        vid_boundaries = idx_preds.reshape(idx_preds.shape[0]).tolist()
        boundaries[vidfile] = vid_boundaries
    return boundaries

def extract_of_features(feat_path, dataset, labspath, train_lst, val_lst):
    
    nbins, mth, grid = 20, 2, 20   # grid should be None for extracting HOOF
    if not os.path.isfile(os.path.join(feat_path, "of_feats_grid"+str(grid)+".pkl")):
        if not os.path.exists(feat_path):
            os.makedirs(feat_path)
        #    # Extract Grid OF / HOOF features {mth = 2, and vary nbins}
        print("Training extraction ... ")
        features, strokes_name_id = extract_stroke_feats(dataset, labspath, train_lst, \
                                                     nbins, mth, True, grid) 
        with open(os.path.join(feat_path, "of_feats_grid"+str(grid)+".pkl"), "wb") as fp:
            pickle.dump(features, fp)
        with open(os.path.join(feat_path, "of_snames_grid"+str(grid)+".pkl"), "wb") as fp:
            pickle.dump(strokes_name_id, fp)
                
    if not os.path.isfile(os.path.join(feat_path, "of_feats_val_grid"+str(grid)+".pkl")):
        print("Validation extraction ....")
        features_val, strokes_name_id_val = extract_stroke_feats(dataset, labspath, val_lst, \
                                                         nbins, mth, True, grid)

        with open(os.path.join(feat_path, "of_feats_val_grid"+str(grid)+".pkl"), "wb") as fp:
            pickle.dump(features_val, fp)
        with open(os.path.join(feat_path, "of_snames_val_grid"+str(grid)+".pkl"), "wb") as fp:
            pickle.dump(strokes_name_id_val, fp)


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
            
            count = [0.] * 5

            # Iterate over data.
            for bno, (inputs, vid_path, stroke, labels) in enumerate(dataloaders[phase]):
                # inputs of shape BATCH x SEQ_LEN x FEATURE_DIM
                labels = attn_utils.get_batch_labels(vid_path, stroke, labs_keys, labs_values, 1)
                # Extract spatio-temporal features from clip using 3D ResNet (For SL >= 16)
                inputs = inputs.float()
#                inp_emb = attn_utils.get_long_tensor(inputs)    # comment out for SA
#                inputs = inp_emb.to(device)                     # comment out for SA
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
                running_loss += loss.item() * inputs.size(0)
#                print("Iter : {} :: Running Loss : {}".format(bno, running_loss))
                running_corrects += torch.sum(preds == labels.data)
                                    
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
    with open(os.path.join(log_path, "preds_Seq"+str(seq)+"_C"+str(cluster_size)+".pkl"), "wb") as fp:
        pickle.dump(predictions, fp)
    with open(os.path.join(log_path, "preds_Seq"+str(seq)+"_C"+str(cluster_size)+".pkl"), "rb") as fp:
        predictions = pickle.load(fp)
    gt_list = predictions['gt']
    pred_list = predictions['pred']
    
#    # get boundaries (worse accuracy when used)
#    vkeys = list(set([v.rsplit('_', 2)[0] for v in stroke_ids]))
#    boundaries = read_boundaries(vkeys, HIST_DIFFS, SBD_MODEL)
    #
    
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
    
#    extract_of_features(feat_path, DATASET, LABELS, train_lst, val_lst)
    
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
    onehot_feats = create_bovw_SA(features, stroke_names_id, km_model)
    
    ft_path = os.path.join(log_path, "C"+str(cluster_size)+"_train.pkl")
    with open(ft_path, "wb") as fp:
        pickle.dump(onehot_feats, fp)
    
    ###########################################################################
    
    features_val, stroke_names_id_val = attn_utils.read_feats(feat_path, feat_val, 
                                                              snames_val)
    
    print("Create numpy one hot representation for val features...")
    onehot_feats_val = create_bovw_SA(features_val, stroke_names_id_val, km_model)
    
    ft_path_val = os.path.join(log_path, "C"+str(cluster_size)+"_val.pkl")
    with open(ft_path_val, "wb") as fp:
        pickle.dump(onehot_feats_val, fp)    
    
    ###########################################################################
    # Create a Dataset    

#    ft_path = os.path.join(base_name, ft_dir, feat)
    train_dataset = StrokeFeatureSequenceDataset(ft_path, train_lst, DATASET, LABELS, CLASS_IDS, 
                                         frames_per_clip=SEQ_SIZE, extracted_frames_per_clip=2,
                                         step_between_clips=STEP, train=True)
#    ft_path_val = os.path.join(base_name, ft_dir, feat_val)
    val_dataset = StrokeFeatureSequenceDataset(ft_path_val, test_lst, DATASET, LABELS, CLASS_IDS, 
                                         frames_per_clip=SEQ_SIZE, extracted_frames_per_clip=2,
                                         step_between_clips=STEP, train=False)
    
    # get labels
    labs_keys, labs_values = attn_utils.get_cluster_labels(ANNOTATION_FILE)
    # created weighted Sampler for class imbalance
    samples_weight = attn_utils.get_sample_weights(train_dataset, labs_keys, labs_values, 
                                                   train_lst)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, 
                              sampler=sampler, worker_init_fn=np.random.seed(12))
    
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    data_loaders = {"train": train_loader, "test": val_loader}

    num_classes = len(list(set(labs_values)))
    
#    vis_cluster(onehot_feats, stroke_names_id, km_model, 3, 2, DATASET, "logs")
    
    ###########################################################################    
    
    # load model and set loss function
    model = attn_model.GRUBoWSAClassifier(INPUT_SIZE, HIDDEN_SIZE, num_classes, 
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
#    optimizer_ft = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer_ft = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = StepLR(optimizer_ft, step_size=10, gamma=0.1)
    
    ###########################################################################
    # Training the model    
    
    start = time.time()
    
#    model = train_model(features, stroke_names_id, model, data_loaders, criterion, 
#                        optimizer_ft, exp_lr_scheduler, labs_keys, labs_values,
#                        num_epochs=N_EPOCHS)
    
    end = time.time()
    
#    # save the best performing model
#    attn_utils.save_model_checkpoint(log_path, model, N_EPOCHS, 
#                                     "S"+str(SEQ_SIZE)+"C"+str(cluster_size)+"_SGD")
    # Load model checkpoints
    model = attn_utils.load_weights(log_path, model, N_EPOCHS, 
                                    "S"+str(SEQ_SIZE)+"C"+str(cluster_size)+"_SGD")
    
    print("Total Execution time for {} epoch : {}".format(N_EPOCHS, (end-start)))

#    ###########################################################################
    
    acc = predict(features_val, stroke_names_id_val, model, data_loaders, labs_keys, 
                  labs_values, SEQ_SIZE, phase='test')
    
    # call count_paramters(model)  for displaying total no. of parameters
    print("#Parameters : {} ".format(autoenc_utils.count_parameters(model)))
    return acc


if __name__ == '__main__':
    # Local Paths
    LABELS = "/home/arpan/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
    DATASET = "/home/arpan/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
    CLASS_IDS = "/home/arpan/VisionWorkspace/Cricket/cluster_strokes/configs/Class Index_Strokes.txt"    
    ANNOTATION_FILE = "/home/arpan/VisionWorkspace/Cricket/CricketStrokeLocalizationBOVW/shots_classes.txt"

    seq_sizes = range(30, 31, 2)
    STEP = 1
    BATCH_SIZE = 32
    N_EPOCHS = 30
    
    attn_utils.seed_everything(1234)
    acc = []

    print("OF20 BOV GRU HA without Embedding...")
    print("EPOCHS = {} : HIDDEN_SIZE = {} : GRU LAYERS = {}".format(N_EPOCHS, 
          HIDDEN_SIZE, N_LAYERS))
    for SEQ_SIZE in seq_sizes:
        print("SEQ_SIZE : {} :: CLUSTER_SIZE : {}".format(SEQ_SIZE, cluster_size))
        acc.append(main(DATASET, LABELS, CLASS_IDS, BATCH_SIZE, ANNOTATION_FILE,
                        SEQ_SIZE, STEP, nstrokes=-1, N_EPOCHS=N_EPOCHS))
        
    print("*"*60)
    print("SEQ_SIZES : {}".format(seq_sizes))
    print("Accuracy values : {}".format(acc))
