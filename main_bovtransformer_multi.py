#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 10:39:42 2021

@author: arpan

@Description: Training Transformer model on BoV sequences for multi stream.
"""

import os
import sys
import numpy as np

sys.path.insert(0, '../cluster_strokes')
sys.path.insert(0, '../cluster_strokes/lib')

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from utils import autoenc_utils
from datasets.dataset import StrokeMultiFeatureSequenceDataset
from datasets.dataset import StrokeMultiFeaturePairsDataset
import copy
import time
import pickle
import model_transformer as tt
import attn_utils
from create_bovw import make_codebook
from create_bovw import create_bovw_onehot
import warnings

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore")

sys.path.insert(0, '../CricketStrokeLocalizationBOVW')
from extract_hoof_feats import extract_stroke_feats

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# "of_feats_grid20.pkl", "of_feats_val_grid20.pkl" ; "hoof_feats_b20.pkl"
# "2dcnn_feats_train.pkl" ; "3dcnn_feats_train.pkl" ; "hoof_feats_val_b20.pkl"
feat = ["of_feats_grid20.pkl", "hoof_feats_b20.pkl", "hog_feats.pkl"]
feat_val = ["of_feats_val_grid20.pkl", "hoof_feats_val_b20.pkl", "hog_feats_val.pkl"]
feat_test = ["of_feats_test_grid20.pkl", "hoof_feats_test_b20.pkl", "hog_feats_test.pkl"]
# "of_snames_grid20.pkl" ; "2dcnn_snames_train.pkl" ; "3dcnn_snames_train.pkl";
# "hoof_snames_b20.pkl"
snames = ["of_snames_grid20.pkl", "hoof_snames_b20.pkl", "hog_snames.pkl"]
snames_val = ["of_snames_val_grid20.pkl", "hoof_snames_val_b20.pkl", "hog_snames_val.pkl"] 
snames_test = ["of_snames_test_grid20.pkl", "hoof_snames_test_b20.pkl", "hog_snames_test.pkl"]

nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
cluster_size = 200
INPUT_SIZE = cluster_size      # OFGRID: 576, 3DCNN: 512, 2DCNN: 2048
HIDDEN_SIZE = nhid
N_LAYERS = nlayers
bidirectional = True

km_filename = "km_onehot"
log_path = "logs/bovtransMulti/HA_of20_hoofb20_HOG_Hidden200"
# bow_HL_ofAng_grid20 ; bow_HL_2dres ; bow_HL_3dres_seq16; bow_HL_hoof_b20_mth2
feat_path = ["/home/arpan/VisionWorkspace/Cricket/CricketStrokeLocalizationBOVW/logs/bow_HL_ofAng_grid20",
             "/home/arpan/VisionWorkspace/Cricket/CricketStrokeLocalizationBOVW/logs/bow_HL_hoof_b20_mth2",
             "/home/arpan/VisionWorkspace/Cricket/CricketStrokeLocalizationBOVW/logs/bow_HL_HOG"
             ]

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


def train_model(model, dataloaders, criterion, optimizer, scheduler, 
                labs_keys, labs_values, num_epochs=25):
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
            running_corrects = 0.0
            # Iterate over data.
            for bno, (inputs, vid_path, stroke, targets, _) in enumerate(dataloaders[phase]):
                # inputs of shape BATCH x SEQ_LEN x FEATURE_DIM
#                labels = attn_utils.get_batch_labels(vid_path, stroke, labs_keys, labs_values, 1)
                # Extract spatio-temporal features from clip using 3D ResNet (For SL >= 16)
                inputs = inputs.float()
                inp_emb = attn_utils.get_long_tensor(inputs)    # comment out for SA
                inputs = inp_emb.to(device)                     # comment out for SA
                targets = attn_utils.get_long_tensor(targets).to(device)
                inputs = inputs.t().contiguous()       # Convert to (SEQ, BATCH)
                targets = targets.t().contiguous().view(-1)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                output = model(inputs)  # output size (SEQ_SIZE, BATCH, NCLUSTERS)
                output = output.view(-1, INPUT_SIZE * len(feat_path))    # To (BATCH*SEQ_SIZE, NCLUSTERS)
                loss = criterion(output, targets)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                
                # track history if only in train
                _, preds = torch.max(output, 1)

                # statistics
                running_loss += loss.item()  #* inputs.size(0)
#                print("Iter : {} :: Running Loss : {}".format(bno, running_loss))
                running_corrects += torch.sum(preds == targets.data)
#                if bno==20:
#                    break

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / (inputs.size(0) * len(dataloaders[phase].dataset))

            print('{} Loss: {:.4f} Acc: {:.4f} LR: {}'.format(phase, epoch_loss, epoch_acc, 
                  scheduler.get_lr()[0]))

            if phase == 'train':
                scheduler.step()
#            # deep copy the model for best test accuracy
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, \
          time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def predict(features, stroke_names_id, model, dataloaders, labs_keys, labs_values, 
            seq, phase="val"):
    assert phase == "val" or phase=="test", "Incorrect Phase."
    model = model.eval()
    gt_list, pred_list, stroke_ids  = [], [], []
    # Iterate over data.
    for bno, (inputs, vid_path, stroke, targets, labels) in enumerate(dataloaders[phase]):
        # inputs of shape BATCH x SEQ_LEN x FEATURE_DIM
        labels = attn_utils.get_batch_labels(vid_path, stroke, labs_keys, labs_values, 1)
        inputs = inputs.float()
        inp_emb = attn_utils.get_long_tensor(inputs)    # comment out for SA
        inputs = inp_emb.to(device)                     # comment out for SA
        targets = attn_utils.get_long_tensor(targets).to(device)
        inputs = inputs.t().contiguous()
        targets = targets.t().contiguous().view(-1)
#        labels = labels.to(device)
        
        # forward
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)     # output size (BATCH, SEQ_SIZE, NCLUSTERS)
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
    
def save_model_checkpoint(base_name, model, ep, opt):
    """
    Save the optimizer state with epoch no. along with the weights
    """
    # Save only the model params
    model_name = os.path.join(base_name, "transformer_ep"+str(ep)+"_"+opt+".pt")

    torch.save(model.state_dict(), model_name)
    print("Model saved to disk... : {}".format(model_name))

def load_weights(base_name, model, ep, opt):
    """
    Load the pretrained weights to the models' encoder and decoder modules
    """
    # Paths to encoder and decoder files
    model_name = os.path.join(base_name, "transformer_ep"+str(ep)+"_"+opt+".pt")
    if os.path.isfile(model_name):
        model.load_state_dict(torch.load(model_name))
        print("Loading Transformer weights... : {}".format(model_name))
    return model

def extract_trans_feats(model, DATASET, LABELS, CLASS_IDS, BATCH_SIZE, 
                       SEQ_SIZE=16, STEP=16, partition='train', nstrokes=-1, base_name=""):
    '''
    Extract sequence features from AutoEncoder.
    
    Parameters:
    -----------
    model : tt.TransformerModel 
        TransformerModel object
    DATASET : str
        path to the video dataset
    LABELS : str
        path containing stroke labels
    BATCH_SIZE : int
        size for batch of clips
    SEQ_SIZE : int
        no. of frames in a clip
    STEP : int
        stride for next example. If SEQ_SIZE=16, STEP=8, use frames (0, 15), (8, 23) ...
    partition : str
        'train' / 'test' / 'val' : Videos to be considered
    nstrokes : int
        partial extraction of features (do not execute for entire dataset)
    base_name : str
        path containing the pickled feature dumps
    
    Returns:
    --------
    features_dictionary, stroke_names
    
    '''
    
    ###########################################################################
    # Read the strokes 
    # Divide the highlight dataset files into training, validation and test sets
    train_lst, val_lst, test_lst = autoenc_utils.split_dataset_files(DATASET)
    print("No. of training videos : {}".format(len(train_lst)))
    
    #####################################################################
    
    if partition == 'train':
        partition_lst = train_lst
        ft_path = [os.path.join(base_name, "F"+str(i+1)+"_C"+str(cluster_size)+"_train.pkl") \
                    for i in range(len(feat_path))]
    elif partition == 'val':
        partition_lst = val_lst
        ft_path = [os.path.join(base_name, "F"+str(i+1)+"_C"+str(cluster_size)+"_val.pkl") \
                    for i in range(len(feat_path))]
    elif partition == 'test':
        partition_lst = test_lst
        ft_path = [os.path.join(base_name, "F"+str(i+1)+"_C"+str(cluster_size)+"_test.pkl") \
                    for i in range(len(feat_path))]
    else:
        print("Partition should be : train / val / test")
        return
    
    ###########################################################################
    # Create a Dataset
    
    part_dataset = StrokeMultiFeatureSequenceDataset(ft_path, partition_lst, 
                                    DATASET, LABELS, CLASS_IDS, 
                                    frames_per_clip=SEQ_SIZE, extracted_frames_per_clip=2,
                                    step_between_clips=STEP, train=True)
    
    data_loader = DataLoader(dataset=part_dataset, batch_size=BATCH_SIZE, shuffle=False)

    ###########################################################################
    # Validate / Evaluate
    model.eval()
    stroke_names = []
    trajectories, stroke_traj = [], []
    num_strokes = 0
    prev_stroke = None
    print("Total Batches : {} :: BATCH_SIZE : {}".format(data_loader.__len__(), BATCH_SIZE))
    ###########################################################################
    for bno, (inputs, vid_path, stroke, labels) in enumerate(data_loader):
        # inputs of shape BATCH x SEQ_LEN x FEATURE_DIM
        inputs = inputs.float()
        inp_emb = attn_utils.get_long_tensor(inputs)    # comment out for SA
        inputs = inp_emb.t().contiguous().to(device)    # comment out for SA
        
        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            
            outputs = model.get_vec(inputs)  # output size (BATCH, SEQ_SIZE, NCLUSTERS)
            outputs = outputs.transpose(0, 1).contiguous()
#            output = output.view(-1, INPUT_SIZE)    # To (BATCH*SEQ_SIZE, NCLUSTERS)
#            loss = criterion(output, targets)
            
#            batch_size = inputs.size(0)
#            enc_h = encoder.init_hidden(batch_size)
#            enc_out, h = encoder(inputs, enc_h)
#            dec_h = h
#            dec_in = torch.zeros(batch_size, inputs.size(2)).to(device)
#            dec_out_lst = []
#            target_length = inputs.size(1)      # assign SEQ_LEN as target length for now
#            # run for each word of the sequence (use teacher forcing)
#            for ti in range(target_length):
#                dec_out, dec_h, dec_attn = decoder(dec_h, enc_out, dec_in)
#                dec_out_lst.append(dec_out)
#                dec_in = dec_out
    
#            outputs = torch.stack(dec_out_lst, dim=1)
            
        # convert to start frames and end frames from tensors to lists
        stroke = [s.tolist() for s in stroke]
        # outputs are the reconstructed features. Use compressed enc_out values(maybe wtd.).
        inputs_lst, batch_stroke_names = autoenc_utils.separate_stroke_tensors(outputs, \
                                                                    vid_path, stroke)
        
        # for sequence of features from batch segregated extracted features.
        if bno == 0:
            prev_stroke = batch_stroke_names[0]
        
        for enc_idx, enc_input in enumerate(inputs_lst):
            # get no of sequences that can be extracted from enc_input tensor
            nSeqs = enc_input.size(0)
            if prev_stroke != batch_stroke_names[enc_idx]:
                # append old stroke to trajectories
                if len(stroke_traj) > 0:
                    num_strokes += 1
                    trajectories.append(stroke_traj)
                    stroke_names.append(prev_stroke)
                    stroke_traj = []
            
#            enc_output = model.encoder(enc_input.to(device))
#            enc_output = enc_output.squeeze(axis=1).cpu().data.numpy()
            enc_output = enc_input.cpu().data.numpy()
            
            # convert to [[[stroke1(size 32 each) ... ], [], ...], [ [], ... ]]
            stroke_traj.extend([enc_output[i,j,:] for i in range(enc_output.shape[0]) \
                                                for j in range(enc_output.shape[1])])
            prev_stroke = batch_stroke_names[enc_idx]
            
        if nstrokes > -1 and num_strokes >= nstrokes:
            break
       
    # for last batch only if extracted for full dataset
    if len(stroke_traj) > 0 and nstrokes < 0:
        trajectories.append(stroke_traj)
        stroke_names.append(batch_stroke_names[-1])
    
    # convert to dictionary of features with keys as stroke names(with ext). 
    features = {}
    for i, t in enumerate(trajectories):
        features[stroke_names[i]] = np.array(t)
    
#    trajectories, stroke_names = autoenc_utils.group_strokewise(trajectories, stroke_names)
    
    return features, stroke_names


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
    ft_path, ft_path_val, ft_path_test = [], [], []
    for i, ft_dir in enumerate(feat_path):
        print("Feature : {}".format(ft_dir))
        features, stroke_names_id = attn_utils.read_feats(ft_dir, feat[i], snames[i])
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
    
        km_filepath = os.path.join(log_path, km_filename+"_F"+str(i+1))
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
        
        ft_path.append(os.path.join(log_path, "F"+str(i+1)+"_C"+str(cluster_size)+"_train.pkl"))
        with open(ft_path[-1], "wb") as fp:
            pickle.dump(onehot_feats, fp)
    
    ###########################################################################
    
        features_val, stroke_names_id_val = attn_utils.read_feats(ft_dir, feat_val[i], 
                                                                  snames_val[i])
        
        print("Create numpy one hot representation for val features...")
        onehot_feats_val = create_bovw_onehot(features_val, stroke_names_id_val, km_model)
        
        ft_path_val.append(os.path.join(log_path, "F"+str(i+1)+"_C"+str(cluster_size)+"_val.pkl"))
        with open(ft_path_val[-1], "wb") as fp:
            pickle.dump(onehot_feats_val, fp)
    
    ###########################################################################
    
        features_test, stroke_names_id_test = attn_utils.read_feats(ft_dir, feat_test[i], 
                                                                    snames_test[i])
        
        print("Create numpy one hot representation for test features...")
        onehot_feats_test = create_bovw_onehot(features_test, stroke_names_id_test, km_model)
        
        ft_path_test.append(os.path.join(log_path, "F"+str(i+1)+"_C"+str(cluster_size)+"_test.pkl"))
        with open(ft_path_test[-1], "wb") as fp:
            pickle.dump(onehot_feats_test, fp)
    
    ###########################################################################    
    # Create a Dataset    
    train_dataset = StrokeMultiFeaturePairsDataset(ft_path, train_lst, DATASET, LABELS, CLASS_IDS, 
                                         frames_per_clip=SEQ_SIZE, extracted_frames_per_clip=2,
                                         step_between_clips=STEP, train=True)
    val_dataset = StrokeMultiFeaturePairsDataset(ft_path_val, val_lst, DATASET, LABELS, CLASS_IDS, 
                                         frames_per_clip=SEQ_SIZE, extracted_frames_per_clip=2,
                                         step_between_clips=STEP, train=False)

#    # created weighted Sampler for class imbalance
#    samples_weight = attn_utils.get_sample_weights(train_dataset, labs_keys, labs_values, 
#                                                   train_lst)
#    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True) 
#                              sampler=sampler, worker_init_fn=np.random.seed(12))
    
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    data_loaders = {"train": train_loader, "test": val_loader}
    
    ###########################################################################    
    # get labels
    labs_keys, labs_values = attn_utils.get_cluster_labels(ANNOTATION_FILE)
    num_classes = len(list(set(labs_values)))
    
    ###########################################################################    
    
    # load model and set loss function
    ntokens = cluster_size * len(feat_path) # the size of vocabulary
    emsize = 200 # embedding dimension
    nhead = 2 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    model = tt.TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
    
#    model = load_weights(log_path, model, N_EPOCHS, 
#                                    "S"+str(SEQ_SIZE)+"C"+str(cluster_size)+"_SGD")
    
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
#    print("Params to learn:")
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
    
    # Observe that all parameters are being optimized
#    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = StepLR(optimizer, step_size=15, gamma=0.1)
    
#    lr = 5.0 # learning rate
#    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
#    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)    
    ###########################################################################
    # Training the model    
    
    start = time.time()
    
    model = train_model(model, data_loaders, criterion, optimizer, scheduler, 
                        labs_keys, labs_values, num_epochs=N_EPOCHS)
    
    end = time.time()
    
#    # save the best performing model
    save_model_checkpoint(log_path, model, N_EPOCHS, 
                                     "S"+str(SEQ_SIZE)+"C"+str(cluster_size)+"_SGD")
    # Load model checkpoints
    model = load_weights(log_path, model, N_EPOCHS, 
                                    "S"+str(SEQ_SIZE)+"C"+str(cluster_size)+"_SGD")
    
    print("Total Execution time for {} epoch : {}".format(N_EPOCHS, (end-start)))

    ###########################################################################
    
#    acc = predict(features_val, stroke_names_id_val, model, data_loaders, labs_keys, 
#                  labs_values, SEQ_SIZE, phase='test')
    
    ###########################################################################
    
    # Extract attention model features 
    if not os.path.isfile(os.path.join(log_path, "trans_feats.pkl")):
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        #    # Extract Grid OF / HOOF features {mth = 2, and vary nbins}
        print("Training extraction ... ")
        feats_dict, stroke_names = extract_trans_feats(model, DATASET, LABELS, 
                                                      CLASS_IDS, BATCH_SIZE, SEQ_SIZE, 
                                                      SEQ_SIZE-1, partition='train', nstrokes=nstrokes, 
                                                      base_name=log_path)

        with open(os.path.join(log_path, "trans_feats.pkl"), "wb") as fp:
            pickle.dump(feats_dict, fp)
        with open(os.path.join(log_path, "trans_snames.pkl"), "wb") as fp:
            pickle.dump(stroke_names, fp)
            
    if not os.path.isfile(os.path.join(log_path, "trans_feats_val.pkl")):
        print("Validation extraction ....")
        feats_dict_val, stroke_names_val = extract_trans_feats(model, DATASET, LABELS, 
                                                      CLASS_IDS, BATCH_SIZE, SEQ_SIZE, 
                                                      SEQ_SIZE-1, partition='val', nstrokes=nstrokes, 
                                                      base_name=log_path)

        with open(os.path.join(log_path, "trans_feats_val.pkl"), "wb") as fp:
            pickle.dump(feats_dict_val, fp)
        with open(os.path.join(log_path, "trans_snames_val.pkl"), "wb") as fp:
            pickle.dump(stroke_names_val, fp)
    
    if not os.path.isfile(os.path.join(log_path, "trans_feats_test.pkl")):
        print("Testing extraction ....")
        feats_dict_val, stroke_names_val = extract_trans_feats(model, DATASET, LABELS, 
                                                      CLASS_IDS, BATCH_SIZE, SEQ_SIZE, 
                                                      SEQ_SIZE-1, partition='test', nstrokes=nstrokes, 
                                                      base_name=log_path)

        with open(os.path.join(log_path, "trans_feats_test.pkl"), "wb") as fp:
            pickle.dump(feats_dict_val, fp)
        with open(os.path.join(log_path, "trans_snames_test.pkl"), "wb") as fp:
            pickle.dump(stroke_names_val, fp)
            
    # call count_paramters(model)  for displaying total no. of parameters
    print("#Parameters : {} ".format(autoenc_utils.count_parameters(model)))
    return 0


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

    print("OF20 BOV Transformer HA with Embedding...")
    print("EPOCHS = {} : HIDDEN_SIZE = {} : LAYERS = {}".format(N_EPOCHS, 
          HIDDEN_SIZE, N_LAYERS))
    for SEQ_SIZE in seq_sizes:
        print("SEQ_SIZE : {} :: CLUSTER_SIZE : {}".format(SEQ_SIZE, cluster_size))
        acc.append(main(DATASET, LABELS, CLASS_IDS, BATCH_SIZE, ANNOTATION_FILE,
                        SEQ_SIZE, STEP, nstrokes=-1, N_EPOCHS=N_EPOCHS))
        
    print("*"*60)
    print("SEQ_SIZES : {}".format(seq_sizes))
    print("Accuracy values : {}".format(acc))
