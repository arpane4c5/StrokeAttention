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
from torch import nn, optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

#from utils.resnet_feature_extracter import Clip2Vec
from utils import autoenc_utils
#import datasets.videotransforms as videotransforms
from datasets.dataset import StrokeFeatureSequenceDataset
from datasets.dataset import StrokeFeaturePairsDataset
import copy
import time
import pickle
import attn_model
import attn_utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_path = "/home/arpan/VisionWorkspace/Cricket/CricketStrokeLocalizationBOVW/logs"
ft_dir = "bow_HL_ofAng_grid20"
feat, feat_val, feat_test = "of_feats_grid20.pkl", "of_feats_val_grid20.pkl", "of_feats_test_grid20.pkl"
snames, snames_val = "of_snames_grid20.pkl", "of_snames_val_grid20.pkl"
INPUT_SIZE, TARGET_SIZE = 576, 576      # OFGRID: 576, 3DCNN: 512, 2DCNN: 2048
HIDDEN_SIZE = 256 #64#1024
bidirectional = False


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
                labels = attn_utils.get_batch_labels(vid_path, stroke, labs_keys, labs_values, 1)
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
                    
            if phase == 'train':
                scheduler.step()

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

def predict(features, stroke_names_id, encoder, decoder, dataloaders, labs_keys, 
            labs_values, phase='test'):
    encoder = encoder.eval()
    decoder = decoder.eval()
    # Iterate over data.
    for bno, (inputs, targets, vid_path, stroke, _) in enumerate(dataloaders[phase]):
        # inputs of shape BATCH x SEQ_LEN x FEATURE_DIM
#        labels = attn_utils.get_batch_labels(vid_path, stroke, labs_keys, labs_values, 1)
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        # forward
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
            dec_in = dec_out
        outputs = torch.stack(dec_out_lst, dim=1)
        
        # statistics
#        running_loss += loss.item()
#                print("Iter : {} :: Running Loss : {}".format(bno, running_loss))
#                running_corrects += torch.sum(preds == labels.data)
        
#                print("Batch No : {} / {}".format(bno, len(dataloaders[phase])))
#                if (bno+1) % 10 == 0:
#                    break

    
def extract_attn_feats(encoder, decoder, DATASET, LABELS, CLASS_IDS, BATCH_SIZE, 
                       SEQ_SIZE=16, STEP=16, partition='train', nstrokes=-1, base_name=""):
    '''
    Extract sequence features from AutoEncoder.
    
    Parameters:
    -----------
    encoder, decoder : attn_model.Encoder 
        relative path to the checkpoint file for Autoencoder
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
        ft_path = os.path.join(base_name, ft_dir, feat)
    elif partition == 'val':
        partition_lst = val_lst
        ft_path = os.path.join(base_name, ft_dir, feat_val)
    elif partition == 'test':
        partition_lst = test_lst
        ft_path = os.path.join(base_name, ft_dir, feat_test)
    else:
        print("Partition should be : train / val / test")
        return
        
    ###########################################################################
    # Create a Dataset
    
    part_dataset = StrokeFeatureSequenceDataset(ft_path, partition_lst, DATASET, LABELS, CLASS_IDS, 
                                         frames_per_clip=SEQ_SIZE, extracted_frames_per_clip=2,
                                         step_between_clips=STEP, train=True)
    
    data_loader = DataLoader(dataset=part_dataset, batch_size=BATCH_SIZE, shuffle=False)

    ###########################################################################
    # Validate / Evaluate
    encoder.eval()
    decoder.eval()
    stroke_names = []
    trajectories, stroke_traj = [], []
    num_strokes = 0
    prev_stroke = None
    print("Total Batches : {} :: BATCH_SIZE : {}".format(data_loader.__len__(), BATCH_SIZE))
    ###########################################################################
    for bno, (inputs, vid_path, stroke, labels) in enumerate(data_loader):
        # inputs of shape BATCH x SEQ_LEN x FEATURE_DIM
        inputs = inputs.to(device)
        
        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            
            batch_size = inputs.size(0)
            enc_h = encoder.init_hidden(batch_size)
            enc_out, h = encoder(inputs, enc_h)
            dec_h = h
            dec_in = torch.zeros(batch_size, inputs.size(2)).to(device)
            dec_out_lst = []
            target_length = inputs.size(1)      # assign SEQ_LEN as target length for now
            # run for each word of the sequence (use teacher forcing)
            for ti in range(target_length):
                dec_out, dec_h, dec_attn = decoder(dec_h, enc_out, dec_in)
                dec_out_lst.append(dec_out)
                dec_in = dec_out
    
            outputs = torch.stack(dec_out_lst, dim=1)
            
        # convert to start frames and end frames from tensors to lists
        stroke = [s.tolist() for s in stroke]
        # outputs are the reconstructed features. Use compressed enc_out values(maybe wtd.).
        inputs_lst, batch_stroke_names = autoenc_utils.separate_stroke_tensors(enc_out, \
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

def train_attn_model(DATASET, LABELS, CLASS_IDS, BATCH_SIZE, ANNOTATION_FILE, 
                   SEQ_SIZE=16, STEP=16, nstrokes=-1, N_EPOCHS=25, base_name="",
                   dest_dir=""):
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
    
    encoder = attn_model.Encoder(INPUT_SIZE, HIDDEN_SIZE, 1, bidirectional)
    decoder = attn_model.AttentionDecoder(HIDDEN_SIZE*(1+bidirectional), TARGET_SIZE, 
                               max_length=SEQ_SIZE-2+1)
    encoder, decoder = encoder.to(device), decoder.to(device)
        
    # Setup the loss fxn
#    criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    
#    # Layers to finetune. Last layer should be displayed
#    params_to_update = model.parameters()
#    print("Params to learn:")
#    params_to_update = []
#    for name, param in model.named_parameters():
#        if param.requires_grad == True:
#            params_to_update.append(param)
#            print("\t",name)
    
    # Observe that all parameters are being optimized
#    optimizer_ft = torch.optim.Adam(params_to_update, lr=0.001)
    encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=0.01, momentum=0.9)
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=0.01, momentum=0.9)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    lr_scheduler = StepLR(encoder_optimizer, step_size=10, gamma=0.1)
    
#    # Observe that all parameters are being optimized
#    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    
    features, stroke_names_id = attn_utils.read_feats(os.path.join(base_name, ft_dir), feat, snames)
    
#    ###########################################################################
    # Training the model        
    start = time.time()
    
#    (encoder, decoder) = train_model(features, stroke_names_id, encoder, decoder, data_loaders, criterion, 
#                           encoder_optimizer, decoder_optimizer, lr_scheduler, labs_keys, labs_values,
#                           num_epochs=N_EPOCHS)
        
    end = time.time()
    
    # save the best performing model
#    attn_utils.save_attn_model_checkpoint(base_name, (encoder, decoder), N_EPOCHS, "Adam")
    # Load model checkpoints
    encoder, decoder = attn_utils.load_attn_model_checkpoint(base_name, encoder, decoder, N_EPOCHS, "Adam")
    
    print("Total Execution time for {} epoch : {}".format(N_EPOCHS, (end-start)))

#    ###########################################################################    
    
    features_val, stroke_names_id_val = attn_utils.read_feats(os.path.join(base_name, ft_dir), 
                                                              feat_val, snames_val)
    
#    predict(features_val, stroke_names_id_val, encoder, decoder, data_loaders,
#            labs_keys, labs_values)

    ###########################################################################
    # Extract attention model features 
    
    if not os.path.isfile(os.path.join(dest_dir, "attn_feats.pkl")):
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        #    # Extract Grid OF / HOOF features {mth = 2, and vary nbins}
        print("Training extraction ... ")
        feats_dict, stroke_names = extract_attn_feats(encoder, decoder, DATASET, LABELS, 
                                                      CLASS_IDS, BATCH_SIZE, SEQ_SIZE, 
                                                      STEP, partition='train', nstrokes=nstrokes, 
                                                      base_name=base_name)
        with open(os.path.join(dest_dir, "attn_feats.pkl"), "wb") as fp:
            pickle.dump(feats_dict, fp)
        with open(os.path.join(dest_dir, "attn_snames.pkl"), "wb") as fp:
            pickle.dump(stroke_names, fp)
                
    if not os.path.isfile(os.path.join(dest_dir, "attn_feats_val.pkl")):
        print("Validation extraction ....")
        feats_dict_val, stroke_names_val = extract_attn_feats(encoder, decoder, DATASET, LABELS, 
                                                      CLASS_IDS, BATCH_SIZE, SEQ_SIZE, 
                                                      STEP, partition='val', nstrokes=nstrokes, 
                                                      base_name=base_name)
        with open(os.path.join(dest_dir, "attn_feats_val.pkl"), "wb") as fp:
            pickle.dump(feats_dict_val, fp)
        with open(os.path.join(dest_dir, "attn_snames_val.pkl"), "wb") as fp:
            pickle.dump(stroke_names_val, fp)
            
    if not os.path.isfile(os.path.join(dest_dir, "attn_feats_test.pkl")):
        print("Testing extraction ....")
        feats_dict_test, stroke_names_test = extract_attn_feats(encoder, decoder, DATASET, LABELS, 
                                                      CLASS_IDS, BATCH_SIZE, SEQ_SIZE, 
                                                      STEP, partition='test', nstrokes=nstrokes, 
                                                      base_name=base_name)
        with open(os.path.join(dest_dir, "attn_feats_test.pkl"), "wb") as fp:
            pickle.dump(feats_dict_test, fp)
        with open(os.path.join(dest_dir, "attn_snames_test.pkl"), "wb") as fp:
            pickle.dump(stroke_names_test, fp)    
    
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
    N_EPOCHS = 30
    
    extract_dir = os.path.join(base_path, "attn_seq"+str(SEQ_SIZE))
    trajectories, stroke_names = train_attn_model(DATASET, LABELS, CLASS_IDS, 
                                                BATCH_SIZE, ANNOTATION_FILE,
                                                SEQ_SIZE, STEP, nstrokes=-1, 
                                                N_EPOCHS=N_EPOCHS, base_name=base_path,
                                                dest_dir=extract_dir)



