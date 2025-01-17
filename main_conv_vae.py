#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 23:43:52 2020

@author: arpan

@Description: Train a Conv VAE model on strokes. Problem: 
"""

import os
import sys
import numpy as np


sys.path.insert(0, '../cluster_strokes')
sys.path.insert(0, '../cluster_strokes/lib')

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
import datasets.videotransforms as videotransforms
#from datasets.dataset import StrokeFeatureSequenceDataset
from datasets.dataset import CricketStrokesFlowDataset, CricketStrokesDataset
import copy
import time
import pickle
import conv_attn_model
import conv_encdec_model
import attn_utils

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
base_path = "/home/arpan/VisionWorkspace/Cricket/StrokeAttention/logs"
#INPUT_SIZE, TARGET_SIZE = 576, 576      # OFGRID: 576, 3DCNN: 512, 2DCNN: 2048
HIDDEN_SIZE = 128 #64#1024
bidirectional = False
SHIFT = 4

def save_model_checkpoint(base_name, model, ep, opt):
    """
    Save the optimizer state with epoch no. along with the weights
    """
    # Save only the model params
    model_name = os.path.join(base_name, "convVAE_ep"+str(ep)+"_"+opt+".pt")

    torch.save(model.state_dict(), model_name)
    print("Model saved to disk... : {}".format(model_name))

def load_weights(base_name, model, ep, opt):
    """
    Load the pretrained weights to the models' encoder and decoder modules
    """
    # Paths to encoder and decoder files
    model_name = os.path.join(base_name, "convVAE_ep"+str(ep)+"_"+opt+".pt")
    if os.path.isfile(model_name):
        model.load_state_dict(torch.load(model_name))
        print("Loading ConvVAE weights... : {}".format(model_name))
    return model

def train_model(model, dataloaders, criterion, optimizer,
                scheduler, labs_keys, labs_values, seq=8, num_epochs=25):
    since = time.time()

#    best_model_wts = copy.deepcopy(model.state_dict())
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

            # Iterate over data. or (inputs, vid_path, stroke, flow, labels)
            for bno, (inputs, vid_path, stroke, labels) in enumerate(dataloaders[phase]):
                # inputs of shape BATCH x SEQ_LEN x FEATURE_DIM
#                print("Batch No : {} / {}".format(bno, len(dataloaders[phase])))
#                labels = attn_utils.get_batch_labels(vid_path, stroke, labs_keys, labs_values, 1)
                # Extract spatio-temporal features from clip using 3D ResNet (For SL >= 16)
                inputs = inputs.permute(0, 2, 1, 3, 4).float()
                inputs = inputs.to(device)
#                b, t, c, h, w = flow.shape
#                flow = torch.cat((torch.zeros(b, 1, c, h, w), flow), 1)
#                flow = flow.permute(0, 2, 1, 3, 4).float()
##                flow = flow.permute(0, 4, 1, 2, 3)
#                flow = flow.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
#                decoder_optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
                    batch_size = inputs.size(0)
#                    enc_h = encoder._init_hidden(batch_size)
                    out, mu, logvar = model(inputs)
#                    dec_out = decoder(enc_out)
                    batch_loss = conv_encdec_model.loss_function(out, inputs, mu, logvar) # or (out, flow, mu, logvar)

###############################################################################
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        batch_loss.backward()
                        optimizer.step()

                # statistics
                running_loss += batch_loss.item()
#                print("Iter : {} / {} :: Running Loss : {}".format(bno, 
#                      len(dataloaders[phase]), running_loss))
                
#                print("Batch No : {} / {}".format(bno, len(dataloaders[phase])))
#                if (bno+1) % 50 == 0:
#                    break
                    
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss #/ len(dataloaders[phase].dataset)
#            epoch_acc = running_corrects.double() #/ len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

#            # deep copy the model
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


def extract_attn_feats(model, DATASET, LABELS, CLASS_IDS, BATCH_SIZE, 
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
    elif partition == 'val':
        partition_lst = val_lst
    elif partition == 'test':
        partition_lst = test_lst
    else:
        print("Partition should be : train / val / test")
        return
        
    ###########################################################################
    # Create a Dataset
    # Clip level transform. Use this with framewiseTransform flag turned off
    clip_transform = transforms.Compose([videotransforms.CenterCrop(224),
                                         videotransforms.ToPILClip(), 
                                         videotransforms.Resize((112, 112)),
#                                         videotransforms.RandomCrop(112), 
                                         videotransforms.ToTensor(), 
#                                         videotransforms.Normalize(),
                                        #videotransforms.RandomHorizontalFlip(),\
                                        ])    
    part_dataset = CricketStrokesDataset(partition_lst, DATASET, LABELS, CLASS_IDS, 
                                         frames_per_clip=SEQ_SIZE, step_between_clips=STEP, 
                                         train=False, framewiseTransform=False, 
                                         transform=clip_transform)
    
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
        inputs = inputs.permute(0, 2, 1, 3, 4).float()
        inputs = inputs.to(device)
#        print("Batch No : {} / {}".format(bno, len(data_loader)))
        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            
            recon_x, mu, logvar = model(inputs)

#            dec_out_lst = []
#            dec_out_lst.append(out_mu)
#
#            outputs = torch.stack(dec_out_lst, dim=1)
            
        # convert to start frames and end frames from tensors to lists
        stroke = [s.tolist() for s in stroke]
        # outputs are the reconstructed features. Use compressed enc_out values(maybe wtd.).
        inputs_lst, batch_stroke_names = autoenc_utils.separate_stroke_tensors(mu, \
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
            stroke_traj.extend([enc_output[i,:] for i in range(enc_output.shape[0])])
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
         STEP=16, nstrokes=-1, N_EPOCHS=25, base_name=""):
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
    # seed everything
    
    
    if not os.path.isdir(base_name):
        os.makedirs(base_name)
    
    # Read the strokes 
    # Divide the highlight dataset files into training, validation and test sets
    train_lst, val_lst, test_lst = autoenc_utils.split_dataset_files(DATASET)
    print("No. of training videos : {}".format(len(train_lst)))
    
    ###########################################################################
    # Create a Dataset    
    # Clip level transform. Use this with framewiseTransform flag turned off
    clip_transform = transforms.Compose([videotransforms.CenterCrop(224),
                                         videotransforms.ToPILClip(), 
                                         videotransforms.Resize((112, 112)),
#                                         videotransforms.RandomCrop(112), 
                                         videotransforms.ToTensor(), 
                                         videotransforms.Normalize(),
                                        #videotransforms.RandomHorizontalFlip(),\
                                        ])
    # or use CricketStrokesFlowDataset
    train_dataset = CricketStrokesDataset(train_lst, DATASET, LABELS, CLASS_IDS, 
                                         frames_per_clip=SEQ_SIZE, step_between_clips=STEP, 
                                         train=True, framewiseTransform=False, 
                                         transform=clip_transform)
    # or use CricketStrokesFlowDataset
    val_dataset = CricketStrokesDataset(val_lst, DATASET, LABELS, CLASS_IDS, 
                                        frames_per_clip=SEQ_SIZE, step_between_clips=STEP, 
                                        train=False, framewiseTransform=False, 
                                        transform=clip_transform)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    data_loaders = {"train": train_loader, "test": val_loader}

    ###########################################################################
    
    labs_keys, labs_values = attn_utils.get_cluster_labels(ANNOTATION_FILE)
    
    num_classes = len(list(set(labs_values)))
    
    ###########################################################################    
    # load model and set loss function
    model = conv_encdec_model.ConvVAE()
    
    model = model.to(device)
#    # load checkpoint:
    
    # Setup the loss fxn
    criterion = nn.MSELoss()
    
#    # Layers to finetune. Last layer should be displayed
    print("Params to learn:")
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("ConvVAE : {}".format(name))
    
    # Observe that all parameters are being optimized
#    optimizer_ft = torch.optim.Adam(params_to_update, lr=0.001)
    optimizer_ft = torch.optim.SGD(params_to_update, lr=0.01, momentum=0.9)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    lr_scheduler = StepLR(optimizer_ft, step_size=10, gamma=0.1)
    
#    # Observe that all parameters are being optimized
#    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
        
    ###########################################################################
    # Training the model
    start = time.time()
    
    model = train_model(model, data_loaders, criterion, optimizer_ft,
                        lr_scheduler, labs_keys, labs_values, seq=8,
                        num_epochs=N_EPOCHS)
    
    end = time.time()
    
    print("Total Execution time for {} epoch : {}".format(N_EPOCHS, (end-start)))
    ###########################################################################    
    # Save only the model params
    model_name = os.path.join(base_name, "conv_vae_ep"+str(N_EPOCHS)+"_SGD.pt")

#    torch.save(model.state_dict(), model_name)
#    print("Model saved to disk... : {}".format(model_name))    # Load model checkpoints
    
    # Loading the saved model 
    model_name = os.path.join(base_name, "conv_vae_ep"+str(N_EPOCHS)+"_SGD.pt")
    if os.path.isfile(model_name):
        model.load_state_dict(torch.load(model_name))
        print("Loading ConvVAE weights... : {}".format(model_name))
    
    ###########################################################################    
    
    print("Writing prediction dictionary....")
#    pred_out_dict = predict(encoder, decoder, data_loaders, criterion, labs_keys, 
#                            labs_values, phase='test')
    if not os.path.isfile(os.path.join(base_name, "conv_vae_train.pkl")):
        if not os.path.exists(base_name):
            os.makedirs(base_name)
        feats_dict, stroke_names = extract_attn_feats(model, DATASET, 
                                                      LABELS, CLASS_IDS, BATCH_SIZE, 
                                                      SEQ_SIZE, 16, 'train', -1, 
                                                      base_name)
        with open(os.path.join(base_name, "conv_vae_train.pkl"), "wb") as fp:
            pickle.dump(feats_dict, fp)
        with open(os.path.join(base_name, "conv_vae_snames_train.pkl"), "wb") as fp:
            pickle.dump(stroke_names, fp)
    if not os.path.isfile(os.path.join(base_name, "conv_vae_val.pkl")):
        if not os.path.exists(base_name):
            os.makedirs(base_name)
        feats_dict, stroke_names = extract_attn_feats(model, DATASET, 
                                                      LABELS, CLASS_IDS, BATCH_SIZE, 
                                                      SEQ_SIZE, 16, 'val', -1, 
                                                      base_name)
        with open(os.path.join(base_name, "conv_vae_val.pkl"), "wb") as fp:
            pickle.dump(feats_dict, fp)
        with open(os.path.join(base_name, "conv_vae_snames_val.pkl"), "wb") as fp:
            pickle.dump(stroke_names, fp)
    if not os.path.isfile(os.path.join(base_name, "conv_vae_test.pkl")):
        if not os.path.exists(base_name):
            os.makedirs(base_name)
        feats_dict, stroke_names = extract_attn_feats(model, DATASET, 
                                                      LABELS, CLASS_IDS, BATCH_SIZE, 
                                                      SEQ_SIZE, 16, 'test', -1, 
                                                      base_name)
        with open(os.path.join(base_name, "conv_vae_test.pkl"), "wb") as fp:
            pickle.dump(feats_dict, fp)
        with open(os.path.join(base_name, "conv_vae_snames_test.pkl"), "wb") as fp:
            pickle.dump(stroke_names, fp)
    
    print("#Parameters ConvVAE : {} ".format(autoenc_utils.count_parameters(model)))
    
    return model

if __name__ == '__main__':
    # Local Paths
    LABELS = "/home/arpan/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
    DATASET = "/home/arpan/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
    CLASS_IDS = "/home/arpan/VisionWorkspace/Cricket/cluster_strokes/configs/Class Index_Strokes.txt"
    ANNOTATION_FILE = "/home/arpan/VisionWorkspace/Cricket/CricketStrokeLocalizationBOVW/shots_classes.txt"

    SEQ_SIZE = 8
    STEP = 4
    BATCH_SIZE = 128
    N_EPOCHS = 30
    
    extract_path = os.path.join(base_path, "conv_vae_seq"+str(SEQ_SIZE))  # flowviz
    model =  main(DATASET, LABELS, CLASS_IDS, BATCH_SIZE, ANNOTATION_FILE, SEQ_SIZE, 
                  STEP, nstrokes=-1, N_EPOCHS=N_EPOCHS, base_name=extract_path)

