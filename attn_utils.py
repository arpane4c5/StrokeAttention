#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 12:18:48 2020

@author: arpan

@Description: Utility functions for Attention model and GRU models
"""

import torch
import csv
import numpy as np
import os
import pickle
import random
from datasets.dataset import CricketStrokesDataset


def seed_everything(seed=1234):
#    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

def get_cluster_labels(cluster_labels_path):
    '''Read the stroke names and category labels from file.
    Parameters : 
    ------------
    cluster_labels_path : str
        complete path to the stroke_labels.txt file
        
    Returns:
    --------
    labs_keys : list of stroke_keys (strokename_stFrm_endFrm)
    labs_values : list of int (category labels) from {0, 1, 2, 3, 4}
    '''
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

def get_long_tensor(inputs):
    '''Used while BoW training and prediction. Converts a batch of input onehot tensors
    to the corresponding word indices long tensor for input to the embedding layer.
    Params:
    -------
    inputs : torch.tensor
        batch onehot inputs of size (B, Seq, N_clusters)
    
    Returns:
    --------
    torch.LongTensor of size (B, Seq) with each onehot vector represented by cluster no.
    '''
    t = torch.zeros((inputs.size(0), inputs.size(1)), dtype=torch.int64)
    clus_indexes = inputs.nonzero()
#    [[seq_item.nonzero().item()] for bat_item in inputs for seq_item in bat_item]
    for ind in clus_indexes:
        t[ind[0], ind[1]] = ind[2]
    return t

def get_batch_labels(vid_path, stroke, labs_keys, labs_values, seq_len):
    '''
    Retrieve stroke labels for a batch of input sequences from the list of keys and
    category labels. 
    Params:
    ------
    vid_path : list of str
        list of video paths for a batch of samples
    stroke : list of Tensors
        2 lists of tensors for starting frames and ending frames of strokes, each of 
        size batch_size
    labs_keys : list of str
        list of stroke_ids for entire dataset
    labs_values : list of int
        list of category labels corresponding to the stroke_ids for the entire dataset
    seq_len : int
        No. of vectors in a sequence. May be used when prediction is done on all
        the output sequence elements. Softmax on output sequence instead of hidden
        context.
        
    Returns:
    --------
    torch.tensor : list of int values of batch_size representing category labels.
    '''
    labels = []
    for i, vid_name in enumerate(vid_path):
        
        vid_name = vid_name.rsplit('/', 1)[1]
        if '.avi' in vid_name:
            vid_name = vid_name.replace('.avi', '')
        elif '.mp4' in vid_name:
            vid_name = vid_name.replace('.mp4', '')
        stroke_name = vid_name+"_"+str(stroke[0][i].item())+"_"+str(stroke[1][i].item())
        assert stroke_name in labs_keys, "Key does not match : {}".format(stroke_name)
        labels.extend([labs_values[labs_keys.index(stroke_name)]] * seq_len)
        
    return torch.tensor(labels)
        
def save_model_checkpoint(base_name, model, ep, opt):
    """
    Save the optimizer state with epoch no. along with the weights
    """
    # Save only the model params
    model_name = os.path.join(base_name, "gru_classifier_ep"+str(ep)+"_"+opt+".pt")

    torch.save(model.state_dict(), model_name)
    print("Model saved to disk... : {}".format(model_name))

def load_weights(base_name, model, ep, opt):
    """
    Load the pretrained weights to the models' encoder and decoder modules
    """
    # Paths to encoder and decoder files
    model_name = os.path.join(base_name, "gru_classifier_ep"+str(ep)+"_"+opt+".pt")
    if os.path.isfile(model_name):
        model.load_state_dict(torch.load(model_name))
        print("Loading Encoder weights... : {}".format(model_name))
    return model

def save_attn_model_checkpoint(base_name, model, ep, opt):
    """
    Save the optimizer state with epoch no. along with the weights
    """
    # Save only the model params
    enc_name = os.path.join(base_name, "enc_attn_ep"+str(ep)+"_"+opt+".pt")
    dec_name = os.path.join(base_name, "dec_attn_ep"+str(ep)+"_"+opt+".pt")

    torch.save(model[0].state_dict(), enc_name)
    torch.save(model[1].state_dict(), dec_name)
    print("Model saved to disk... \n Enc : {} \n Dec : {}".format(enc_name, dec_name))
    
def load_attn_model_checkpoint(base_name, encoder, decoder, ep, opt):
    """
    Load the pretrained weights from disk
    """
    # Paths to encoder and decoder files
    enc_name = os.path.join(base_name, "enc_attn_ep"+str(ep)+"_"+opt+".pt")
    dec_name = os.path.join(base_name, "dec_attn_ep"+str(ep)+"_"+opt+".pt")
    if os.path.isfile(enc_name):
        encoder.load_state_dict(torch.load(enc_name))
        print("Loading Encoder weights... : {}".format(enc_name))
    if os.path.isfile(dec_name):
        decoder.load_state_dict(torch.load(dec_name))
        print("Loading Decoder weights... : {}".format(dec_name))
    return encoder, decoder    

def read_feats(base_name, feat, snames):
    with open(os.path.join(base_name, feat), "rb") as fp:
        features = pickle.load(fp)
    with open(os.path.join(base_name, snames), "rb") as fp:
        strokes_name_id = pickle.load(fp)
    return features, strokes_name_id

def get_sample_weights(train_dataset, labs_keys, labs_values, train_lst):
    # filter 
    
    train_lst = [t.rsplit('.', 1)[0] for t in train_lst]
    n_classes = len(list(set(labs_values)))
    train_keys, train_values = [], []
    sample_counts = [0.] * n_classes
    for i, key in enumerate(labs_keys):
        if '/' in key:
            key = key.rsplit('/', 1)[1]
        k = key.rsplit('_', 2)[0].rsplit('.', 1)[0]
        if k in train_lst:
            train_keys.append(key)
            train_values.append(labs_values[i])
            
    train_set_keys = []
    # count the number of samples for each class
    for i in range(train_dataset.__len__()):
        if isinstance(train_dataset, CricketStrokesDataset):
            _, vpath, stroke, *_ = train_dataset.video_clips.get_clip(i)
        else:
            seq, vpath, stroke, *_ = train_dataset.__getitem__(i)
        key = vpath.rsplit('/', 1)[1].rsplit('.', 1)[0]+'_'+\
                str(stroke[0])+'_'+str(stroke[1])
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


#def predict_videowise(features, stroke_names_id, model, dataloaders, labs_keys, labs_values, 
#            phase="val"):
#    assert phase == "val" or phase=="test", "Incorrect Phase."
#    model = model.eval()
#    gt_list, pred_list, stroke_ids  = [], [], []
#    # Iterate over data.
#    for bno, (inputs, vid_path, stroke, labels) in enumerate(dataloaders[phase]):
#        # inputs of shape BATCH x SEQ_LEN x FEATURE_DIM
#        labels = get_batch_labels(vid_path, stroke, labs_keys, labs_values, 1)
#        
#        inputs = inputs.to(device)
#        labels = labels.to(device)
#        # forward
#        if bno == 0:
#            prev_stroke = vid_path[0]+"_"+str(stroke[0][0].item())+"_"+str(stroke[1][0].item())
#            hidden = model.init_hidden(1)
#        for i, vid in enumerate(vid_path):
#            curr_stroke = vid+"_"+str(stroke[0][i].item())+"_"+str(stroke[1][i].item())
##            stroke_ids.extend([vid+"_"+str(stroke[0][i].item())+"_"+str(stroke[1][i].item())] * 1)
#            if curr_stroke != prev_stroke:
#                hidden = model.init_hidden(1)
#            outputs, hidden = model(inputs[i].unsqueeze(0), hidden)
#            gt_list.append(labels[i].item())
#            pred_list.append((torch.max(outputs, 1)[1]).tolist()[0])
#            stroke_ids.append(curr_stroke)
#            prev_stroke = curr_stroke
#                
##    epoch_loss = running_loss #/ len(dataloaders[phase].dataset)
##            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
##    print('{} Loss: {:.4f}'.format(phase, epoch_loss))
#                
#    confusion_mat = np.zeros((model.n_classes, model.n_classes))
##    gt_list = [g for batch_list in gt_list for g in batch_list]
##    pred_list = [p for batch_list in pred_list for p in batch_list]
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
    

#    # Layers to finetune. Last layer should be displayed
#    params_to_update = model.parameters()
#    print("Fix Weights : ")
#    for name, param in model.encoder.named_parameters():
#        print("Encoder : {}".format(name))
#        param.requires_grad = False
#    for name, param in model.decoder.named_parameters():
#        print("Decoder : {}".format(name))
#        param.requires_grad = False
##    model.decoder.train(False)