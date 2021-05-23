#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 23:43:52 2020

@author: arpan

@Description: Finetune a C3D model on strokes.
"""

import os
import sys
import numpy as np

sys.path.insert(0, '../localization_finetuneC3D')
sys.path.insert(0, '../cluster_strokes')
sys.path.insert(0, '../cluster_strokes/lib')

##from utils import spectral_utils
#from utils import plot_utils
#from evaluation import eval_of_clusters
import torch
from torch import nn, optim
from torchvision import transforms
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

#from utils.resnet_feature_extracter import Clip2Vec
from utils import autoenc_utils
import datasets.videotransforms as videotransforms
#from datasets.dataset import StrokeFeatureSequenceDataset
from datasets.dataset import CricketStrokesDataset
import copy
import time
import pickle
import attn_utils
#import model_c3d as c3d_pre
import model_c3d_finetune as c3d

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
base_path = "/home/arpan/VisionWorkspace/Cricket/StrokeAttention/logs/c3dFine"
wts_path = 'c3d.pickle'
HIDDEN_SIZE = 1024 #64#1024

def vis_samples(dataset, normalize=False):
#    import cv2
    from matplotlib import pyplot as plt
    inputs, *_ = dataset.__getitem__(456)
    ch_means = torch.tensor([0.43216, 0.394666, 0.37645], dtype=torch.float32)
    ch_std = torch.tensor([0.22803, 0.22145, 0.216989], dtype=torch.float32)
    plt.figure()
#    f, axarr = plt.subplots(4,4)
    for i in range(inputs.shape[0]):
        t = inputs[i]#*255
        t = t.permute(1, 2, 0)
#        t[:,:,[0, 2]] = t[:, :, [2, 0]]
#        axarr[i][j].imshow(np.array(t, dtype=np.uint8))
        if normalize:
            t = (t*ch_std[None,None,:] + ch_means[None,None,:])
        plt.imshow(np.array(t*255, dtype=np.uint8))
        plt.savefig("visualize_samps{}.png".format(i)) #, bbox_inches='tight')

def train_model(model, dataloaders, criterion, optimizer, scheduler, labs_keys, 
                labs_values, num_epochs=25):
    since = time.time()
#    vis_samples(dataloaders['test'].dataset, True)
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
            running_corrects = 0

            # Iterate over data.
            for bno, (inputs, vid_path, stroke, labels) in enumerate(dataloaders[phase]):
                # inputs of shape BATCH x SEQ_LEN x FEATURE_DIM
                labels = attn_utils.get_batch_labels(vid_path, stroke, labs_keys, labs_values, 1)# inputs.size(1))
                # Extract spatio-temporal features from clip using 3D ResNet (For SL >= 16)
                inputs = inputs.permute(0, 2, 1, 3, 4).float()
#                inputs[:, [0, 2], ...] = inputs[:, [2, 0], ...]       # convert RGB to BGR for C3D pretrained
#                inputs = inputs.permute(0, 4, 1, 2, 3).float()
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                loss = 0
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
                    logits = model(inputs)
                    probs = F.softmax(logits, dim=1)
                    loss = criterion(probs, labels)
                    _, preds = torch.max(probs, 1)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
#                print("Iter : {} / {} :: Running Loss : {}".format(bno, 
#                      len(dataloaders[phase]), running_loss))
                running_corrects += torch.sum(preds == labels.data)
                
#                print("Batch No : {} / {}".format(bno, len(dataloaders[phase])))
                if (bno+1) % 150 == 0:
                    break
                    
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / (bno+1)  #len(dataloaders[phase].dataset) 
            epoch_acc = running_corrects.double() / ((bno+1)*inputs.size(2))   #len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} :: Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc >= best_acc:
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
                

def predict(model, dataloaders, labs_keys, labs_values, phase="val"):
    assert phase == "val" or phase=="test", "Incorrect Phase."
    model = model.eval()
    gt_list, pred_list, stroke_ids  = [], [], []
    # Iterate over data.
    for bno, (inputs, vid_path, stroke, labels) in enumerate(dataloaders[phase]):
        # inputs of shape BATCH x SEQ_LEN x FEATURE_DIM
        labels = attn_utils.get_batch_labels(vid_path, stroke, labs_keys, labs_values, 1) #inputs.size(1))
        # Extract spatio-temporal features from clip using 3D ResNet (For SL >= 16)
        inputs = inputs.permute(0, 2, 1, 3, 4).float()
        inputs[:, [0, 2], ...] = inputs[:, [2, 0], ...]
#        inputs = inputs.permute(0, 4, 1, 2, 3).float()
        inputs = inputs.to(device)
        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            
            probs = model(inputs)
            gt_list.append(labels.tolist())
            pred_list.append((torch.max(probs, 1)[1]).tolist())
            for i, vid in enumerate(vid_path):
                stroke_ids.extend([vid+"_"+str(stroke[0][i].item())+"_"+str(stroke[1][i].item())] * 1) #inputs.size(2))
        # statistics
#        running_loss += loss.item()
#                print("Iter : {} :: Running Loss : {}".format(bno, running_loss))
#                running_corrects += torch.sum(preds == labels.data)
        
        print("Batch No : {} / {}".format(bno, len(dataloaders[phase])))
#        if (bno+1) % 20 == 0:
#            break
#    epoch_loss = running_loss #/ len(dataloaders[phase].dataset)
#            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
    confusion_mat = np.zeros((model.fc8.out_features, model.fc8.out_features))
    gt_list = [g for batch_list in gt_list for g in batch_list]
    pred_list = [p for batch_list in pred_list for p in batch_list]
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
    if not os.path.isdir(base_name):
        os.makedirs(base_name)
    seed = 1234
    attn_utils.seed_everything(seed)
    ###########################################################################
    # Read the strokes 
    # Divide the highlight dataset files into training, validation and test sets
    train_lst, val_lst, test_lst = autoenc_utils.split_dataset_files(DATASET)
    print("No. of training videos : {}".format(len(train_lst)))
        
    ###########################################################################
    # Create a Dataset    
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.ToPILClip(),
                                           videotransforms.Resize((112, 112)),
                                           videotransforms.ToTensor(),
                                           videotransforms.Normalize(),
#                                           videotransforms.ScaledNormMinMax(),
    ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224), 
                                          videotransforms.ToPILClip(),
                                          videotransforms.Resize((112, 112)),
                                          videotransforms.ToTensor(),
                                          videotransforms.Normalize(),
#                                          videotransforms.ScaledNormMinMax(),
                                         ])
    train_dataset = CricketStrokesDataset(train_lst, DATASET, LABELS, CLASS_IDS, 
                                         frames_per_clip=SEQ_SIZE, step_between_clips=STEP, 
                                         train=True, framewiseTransform=False, 
                                         transform=train_transforms)
    val_dataset = CricketStrokesDataset(val_lst, DATASET, LABELS, CLASS_IDS, 
                                        frames_per_clip=SEQ_SIZE, step_between_clips=STEP, 
                                        train=False, framewiseTransform=False, 
                                        transform=test_transforms)
    
    ###########################################################################
    
    labs_keys, labs_values = attn_utils.get_cluster_labels(ANNOTATION_FILE)
    
    num_classes = len(list(set(labs_values)))
    
    # created weighted Sampler for class imbalance
    if not os.path.isfile(os.path.join(base_name, "weights_c"+str(num_classes)+"_"+str(len(train_dataset))+".pkl")):
        samples_weight = attn_utils.get_sample_weights(train_dataset, labs_keys, labs_values, 
                                                       train_lst)
        with open(os.path.join(base_name, "weights_c"+str(num_classes)+"_"+str(len(train_dataset))+".pkl"), "wb") as fp:
            pickle.dump(samples_weight, fp)
    with open(os.path.join(base_name, "weights_c"+str(num_classes)+"_"+str(len(train_dataset))+".pkl"), "rb") as fp:
        samples_weight = pickle.load(fp)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, 
                              sampler=sampler, worker_init_fn=np.random.seed(12))
    
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    data_loaders = {"train": train_loader, "test": val_loader}
    
    ###########################################################################    
    # load model and set loss function    
    model = c3d.C3D()
    model.load_state_dict(torch.load("../localization_rnn/"+wts_path))
    for param in model.parameters():
        param.requires_grad = False
    # reset the last layer (default requires_grad is True)
    model.fc8 = nn.Linear(4096, num_classes)
    model.fc7 = nn.Linear(4096, 4096)
    model.fc6 = nn.Linear(8192, 4096)
    model.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
#    for ft in model.parameters():
#        ft.requires_grad = False
    model = model.to(device)
    
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
#    criterion = nn.MSELoss()
    
#    # Layers to finetune. Last layer should be displayed
    print("Params to learn:")
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t {}".format(name))
    
    # Observe that all parameters are being optimized
#    optimizer_ft = torch.optim.Adam(params_to_update, lr=0.01)
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    lr_scheduler = StepLR(optimizer_ft, step_size=15, gamma=0.1)
    
    ###########################################################################
    # Training the model
    start = time.time()
    
    model = train_model(model, data_loaders, criterion, optimizer_ft, lr_scheduler, 
                        labs_keys, labs_values, num_epochs=N_EPOCHS)
        
    end = time.time()
    
    # save the best performing model
    attn_utils.save_model_checkpoint(base_name, model, N_EPOCHS, "SGD_c3dConv5bFC678_Iter150_c8")
    # Load model checkpoints
    model = attn_utils.load_weights(base_name, model, N_EPOCHS, "SGD_c3dConv5bFC678_Iter150_c8")
    
    print("Total Execution time for {} epoch : {}".format(N_EPOCHS, (end-start)))

#    ###########################################################################    
    
    print("Predicting ...")
    acc = predict(model, data_loaders, labs_keys, labs_values, phase='test')
    
    print("#Parameters : {} ".format(autoenc_utils.count_parameters(model)))
    
    return model

if __name__ == '__main__':
    # Local Paths
    LABELS = "/home/arpan/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
    DATASET = "/home/arpan/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
    CLASS_IDS = "/home/arpan/VisionWorkspace/Cricket/cluster_strokes/configs/Class Index_Strokes.txt"
#    ANNOTATION_FILE = "/home/arpan/VisionWorkspace/Cricket/CricketStrokeLocalizationBOVW/shots_classes.txt"
    ANNOTATION_FILE = "/home/arpan/VisionWorkspace/Cricket/stroke_recognition/config/stroke_types_classes.txt"    

    SEQ_SIZE = 16
    STEP = 4
    BATCH_SIZE = 16
    N_EPOCHS = 30
    
    attn_utils.seed_everything(1234)
    
    model =  main(DATASET, LABELS, CLASS_IDS, BATCH_SIZE, ANNOTATION_FILE, 
                  SEQ_SIZE, STEP, nstrokes=-1, N_EPOCHS=N_EPOCHS, 
                  base_name=base_path)

