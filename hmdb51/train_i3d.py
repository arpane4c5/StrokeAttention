import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse
sys.path.insert(0, '../../../pytorch-i3d')
sys.path.insert(0, '../../cluster_strokes')
sys.path.insert(0, '../../cluster_strokes/lib')

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow', default='rgb')
parser.add_argument('-save_model', type=str, default='.')
parser.add_argument('-root', type=str, default='.')

args = parser.parse_args()

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import WeightedRandomSampler

import datasets.videotransforms as T
#import videotransforms as T
import torchvision
from torchvision import transforms
import numpy as np
import time
import random
import pickle

from pytorch_i3d import InceptionI3d
from dataset_hmdb import HMDB51
from collections import Counter
#from charades_dataset import Charades as Dataset

device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
LABELS = "/home/arpan/VisionWorkspace/VideoData/hmdb51/train_test_splits"
DATASET = "/home/arpan/VisionWorkspace/VideoData/hmdb51/videos"
log_path = "logs/I3DFine"

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

def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    
def predict(model, dataloaders, seq, phase="val"):
    assert phase == "val" or phase=="test", "Incorrect Phase."
    model = model.eval()
    gt_list, pred_list, stroke_ids = [], [], []
#    count = [0.] * cluster_size
    # Iterate over data.
    for bno, (inputs, vid_path, start_pts, end_pts, labels) in enumerate(dataloaders[phase]):
        # inputs of shape BATCH x SEQ_LEN x FEATURE_DIM
        inputs = inputs.permute(0, 2, 1, 3, 4).float()
        inputs = inputs.to(device)
        labels = labels.to(device)
#        iter_counts = Counter(inp_emb.flatten().tolist())
#        for k,v in iter_counts.items():
#            count[k]+=v
        # forward
        with torch.set_grad_enabled(phase == 'train'):
            logits = model(inputs)
            probs = F.softmax(logits.squeeze(axis=2), dim=1)
            gt_list.append(labels.tolist())
            pred_list.append((torch.max(probs, 1)[1]).tolist())
            for i, vid in enumerate(vid_path):
                stroke_ids.extend([vid])
                
#            batch_size = inputs.size(0)
#            hidden = model.init_hidden(batch_size)
#            outputs, hidden = model(inputs, hidden)
#            gt_list.append(labels.tolist())
#            pred_list.append((torch.max(outputs, 1)[1]).tolist())
#            for i, vid in enumerate(vid_path):
#                stroke_ids.extend([vid] * 1)
    
    ###########################################################################
#    print("Clusters : ")
#    print(count)
    confusion_mat = np.zeros((model._num_classes, model._num_classes))
    gt_list = [g for batch_list in gt_list for g in batch_list]
    pred_list = [p for batch_list in pred_list for p in batch_list]
    
    predictions = {"gt": gt_list, "pred": pred_list}
    
    # Save prediction and ground truth labels
    with open(os.path.join(log_path, "preds_Seq"+str(seq)+".pkl"), "wb") as fp:
        pickle.dump(predictions, fp)
    with open(os.path.join(log_path, "preds_Seq"+str(seq)+".pkl"), "rb") as fp:
        predictions = pickle.load(fp)
    gt_list = predictions['gt']
    pred_list = predictions['pred']
    
    tm = 0
    prev_gt = stroke_ids[0]
    val_labels, pred_labels, vid_preds = [], [], []
    for i, pr in enumerate(pred_list):
        if prev_gt != stroke_ids[i]:
            # find max category predicted in pred_labels
            val_labels.append(gt_list[i-1])
            pred_labels.append(max(set(vid_preds), key = vid_preds.count))
            print("Preds {} : {} :: {}".format(tm+1, vid_preds, pred_labels[-1]))
            print("GT {} : {}".format(tm+1, gt_list[i-1]))
            tm+=1
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

def run(init_lr=0.1, max_steps=64e3, mode='rgb', root='', batch_size=32, save_model='i3dIter1k_'):
    
    num_epochs = 30
    seed_everything()
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    
    # setup dataset
    train_transforms = transforms.Compose([T.RandomCrop(224),
                                         T.ToPILClip(), 
                                         T.Resize((224, 224)),
#                                         T.RandomCrop(112), 
                                         T.ToTensor(), 
                                         T.Normalize(),
                                        #T.RandomHorizontalFlip(),\
                                        ])
    test_transforms = transforms.Compose([T.CenterCrop(224),
                                         T.ToPILClip(), 
                                         T.Resize((224, 224)),
#                                         T.RandomCrop(112), 
                                         T.ToTensor(), 
                                         T.Normalize(),
                                        #T.RandomHorizontalFlip(),\
                                        ])    
#    train_transforms = transforms.Compose([T.RandomCrop(224),
#                                           T.RandomHorizontalFlip(),
#    ])
#    test_transforms = transforms.Compose([T.CenterCrop(224)])

    dataset = HMDB51(DATASET, LABELS, 16, step_between_clips = 1, 
                     fold=1, train=True, transform=train_transforms)
#    samples_weight = get_hmdb_sample_weights(dataset)
#    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
#    dataset = Dataset(train_split, 'training', root, mode, train_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)#sampler=sampler, worker_init_fn=np.random.seed(12)) #shuffle=True) #, num_workers=36, pin_memory=True)
    val_dataset = HMDB51(DATASET, LABELS, 16, step_between_clips = 1, 
                     fold=1, train=False, transform=test_transforms)
#    val_dataset = Dataset(train_split, 'testing', root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False) #, num_workers=36, pin_memory=True)    

    dataloaders = {'train': dataloader, 'test': val_dataloader}
    datasets = {'train': dataset, 'test': val_dataset}

#    vis_samples(dataset, True)
    
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('/home/arpan/VisionWorkspace/pytorch-i3d/models/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('/home/arpan/VisionWorkspace/pytorch-i3d/models/rgb_imagenet.pt'))
    i3d.replace_logits(51)
    #i3d.load_state_dict(torch.load('/ssd/models/000920.pt'))
    i3d = i3d.to(device)
#    i3d = nn.DataParallel(i3d)

    lr = init_lr
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
#    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [10, 25]) # [300, 1000])
    # Decay LR by a factor of 0.1 every 7 epochs
    lr_sched = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
#    criterion = nn.CrossEntropyLoss()
    num_steps_per_update = 4 # accum gradient
    steps = 0
    # train it
    start = time.time()
#    print("No. of Iters / Epoch ; {}".format(len(dataloaders['train'])))
#    for epoch in range(num_epochs): #while steps < max_steps:
##        print( 'Step {}/{}'.format(steps, max_steps))
#        print('Epoch {}/{}'.format(epoch+1, num_epochs))
#        print('-' * 10)
#
#        # Each epoch has a training and validation phase
#        for phase in ['train', 'test']:
#            if phase == 'train':
#                i3d.train(True)
#            else:
#                i3d.train(False)  # Set model to evaluate mode
#                
#            tot_loss = 0.0
#            tot_loc_loss = 0.0
#            tot_cls_loss = 0.0
#            num_iter = 0
#            
#            running_corrects = 0
#            count = [0.] * 51
#            
#            # Iterate over data.
#            for bno, (inputs, vid_path, start_pts, end_pts, labels) in enumerate(dataloaders[phase]):
#                num_iter += 1
#                # wrap them in Variable
#                inputs = inputs.permute(0, 2, 1, 3, 4).float()      # for PIL and ToTensor
##                inputs = inputs.permute(0, 4, 1, 2, 3).float()      # for Raw Crops
#                inputs = inputs.to(device)
##                t = inputs.size(2)
#                labels = labels.to(device)
#
#                iter_counts = Counter(labels.tolist())
#                for k,v in iter_counts.items():
#                    count[k]+=v
#                    
#                optimizer.zero_grad()
#                
#                per_frame_logits = i3d(inputs)  # get B x N_CLASSES X 1
#                per_frame_logits = per_frame_logits.squeeze(2)
#                # upsample to input size
##                per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')
#
#                # compute localization loss
##                loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
##                tot_loc_loss += loc_loss.data[0]
#
#                # compute classification loss (with max-pooling along time B x C x T)
##                cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
##                tot_cls_loss += cls_loss.data[0]
#                cls_loss = F.cross_entropy(per_frame_logits, labels)
#
##                loss = (0.5*loc_loss + 0.5*cls_loss)/num_steps_per_update
#                loss = cls_loss     #/num_steps_per_update
#                tot_loss += loss.item()
##                loss.backward()
#                
##                print("{}  : bno : {}".format(phase, bno))
#                                    # backward + optimize only if in training phase
#                if phase == 'train':
#                    loss.backward()
#                    optimizer.step()
#                    
#                running_corrects += torch.sum(torch.max(per_frame_logits, 1)[1] == labels.data)
#
###                if num_iter == num_steps_per_update and phase == 'train':
##                if phase == 'train':
##                    steps += 1
##                    num_iter = 0
##                    optimizer.step()
##                    optimizer.zero_grad()
##                    lr_sched.step()
##                    if steps % 10 == 0:
##                        print('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss/(10*num_steps_per_update), tot_cls_loss/(10*num_steps_per_update), tot_loss/10))
##                        # save model
##                        torch.save(i3d.state_dict(), save_model+str(steps).zfill(6)+'.pt')
##                        tot_loss = tot_loc_loss = tot_cls_loss = 0.
##                if (bno + 1) % 10 == 0:
##                    print('{} : {}/{} Loss: {:.4f} Corrects: {:.4f}'.format(phase, 
##                          bno, len(dataloaders[phase]), tot_loc_loss, running_corrects))
#                if bno == 1000:
#                    break
#            if phase == 'train':
#                lr_sched.step()
#                print("Category Weights : {}".format(count))
#            epoch_loss = tot_loss / (16*(bno+1))  #len(dataloaders[phase].dataset)
#            epoch_acc = running_corrects.double() / (16*(bno+1)) #  len(dataloaders[phase].dataset)
#            print('{} Loss: {:.6f} Acc: {:.6f} LR: {}'.format(phase, epoch_loss, epoch_acc, 
#                  lr_sched.get_last_lr()[0]))
#            
##            if phase == 'val':
##                print('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss/num_iter, tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter) )
#                
#            if (epoch+1) % 10 == 0:
#                torch.save(i3d.state_dict(), os.path.join(log_path, save_model+str(epoch+1).zfill(6)+'.pt'))
                
    i3d.load_state_dict(torch.load(os.path.join(log_path, save_model+str(num_epochs).zfill(6)+'.pt')))
    
                
    end = time.time()
    print("Total Execution time for {} epoch : {}".format(num_epochs, (end-start)))
    
    ###########################################################################
    
    # Predictions
    
    predict(i3d, dataloaders, 16, 'test')
    


if __name__ == '__main__':
    # need to add argparse
    run()
