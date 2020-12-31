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

from pytorch_i3d import InceptionI3d
from dataset_hmdb import HMDB51
from collections import Counter
from main_bovgru import get_hmdb_sample_weights 
#from charades_dataset import Charades as Dataset

LABELS = "/home/arpan/VisionWorkspace/VideoData/hmdb51/train_test_splits"
DATASET = "/home/arpan/VisionWorkspace/VideoData/hmdb51/videos"

def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

def run(init_lr=0.1, max_steps=64e3, mode='rgb', root='', batch_size=2, save_model='i3d_'):
    
    num_epochs = 30
    seed_everything()
    
    # setup dataset
#    train_transforms = transforms.Compose([T.CenterCrop(224),
#                                         T.ToPILClip(), 
#                                         T.Resize((224, 224)),
##                                         T.RandomCrop(112), 
#                                         T.ToHMDBTensor(), 
#                                         T.Normalize(),
#                                        #T.RandomHorizontalFlip(),\
#                                        ])
#    test_transforms = transforms.Compose([T.CenterCrop(224),
#                                         T.ToPILClip(), 
#                                         T.Resize((224, 224)),
##                                         T.RandomCrop(112), 
#                                         T.ToHMDBTensor(), 
#                                         T.Normalize(),
#                                        #T.RandomHorizontalFlip(),\
#                                        ])    
    train_transforms = transforms.Compose([T.RandomCrop(224),
                                           T.RandomHorizontalFlip(),
    ])
    test_transforms = transforms.Compose([T.CenterCrop(224)])

    dataset = HMDB51(DATASET, LABELS, 16, step_between_clips = 1, 
                     fold=1, train=True, transform=train_transforms)
    samples_weight = get_hmdb_sample_weights(dataset)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
#    dataset = Dataset(train_split, 'training', root, mode, train_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, worker_init_fn=np.random.seed(12)) #shuffle=True) #, num_workers=36, pin_memory=True)
    val_dataset = HMDB51(DATASET, LABELS, 16, step_between_clips = 1, 
                     fold=1, train=False, transform=test_transforms)
#    val_dataset = Dataset(train_split, 'testing', root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False) #, num_workers=36, pin_memory=True)    

    dataloaders = {'train': dataloader, 'test': val_dataloader}
    datasets = {'train': dataset, 'test': val_dataset}

    
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('/home/arpan/VisionWorkspace/pytorch-i3d/models/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('/home/arpan/VisionWorkspace/pytorch-i3d/models/rgb_imagenet.pt'))
    i3d.replace_logits(51)
    #i3d.load_state_dict(torch.load('/ssd/models/000920.pt'))
    i3d.cuda()
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
    for epoch in range(num_epochs): #while steps < max_steps:
#        print( 'Step {}/{}'.format(steps, max_steps))
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)  # Set model to evaluate mode
                
            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            
            running_corrects = 0
            count = [0.] * 51
            
            # Iterate over data.
            for bno, data in enumerate(dataloaders[phase]):
                num_iter += 1
                # get the inputs
                inputs, vid_path, start_pts, end_pts, labels = data

                # wrap them in Variable
#                inputs = inputs.permute(0, 2, 1, 3, 4).float()      # for PIL and ToTensor
                inputs = inputs.permute(0, 4, 1, 2, 3).float()      # for Raw Crops
                inputs = inputs.cuda()
#                t = inputs.size(2)
                labels = labels.cuda()

                iter_counts = Counter(labels.tolist())
                for k,v in iter_counts.items():
                    count[k]+=v
                    
                optimizer.zero_grad()
                
                per_frame_logits = i3d(inputs)  # get B x N_CLASSES X 1
                per_frame_logits = per_frame_logits.squeeze(2)
                # upsample to input size
#                per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')

                # compute localization loss
#                loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
#                tot_loc_loss += loc_loss.data[0]

                # compute classification loss (with max-pooling along time B x C x T)
#                cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
#                tot_cls_loss += cls_loss.data[0]
                cls_loss = F.cross_entropy(per_frame_logits, labels)

#                loss = (0.5*loc_loss + 0.5*cls_loss)/num_steps_per_update
                loss = cls_loss     #/num_steps_per_update
                tot_loss += loss.item()
#                loss.backward()
                
#                print("{}  : bno : {}".format(phase, bno))
                                    # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                running_corrects += torch.sum(torch.max(per_frame_logits, 1)[1] == labels.data)

##                if num_iter == num_steps_per_update and phase == 'train':
#                if phase == 'train':
#                    steps += 1
#                    num_iter = 0
#                    optimizer.step()
#                    optimizer.zero_grad()
#                    lr_sched.step()
#                    if steps % 10 == 0:
#                        print('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss/(10*num_steps_per_update), tot_cls_loss/(10*num_steps_per_update), tot_loss/10))
#                        # save model
#                        torch.save(i3d.state_dict(), save_model+str(steps).zfill(6)+'.pt')
#                        tot_loss = tot_loc_loss = tot_cls_loss = 0.
#                if bno == 24:
#                    print()
                if bno == 3000:
                    break
            if phase == 'train':
                lr_sched.step()
                print("Category Weights : {}".format(count))
            epoch_loss = tot_loss / bno #len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / bno #len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
#            if phase == 'val':
#                print('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss/num_iter, tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter) )
                
            if (epoch+1) % 10 == 0:
                torch.save(i3d.state_dict(), save_model+str(epoch+1).zfill(6)+'.pt')
                
    end = time.time()
    print("Total Execution time for {} epoch : {}".format(num_epochs, (end-start)))
    


if __name__ == '__main__':
    # need to add argparse
    run()
