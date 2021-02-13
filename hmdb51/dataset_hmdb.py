#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 19:51:35 2020

@author: arpan
"""

import glob
import os
import pickle
import numpy as np

import torch
from torchvision import transforms
from torchvision.datasets.utils import list_dir
from torchvision.datasets.video_utils import VideoClips
#from utils.video_utils import VideoClips
from torchvision.datasets.vision import VisionDataset

#from .utils import list_dir
from torchvision.datasets.folder import make_dataset
#from .vision import VisionDataset


class HMDB51(VisionDataset):
    """
    HMDB51 <http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/>`_
    dataset.

    HMDB51 is an action recognition video dataset.
    This dataset consider every video as a collection of video clips of fixed size, specified
    by ``frames_per_clip``, where the step in frames between each clip is given by
    ``step_between_clips``.

    To give an example, for 2 videos with 10 and 15 frames respectively, if ``frames_per_clip=5``
    and ``step_between_clips=5``, the dataset size will be (2 + 3) = 5, where the first two
    elements will come from video 1, and the next three elements from video 2.
    Note that we drop clips which do not have exactly ``frames_per_clip`` elements, so not all
    frames in a video might be present.

    Internally, it uses a VideoClips object to handle clip creation.

    Args:
        root (string): Root directory of the HMDB51 Dataset.
        annotation_path (str): path to the folder containing the split files
        frames_per_clip (int): number of frames in a clip.
        step_between_clips (int): number of frames between each clip.
        fold (int, optional): which fold to use. Should be between 1 and 3.
        train (bool, optional): if ``True``, creates a dataset from the train split,
            otherwise from the ``test`` split.
        transform (callable, optional): A function/transform that  takes in a TxHxWxC video
            and returns a transformed version.

    Returns:
        video (Tensor[T, H, W, C]): the `T` video frames
        audio(Tensor[K, L]): the audio frames, where `K` is the number of channels
            and `L` is the number of points
        label (int): class of the video clip
    """

    data_url = "http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar"
    splits = {
        "url": "http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar",
        "md5": "15e67781e70dcfbdce2d7dbb9b3344b5"
    }

    def __init__(self, root, annotation_path, frames_per_clip, step_between_clips=1,
                 fold=1, train=True, framewiseTransform=False, transform=None):
        super(HMDB51, self).__init__(root)
        if not 1 <= fold <= 3:
            raise ValueError("fold should be between 1 and 3, got {}".format(fold))

        extensions = ('avi',)
        self.fold = fold
        self.train = train

        classes = list(sorted(list_dir(root)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        video_list = [x[0] for x in self.samples]
        video_clips = VideoClips(video_list, frames_per_clip, step_between_clips)
        self.indices = self._select_fold(video_list, annotation_path, fold, train)
        self.video_clips = video_clips.subset(self.indices)
        self.video_list = [video_list[i] for i in self.indices]
        self.framewiseTransform = framewiseTransform
        self.transform = transform

    def _select_fold(self, video_list, annotation_path, fold, train):
        target_tag = 1 if train else 2
        name = "*test_split{}.txt".format(fold)
        files = glob.glob(os.path.join(annotation_path, name))
        selected_files = []
        for f in files:
            with open(f, "r") as fid:
                data = fid.readlines()
                data = [x.strip().split(" ") for x in data]
                data = [x[0] for x in data if int(x[1]) == target_tag]
                selected_files.extend(data)
        selected_files = set(selected_files)
        indices = [i for i in range(len(video_list)) if os.path.basename(video_list[i]) in selected_files]
        return indices

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        
        video_idx, clip_idx = self.video_clips.get_clip_location(idx)
        video_path = self.video_clips.video_paths[video_idx]
        clip_pts = self.video_clips.clips[video_idx][clip_idx]
        start_pts = clip_pts[0].item()
        end_pts = clip_pts[-1].item()
        
        label_indx = self.indices[video_idx]
        label = self.samples[label_indx][1]
        
        if self.transform is not None:
            if self.framewiseTransform:
                if isinstance(self.transform, transforms.Compose):
                    # transform frame-wise (takes input as HxWxC)
                    video = torch.stack([self.transform(i) for i in video])
            else:   # clip level transform (takes input as TxHxWxC)
                video = self.transform(video)
            
        if isinstance(video, list):
            video = torch.stack(video)

#        if self.transform is not None:
#            video = self.transform(video)

        return video, video_path, start_pts, end_pts, label
    
    
class HMDB51FeatureSequenceDataset(VisionDataset):
    """
    `HMDB51 Feature Sequence Dataset for HMBD51 dataset.

    Args:
        root (string): Root directory of the HMDB51 Dataset.
        class_ids_path (str): path to the class IDs file
        frames_per_clip (int): number of frames in a clip.
        step_between_clips (int, optional): number of frames between each clip.
        train (bool, optional): if ``True``, creates a dataset from the train split,
            otherwise from the ``test`` split.
        framewiseTransform (bool, optional): If the transform has to be applied
            to each frame separately and resulting frames are to be stacked.
        transform (callable, optional): A function/transform that takes in a HxWxC video
            and returns a transformed version (CxHxW) for a frame. Additional dimension
            for a video level transform

    Returns:
        video (Tensor[T, H, W, C]): the `T` video frames (without transform)
        audio(Tensor[K, L]): the audio frames, where `K` is the number of channels
            and `L` is the number of points
        label (int): class of the video clip
    """
    
    def __init__(self, feat_path, root, annotation_path, frames_per_clip, 
                 extracted_frames_per_clip=2, step_between_clips=1,
                 fold=1, train=True, transform=None):
        super(HMDB51FeatureSequenceDataset, self).__init__(root)
        if not 1 <= fold <= 3:
            raise ValueError("fold should be between 1 and 3, got {}".format(fold))

        extensions = ('avi',)
        self.fold = fold
        self.train = train
        self.frames_per_clip = frames_per_clip
        self.extracted_frames_per_clip = extracted_frames_per_clip
        self.step_between_clips = step_between_clips
        with open(feat_path, "rb") as fp:
            self.features = pickle.load(fp)

        classes = list(sorted(list_dir(root)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        video_list = [x[0] for x in self.samples]
        video_clips = VideoClips(video_list, frames_per_clip, step_between_clips)
        self.indices = self._select_fold(video_list, annotation_path, fold, train)
        self.video_clips = video_clips.subset(self.indices)
        self.video_list = [video_list[i] for i in self.indices]
        self.transform = transform
        
        
    def _select_fold(self, video_list, annotation_path, fold, train):
        target_tag = 1 if train else 2
        name = "*test_split{}.txt".format(fold)
        files = glob.glob(os.path.join(annotation_path, name))
        selected_files = []
        for f in files:
            with open(f, "r") as fid:
                data = fid.readlines()
                data = [x.strip().split(" ") for x in data]
                data = [x[0] for x in data if int(x[1]) == target_tag]
                selected_files.extend(data)
        selected_files = set(selected_files)
        indices = [i for i in range(len(video_list)) if os.path.basename(video_list[i]) in selected_files]
        return indices

    def __len__(self):
        return self.video_clips.num_clips()        

    def __getitem__(self, idx):
        
        video_idx, clip_idx = self.video_clips.get_clip_location(idx)
        video_path = self.video_clips.video_paths[video_idx]
        clip_pts = self.video_clips.clips[video_idx][clip_idx]
        start_pts = clip_pts[0].item()
#        end_pts = clip_pts[-1].item()
        
        # form feature key 
        key = video_path  #.rsplit('/', 1)[1].rsplit('.', 1)[0]+'_'+str(stroke_tuple[0])+'_'+str(stroke_tuple[1])
                
        vid_feats = self.features[key]
        
        vid_feats[np.isnan(vid_feats)] = 0
        vid_feats[np.isinf(vid_feats)] = 0
        
#        seq_len = self.frames_per_clip - self.extracted_frames_per_clip + 1
#        if start_pts+seq_len > vid_feats.shape[0]:
#            start_pts -= ((start_pts + seq_len) - vid_feats.shape[0])
#        sequence = vid_feats[start_pts:(start_pts+seq_len), :]
        st_idx = start_pts // self.step_between_clips
        seq_len = 1 + (self.frames_per_clip - self.extracted_frames_per_clip) // self.step_between_clips
        #seq_len = (self.frames_per_clip // self.extracted_frames_per_clip)
        
        if st_idx+seq_len > vid_feats.shape[0]:
            st_idx -= ((st_idx + seq_len) - vid_feats.shape[0])
        # retrieve the sequence of vectors from the stroke sequences
        sequence = vid_feats[st_idx:(st_idx+seq_len), :]
        
#        if self.train:
        label_indx = self.indices[video_idx]
        label = self.samples[label_indx][1]
#        else:
#            label = -1

        return sequence, video_path, start_pts, label
