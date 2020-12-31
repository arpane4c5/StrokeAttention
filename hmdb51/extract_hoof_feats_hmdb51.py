#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 02:57:52 2020

@author: arpan

@Description: Extract HOOF features from all files
"""

import os
import cv2
import sys
import pickle
import numpy as np
import dataset_hmdb as hmdb
import warnings

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore")

# "of_fold1_feats_grid20.pkl","of_fold1_test_feats_grid20.pkl"
# "unhoof_fold1_feats_bins20.pkl","unhoof_fold1_test_feats_bins20.pkl"
feat, feat_test = "hmdb_2dcnn_featsF1_train.pkl", "hmdb_2dcnn_featsF1_test.pkl"
# "of_fold1_snames_grid20.pkl", "of_fold1_test_snames_grid20.pkl"
# "unhoof_fold1_snames_bins20.pkl", "unhoof_fold1_test_snames_bins20.pkl"
snames, snames_test = "hmdb_2dcnn_snamesF1_train.pkl", "hmdb_2dcnn_snamesF1_test.pkl"
# feats/ofMagAng_grid20" ; 
feat_path = "feats/ofMagAng_grid20"


def extract_of_features(feat_path, video_list, fold=1, isTrain=True):
    
    nbins, mth, grid = 20, 0.5, None   # grid should be None for extracting HOOF
    if isTrain:
        ft_prefix='unhoof_fold'+str(fold)
    else:
        ft_prefix='unhoof_fold'+str(fold)+'_test'
    if not os.path.isfile(os.path.join(feat_path, ft_prefix+"_feats_bins"+str(nbins)+".pkl")):
        if not os.path.exists(feat_path):
            os.makedirs(feat_path)
        if isTrain:
            print("Fold {} : Training extraction ... ".format(fold))
        else:
            print("Fold {} : Test extraction ... ".format(fold))
            
        #    # Extract Grid OF / HOOF features {mth = 2, and vary nbins}
        features, strokes_name_id = extract_feats(video_list, nbins, mth, False, grid)
        with open(os.path.join(feat_path, ft_prefix+"_feats_bins"+str(nbins)+".pkl"), "wb") as fp:
            pickle.dump(features, fp)
        with open(os.path.join(feat_path, ft_prefix+"_snames_bins"+str(nbins)+".pkl"), "wb") as fp:
            pickle.dump(strokes_name_id, fp)
    else:
        print("Already extracted : {}".format(os.path.join(feat_path, ft_prefix+"_feats_bins"+str(nbins)+".pkl")))


def extract_feats(video_list, nbins, mag_thresh=2, density=False, grid_size=None, crop=240):
    """
    Function to iterate on all the training videos and extract the relevant features.
    video_list: list of str
        
    mag_thresh: float
        pixels with >mag_thresh will be considered significant and used for clustering
    nbins: int
        No. of bins in which the angles have to be divided.
    grid_size : int or None
        If it is None, then extract HOOF features using nbins and mag_thresh, else 
        extract grid features
    crop : int
        center crop size of 240 for HMDB51, as videos have height of 240 and width > 240
    """
    strokes_name_id = []
    all_feats = {}
    bins = np.linspace(0, 2*np.pi, (nbins+1))
#    indices = [103, 104, 105, 106, 1798, 1799, 1800]  # video indices with FrameWidth < 240
#    video_list = [video_list[i] for i in indices]
    for i, v_file in enumerate(video_list):
        print('-'*60)
        print("{} / {} :: {} ".format((i+1), len(video_list), v_file.rsplit('/', 3)[-2:]))
        
        k = v_file
        strokes_name_id.append(k)
        # Extract the video features
        if grid_size is None:
            all_feats[k] = extract_flow_angles(v_file, bins, mag_thresh, density, crop)
        else:
            all_feats[k] = extract_flow_grid(v_file, grid_size, crop)
        print("Shape : {}".format(all_feats[k].shape))
#        if i == 10:
#            break
    return all_feats, strokes_name_id


def extract_feats_par(video_list, nbins, mag_thresh=2, density=False, grid_size=None, crop=240, njobs=1):
    """
    Function to iterate on all the training videos and extract the relevant features.
    video_list: list of str
        
    mag_thresh: float
        pixels with >mag_thresh will be considered significant and used for clustering
    nbins: int
        No. of bins in which the angles have to be divided.
    grid_size : int or None
        If it is None, then extract HOOF features using nbins and mag_thresh, else 
        extract grid features
    crop : int
        center crop size of 240 for HMDB51, as videos have height of 240 and width > 240
    """
    strokes_name_id = []
    all_feats = {}
    bins = np.linspace(0, 2*np.pi, (nbins+1))
#    indices = [103, 104, 105, 106, 1798, 1799, 1800]  # video indices with FrameWidth < 240
#    video_list = [video_list[i] for i in indices]
    for i, v_file in enumerate(video_list):
        print('-'*60)
        print("{} / {} :: {} ".format((i+1), len(video_list), v_file.rsplit('/', 3)[-2:]))
        
        k = v_file
        strokes_name_id.append(k)
        # Extract the video features
        if grid_size is None:
            all_feats[k] = extract_flow_angles(v_file, bins, mag_thresh, density, crop)
        else:
            all_feats[k] = extract_flow_grid(v_file, grid_size, crop)
        print("Shape : {}".format(all_feats[k].shape))
#        if i == 10:
#            break
    return all_feats, strokes_name_id


def extract_flow_angles(vidFile, hist_bins, mag_thresh, density=False, crop=240):
    '''
    Extract optical flow maps from video vidFile for all the frames and put the angles with >mag_threshold in different 
    bins. The bins vector is the feature representation for the stroke. 
    Use only the strokes given by list of tuples frame_indx.
    Parameters:
    ------
    vidFile: str
        complete path to a video
    start: int
        starting frame number
    end: int
        ending frame number
    hist_bins: 1d np array 
        bin divisions (boundary values). Used np.linspace(0, 2*PI, 11) for 10 bins
    mag_thresh: int
        minimum size of the magnitude vectors that are considered (no. of pixels shifted in consecutive frames of OF)
    
    '''
    cap = cv2.VideoCapture(vidFile)
    if not cap.isOpened():
        print("Capture object not opened. Aborting !!")
        sys.exit(0)
    ret = True
    start = 0
    end = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stroke_features = []
    prvs, next_ = None, None
    m, n = start, end
    #print("stroke {} ".format((m, n)))
    sum_norm_mag_ang = np.zeros((len(hist_bins)-1))  # for optical flow maxFrames - 1 size
    frameNo = m
    while ret: # and frameNo < n:
#        cap.set(cv2.CAP_PROP_POS_FRAMES, frameNo)
        ret, frame1 = cap.read()
        if not ret:
#            print("Frame not read. Aborting !!")
            break
        # resize
        if frame1.shape[0] < crop or frame1.shape[1] < crop:
            frame1 = cv2.resize(frame1, (crop, crop))        
        if (frameNo-m) == 0:    # first frame condition
            # resize and then convert to grayscale
            if crop is not None:
                frame1 = center_crop(frame1, crop)
            prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            #prvs = scale_and_crop(prvs, scale)
            frameNo +=1
            continue
            
        if crop is not None:
            frame1 = center_crop(frame1, crop)
        # resize and then convert to grayscale
        next_ = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(prvs, next_, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        
        #print("Mag > 5 = {}".format(np.sum(mag>THRESH)))
        pixAboveThresh = np.sum(mag>mag_thresh)
        #use weights=mag[mag>THRESH] to be weighted with magnitudes
        #returns a tuple of (histogram, bin_boundaries)
        ang_hist = np.histogram(ang[mag>mag_thresh], bins=hist_bins, density=density)
        stroke_features.append(ang_hist[0])
        #sum_norm_mag_ang +=ang_hist[0]
#            if not pixAboveThresh==0:
#                sum_norm_mag_ang[frameNo-m-1] = np.sum(mag[mag > THRESH])/pixAboveThresh
#                sum_norm_mag_ang[(maxFrames-1)+frameNo-m-1] = np.sum(ang[mag > THRESH])/pixAboveThresh
        frameNo+=1
        prvs = next_
        #stroke_features.append(sum_norm_mag_ang/(n-m+1))
    cap.release()
    #cv2.destroyAllWindows()
    stroke_features = np.array(stroke_features)
    #Normalize row - wise
    #stroke_features = stroke_features/(1+stroke_features.sum(axis=1)[:, None])
    return stroke_features

def center_crop(img, size):
    h, w, c = img.shape
    th, tw = size, size
    i = int(np.round((h - th) / 2.))
    j = int(np.round((w - tw) / 2.))

    return img[i:i+th, j:j+tw, :]

def extract_flow_grid(vidFile, grid_size, crop):
    '''
    Extract optical flow maps from video vidFile starting from start frame number
    to end frame no. The grid based features are flattened and appended.
    
    Parameters:
    ------
    vidFile: str
        complete path to a video
    grid_size: int
        grid size for sampling at intersection points of 2D flow.
    
    Returns:
    ------
    np.array 2D with N x (360/G * 640/G) where G is grid size
    '''
    cap = cv2.VideoCapture(vidFile)
    if not cap.isOpened():
        print("Capture object not opened. Aborting !!")
        sys.exit(0)
    ret = True
    start = 0
    end = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stroke_features = []
    prvs, next_ = None, None
    m, n = start, end
    #print("stroke {} ".format((m, n)))
    #sum_norm_mag_ang = np.zeros((len(hist_bins)-1))  # for optical flow maxFrames - 1 size
    frameNo = m
    while ret: # and frameNo <= n:
#        cap.set(cv2.CAP_PROP_POS_FRAMES, frameNo)
        ret, frame1 = cap.read()
        if not ret:
#            print("Frame not read. Aborting !!")
            break
        # resize
        if frame1.shape[0] < crop or frame1.shape[1] < crop:
            frame1 = cv2.resize(frame1, (crop, crop))
        if (frameNo-m) == 0:    # first frame condition
            
            # resize and then convert to grayscale
            #cv2.imwrite(os.path.join(flow_numpy_path, str(frameNo)+".png"), frame1)
            if crop is not None:
                frame1 = center_crop(frame1, crop)
            prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            #prvs = scale_and_crop(prvs, scale)
            frameNo +=1
            continue
            
        if crop is not None:
            frame1 = center_crop(frame1, crop)
        next_ = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(prvs, next_, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        
        # stack sliced arrays along the first axis (2, 12, 16)
        sliced_flow = np.stack(( mag[::grid_size, ::grid_size], \
                                ang[::grid_size, ::grid_size]), axis=0)
                
#        stroke_features.append(sliced_flow[1, ...].ravel())     # Only angles
        #feature = np.array(feature)
        stroke_features.append(sliced_flow.ravel())     # Both magnitude and angle
        
        frameNo+=1
        prvs = next_
        
    cap.release()
    #cv2.destroyAllWindows()
    stroke_features = np.array(stroke_features)
    #Normalize row - wise
    #stroke_features = stroke_features/(1+stroke_features.sum(axis=1)[:, None])
    return stroke_features


if __name__ == '__main__':
    
    LABELS = "/home/arpan/VisionWorkspace/VideoData/hmdb51/train_test_splits"
    DATASET = "/home/arpan/VisionWorkspace/VideoData/hmdb51/videos"

    SEQ_SIZE = 16
    STEP = 16
    hmdb51_train = hmdb.HMDB51(DATASET, LABELS, SEQ_SIZE, step_between_clips = STEP, 
                               fold=1, train=True, transform=None)

    hmdb51_test = hmdb.HMDB51(DATASET, LABELS, SEQ_SIZE, step_between_clips = STEP, 
                              fold=1, train=False, transform=None)
    
#    tmp = hmdb51_train.__getitem__(5)

    # Extracting training features
    extract_of_features(feat_path, hmdb51_train.video_list, hmdb51_train.fold, hmdb51_train.train)
    # Extracting testing features
    extract_of_features(feat_path, hmdb51_test.video_list, hmdb51_test.fold, hmdb51_test.train)    