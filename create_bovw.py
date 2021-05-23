#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 13:05:38 2020

@author: arpan

@Description: Clustering and creation of BOVW dataframe
"""
import numpy as np
from scipy.cluster.vq import vq
import pandas as pd
import pickle
import os
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.preprocessing import normalize
import cv2

def get_frame(datasetpath, video_key, position, offset=2):
    vid = video_key.rsplit('_', 2)
    vid_name = vid[0]+'.avi'
    st, _ = int(vid[1]), int(vid[2])
    cap = cv2.VideoCapture(os.path.join(datasetpath, vid_name))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, st+position+offset-1)
    _, img = cap.read()
    cap.release()
    return img

def get_flow(datasetpath, video_key, position, offset=2):
    vid = video_key.rsplit('_', 2)
    vid_name = vid[0]+'.avi'
    st, _ = int(vid[1]), int(vid[2])
    cap = cv2.VideoCapture(os.path.join(datasetpath, vid_name))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, st+position+offset-1)
    _, prev_ = cap.read()
    prev = cv2.cvtColor(prev_, cv2.COLOR_BGR2GRAY)
    _, next_ = cap.read()
    next_ = cv2.cvtColor(next_, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev, next_, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    cap.release()
    return prev_, flow

# draw the OF field on image, with grids, decrease step for finer grid
def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    x, y = np.array(x, dtype=np.uint32), np.array(y, dtype=np.uint32)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
#    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(img, (x1, y1), 1, (0, 255, 0), -1)
    return img

def draw_flow_bgr(flow, grid=20):
    h, w = 360//grid, 640//grid       # Visualize for 360 * 640 (gridsize 20)
#    color_width = 5
    hsv = np.zeros((360, 640, 3), dtype=np.uint8)
    ang = flow.reshape((h, w))      # angles at grid places
#    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])    
    hue = ang*180/np.pi/2
    for x in range(grid-2):
        for y in range(grid-2):
            hsv[x::grid, y::grid, 1] = 255
            hsv[x::grid, y::grid, 0] = hue
            hsv[x::grid, y::grid, 2] = 255    # max magnitude
##    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def write_cluster_samples(features, onehot_feats, cl_vnames, cluster_idx, 
                          offset, ds_path, base, cno):
    # create path
    dest_dir = os.path.join(base, "cl_"+str(cno)+"_"+str(cluster_idx))
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    count, max_frms = 0, 80
    cl_feats = []
    # Make bow vectors for all videos.
    for video_index, video in enumerate(cl_vnames):
        # Get the starting and ending stroke frame positions        
        m, n = video.rsplit('_', 2)[1:]
        m, n = int(m), int(n)
        
        stroke_words = onehot_feats[video]
        # get rows indexes where cluster cl_no is assigned to frame no.
        frm_pos = np.where(stroke_words[:, cluster_idx] > 0)[0]
        for frm_no in frm_pos:
#            ###################################################################
#            # For visualizing means of clusters (comment everything else in this loop)
#            cl_feats.append(features[video][frm_no])
#            ###################################################################
            # retrieve the frame positions 
#            img = get_frame(ds_path, video, frm_no, offset)
#            cv2.imwrite(os.path.join(dest_dir, video+"_"+str(frm_no)+".png"), img)
            img, flow = get_flow(ds_path, video, frm_no, offset)
            vis = draw_flow(img, flow, 20)
            cv2.imwrite(os.path.join(dest_dir, video+"_"+str(frm_no)+".png"), vis)

            count+=1
            if count > max_frms:
                break
#            cv2.imshow("Stroke {}_{} : F{}".format(m, n, frm_no), img)
#            direction = waitTillEscPressed()
        if count > max_frms:
            break
    ###################################################################
#    # For visualizing means of clusters (comment everything else in this loop)            
#    # calculate mean of the flow vectors and visualize in BGR 
#    bgr = draw_flow_bgr(np.mean(np.vstack(cl_feats), axis=0))
#    cv2.imwrite(os.path.join(base, "cl_"+str(cno)+"_"+str(cluster_idx)+"_nsamples_"+str(len(cl_feats))+".png"), bgr)

def vis_clusters(features, onehot_feats, strokes_name_id, offset, ds_path, base=""):
    '''
    searches the data partition for clusters anld writes frames corresponding to 
    the given cluster number.
    
    Parameters:
    ------
    features: dict
        {'video_id1':np.array((N, d)), ...}  , where d is vec dimension
    strokes_name_id: list of str
        key values of features, eg. video-name_34_242 
    km_model: KMeans / GaussianMixture obj
        Learned KMeans / GMM model with nClusters / nComponents
        
    '''
    top_dense = 1000
    
    # find total words in dataset
    words = []
    for indx, video in enumerate(strokes_name_id):
        stroke_words = onehot_feats[video]
        words.append(np.sum(stroke_words, axis=0))
    
    # find densest clusters
    words = np.vstack(words)
    samples_in_clusters = np.sum(words, axis=0)
    sorted_index_array = np.argsort(-samples_in_clusters)
    sorted_index_array = sorted_index_array[:top_dense]
    
    # form a dict with features from chosen cluster
    for cno, cluster_idx in enumerate(sorted_index_array):
        
        cl_samples = words[:, cluster_idx]
        # videos having most no. of cluster samples appear first
        sorted_idxs = np.argsort(-cl_samples)
        
        cl_vnames = [strokes_name_id[i] for i in sorted_idxs]
        
        write_cluster_samples(features, onehot_feats, cl_vnames, cluster_idx, 
                              offset, ds_path, base, cno)
            
    return 

def waitTillEscPressed():
    while(True):
        # For moving forward
        if cv2.waitKey(0)==27:
            print("Esc Pressed. Move Forward.")
            return 1
        # For moving back
        elif cv2.waitKey(0)==98:
            print("'b' pressed. Move Back.")
            return 0
        # start of shot
        elif cv2.waitKey(0)==115:
            print("'s' pressed. Start of shot.")
            return 2
        # end of shot
        elif cv2.waitKey(0)==102:
            print("'f' pressed. End of shot.")
            return 3

def make_codebook(vecs, nclusters, model_type="kmeans"):
    """
    Function to find the clusters using KMeans
    Parameters:    
        vecs: any dataframe representing the input space points
        nclusters: No. of clusters to be formed
        model_type : str
            'kmeans' or 'gmm' for selecting the clustering model.
    Returns:
        KMeans or GaussianMixture object, containing the clustering information.
    """
    assert model_type == "kmeans" or model_type == "gmm", "Invalid model_type."
    if model_type == 'kmeans':
        print("Clustering using KMeans: Input size -> {} :: n_clusters -> {}"\
              .format(vecs.shape, nclusters))   
        model = KMeans(n_clusters=nclusters, n_init=10, n_jobs=2, random_state=128)
        model.fit(vecs)
    elif model_type == 'gmm':
        print("Clustering using GMM: Input size -> {} :: n_components -> {}"\
              .format(vecs.shape, nclusters))
        model = GaussianMixture(n_components=nclusters, covariance_type='diag',
                              random_state=128).fit(vecs)        
    
    print("Done Clustering!")
    return model


def create_bovw_onehot(features, strokes_name_id, model):
    '''
    Form one hot vector representations for  OF / HOOF / C3D FC7 features.
    Returns dictionary with {vidname : np.array((NFeats, FeatSize)), ...}.
    
    Parameters:
    ------
    features: dict
        {'video_id1':np.array((N, d)), ...}  , where d is vec dimension
    strokes_name_id: list of str
        key values of features, eg. video-name_34_242 
    km_model: KMeans / GaussianMixture obj
        Learned KMeans / GMM model with nClusters / nComponents
        
    Returns:
    ------
    Dictionary similar to features dictionary, but having one-hot vector 
    representations for each video.
    
    '''
    # get the cluster centroids
    if isinstance(model, KMeans):
        n_clusters = model.n_clusters
    else:
        n_clusters = model.n_components
    n_strokes = len(strokes_name_id)
        
    words = {}
    row_no = 0
    # Make bow vectors for all videos.
    for video in strokes_name_id:
        # Get the starting and ending stroke frame positions        
#        m, n = video.rsplit('_', 2)[1:]
#        m, n = int(m), int(n)
        
        # select the vectors of size Nx1x4096 and remove mid dimension of 1
        stroke_feats = features[video]
        stroke_feats[np.isnan(stroke_feats)] = 0
        stroke_feats[np.isinf(stroke_feats)] = 0
        # find cluster centroid assignments for all points
        # returns a tuple, with first element having ids of the cluster centroid 
        # to which the row i belongs to. Second element is the distance between 
        # the nearest code and the ith row.
        # visual_word_ids is a 1D array
        if isinstance(model, KMeans):
            word_ids = vq(stroke_feats, model.cluster_centers_)[0]  # ignoring the distances in [1]
        else:
            word_ids = model.predict(stroke_feats)
        
        stroke_onehot = np.zeros((stroke_feats.shape[0], n_clusters), dtype=np.long)
        stroke_onehot[np.arange(stroke_feats.shape[0]), word_ids] = 1
#        for w_no, word_id in enumerate(word_ids):
#            stroke_onehot[w_no, word_id] += 1
#            
        words[video] = stroke_onehot
        row_no +=1
    return words


def create_bovw_SA(features, strokes_name_id, model):
    '''
    Form soft assignment vector representations for OF / HOOF / C3D FC7 features.
    Returns dictionary with {vidname : np.array((NFeats, FeatSize)), ...}.
    
    Parameters:
    ------
    features: dict
        {'video_id1':np.array((N, d)), ...}  , where d is vec dimension
    strokes_name_id: list of str
        key values of features, eg. video-name_34_242 
    km_model: KMeans / GaussianMixture obj
        Learned KMeans / GMM model with nClusters / nComponents
        
    Returns:
    ------
    Dictionary similar to features dictionary, but having soft assignment vector 
    representations for each video.
    
    '''
    # get the cluster centroids
    if isinstance(model, KMeans):
        n_clusters = model.n_clusters
    else:
        n_clusters = model.n_components
    n_strokes = len(strokes_name_id)
    
    words = {}
    row_no = 0
    beta = -1.
    if features[list(features.keys())[0]].shape[1] == 2304:
        beta = -0.6     # For ofGrid10, exp operation gives large values for beta=-1
    # Make bow vectors for all videos.
    for video in strokes_name_id:
        # Get the starting and ending stroke frame positions        
#        m, n = video.rsplit('_', 2)[1:]
#        m, n = int(m), int(n)
        
        # select the vectors of size Nx1x4096 and remove mid dimension of 1
        stroke_feats = features[video]
        stroke_feats[np.isnan(stroke_feats)] = 0
        stroke_feats[np.isinf(stroke_feats)] = 0
        # find cluster centroid assignments for all points
        # returns a tuple, with first element having ids of the cluster centroid 
        # to which the row i belongs to. Second element is the distance between 
        # the nearest code and the ith row.
        # visual_word_ids is a 1D array
        
        if isinstance(model, KMeans):
            # calculate L2 dist of each row from all cluster centers
            cl_dists = [(np.linalg.norm(model.cluster_centers_ - stroke_feats[i,:], axis=1)) \
                        for i in range(stroke_feats.shape[0])]
            # form nFeats x nClusters (distance of a feature from all the cluster centers)
            cl_dists = np.vstack(cl_dists)      # unnormalized
#            cl_dists = normalize(cl_dists, "l2") #**2, axis=1, norm="l2")   # accuracy decreases 
        else:
            cl_dists = model.predict_proba(stroke_feats)
        
##        omega = np.sum(cl_dists, axis=0) / np.sum(cl_dists)
        omega = np.exp(beta * cl_dists)     # beta=1, decreasing it reduces accuracy
        omega = omega / omega.sum(axis = 1)[:, None]    # normalize
#        bovw_df[row_no, :] = np.sum(omega, axis=0) / omega.shape[0]
        words[video] = omega
        row_no +=1

    return words

def create_bovw_OMP(features, strokes_name_id, model):
    '''
    Form a features dataframe of C3D FC7/HOOF features kept in feats_data_dict. 
    Returns one dataframe of (nTrimmedVids, nClusters). Use Soft Assignment
    
    Parameters:
    ------
    features: dict
        {'video_id1':np.array((N, d)), ...}  , where d is vec dimension
    strokes_name_id: list of str
        key values of features, eg. video-name_34_242 
    km_model: KMeans / GaussianMixture obj
        Learned KMeans / GMM model with nClusters / nComponents
        
    Returns:
    ------
    pd.Dataframe of size (nTrimmedVids, nClusters)
    with frequency histogram of trimmed videos
    Also return a string sequence of words with integer values representing
    cluster centers.
    
    '''
    # get the cluster centroids
    if isinstance(model, KMeans):
        n_clusters = model.n_clusters
    else:
        n_clusters = model.n_components
    n_strokes = len(strokes_name_id)
    
    words = {}
    row_no = 0
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=5)
    phi = np.transpose(model.cluster_centers_)
    phi = normalize(phi, axis=0, norm='l2')
    #omp.fit(np.transpose(model.cluster_centers_), stroke_feats[0,:])

    # Make bow vectors for all videos.
    for video_index, video in enumerate(strokes_name_id):
        # Get the starting and ending stroke frame positions        
#        m, n = video.rsplit('_', 2)[1:]
#        m, n = int(m), int(n)
        
        # select the vectors of size Nx1x4096 and remove mid dimension of 1
        stroke_feats = features[video]
        stroke_feats[np.isnan(stroke_feats)] = 0
        stroke_feats[np.isinf(stroke_feats)] = 0
        # find cluster centroid assignments for all points
        # returns a tuple, with first element having ids of the cluster centroid 
        # to which the row i belongs to. Second element is the distance between 
        # the nearest code and the ith row.
        # visual_word_ids is a 1D array
        
        stroke_feats = normalize(stroke_feats, axis=1, norm='l2')
        print("row No : {}".format(row_no))
        sparse_vecs = []
        
        for i in range(stroke_feats.shape[0]):
            omp.fit(phi, stroke_feats[i,:])
            sparse_vecs.append(omp.coef_)
            
        sparse_vecs = np.vstack(sparse_vecs)

##        omega = np.sum(cl_dists, axis=0) / np.sum(cl_dists)
#        omega = np.exp(-1.0 * sparse_vecs)     # beta=1, decreasing it reduces accuracy
#        omega = omega / omega.sum(axis = 1)[:, None]    # normalize
            
#            
        words[video] = sparse_vecs
        row_no +=1
    
#    bovw_df = pd.DataFrame(bovw_df, index=strokes_name_id)
    return words