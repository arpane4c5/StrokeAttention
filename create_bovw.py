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

def vis_cluster(features, strokes_name_id, model, cl_no, offset, ds_path, base=""):
    '''
    searches the data partition for clusters and writes frames corresponding to 
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
    # create path
    dest_dir = os.path.join(base, str(cl_no))
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    count, max_frms = 0, 40
    # Make bow vectors for all videos.
    for video_index, video in enumerate(strokes_name_id):
        # Get the starting and ending stroke frame positions        
        m, n = video.rsplit('_', 2)[1:]
        m, n = int(m), int(n)
        
        stroke_feats = features[video]
        # get rows indexes where cluster cl_no is assigned to frame no.
        frm_pos = np.where(stroke_feats[:,cl_no] > 0)[0]
        for frm_no in frm_pos:
            # retrieve the frame positions 
            img = get_frame(ds_path, video, frm_no, offset)
            cv2.imwrite(os.path.join(dest_dir, video+"_"+str(frm_no)+".png"), img)
            count+=1
            if count > max_frms:
                break
#            cv2.imshow("Stroke {}_{} : F{}".format(m, n, frm_no), img)
#            direction = waitTillEscPressed()
        if count > max_frms:
            break
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
        m, n = video.rsplit('_', 2)[1:]
        m, n = int(m), int(n)
        
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
        m, n = video.rsplit('_', 2)[1:]
        m, n = int(m), int(n)
        
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
#            cl_dists = normalize(cl_dists**2, axis=1, norm="l2")   # accuracy decreases 
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
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=25)
    phi = np.transpose(model.cluster_centers_)
    phi = normalize(phi, axis=0, norm='l2')
    #omp.fit(np.transpose(model.cluster_centers_), stroke_feats[0,:])

    # Make bow vectors for all videos.
    for video_index, video in enumerate(strokes_name_id):
        # Get the starting and ending stroke frame positions        
        m, n = video.rsplit('_', 2)[1:]
        m, n = int(m), int(n)
        
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