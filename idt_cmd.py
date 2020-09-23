#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 01:35:15 2020

@author: arpan


"""

import csv

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
#        print("Read {} ground truth stroke labels from file.".format(line_count))
        
    if min(labs_values) == 1:
        labs_values = [l-1 for l in labs_values]
        labs_keys = [k.replace('.avi', '') for k in labs_keys]
    return labs_keys, labs_values


if __name__ == "__main__":
    ANNOTATION_FILE = "/home/arpan/VisionWorkspace/Cricket/CricketStrokeLocalizationBOVW/shots_classes.txt"
    DATASET = "/home/lnmiit/VisionWorkspace/improved_trajectory_release/datasets/ICC_WT20_320x180"
    
    labs_keys, labs_values = get_cluster_labels(ANNOTATION_FILE)
    
    for stroke in labs_keys:
        src_key = stroke.rsplit('_', 2)
        src_file = src_key[0]+'.avi'
        stFrm, endFrm = src_key[1], src_key[2]
        dest_file = src_key[0] + "_" + stFrm + "_" + endFrm + ".out"
        cmd = "./release/DenseTrackStab " + "\"./datasets/ICC_WT20_320x180/" + \
                src_file + "\" -S " + stFrm + " -E " + endFrm + " -L 25 -W 15 > " + \
                "\"datasets/iDT_strokes/ICC_WT20/" + dest_file + "\""
        print(cmd)