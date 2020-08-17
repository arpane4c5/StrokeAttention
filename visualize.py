#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 19:35:53 2020

@author: arpan
"""

import os
import pickle
import cv2
import numpy as np
import torch

def visualize_attn_wts(pred_dict, base_name=""):
    
    vid_name = vid_path
    
    
    return 



if __name__ == '__main__':
    
    base_path = "/home/arpan/VisionWorkspace/Cricket/StrokeAttention/logs"
    with open(os.path.join(base_path, "pred_dict.pkl"), "rb") as fp:
        pred_out_dict = pickle.load(fp)
    
    visualize_attn_wts(pred_out_dict)
    