#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 06:25:01 2020

@author: arpan
"""

from matplotlib import pyplot as plt
import os
import numpy as np
import re
#plt.style.use('ggplot')


def plot_traintest_loss(keys, l, xlab, ylab, seq, batch, destfile):
    # Plot the loss values for the different epochs in one trained model
    keylist = range(1, len(l[keys[0]])+1)      # x-axis for 30 epochs
    cols = ['r','g','b', 'c']    
    print("Iteration and Accuracy Lists : ")
    print(keylist)
    print(l)
    fig = plt.figure(2)
    plt.title("Loss Vs Epoch (Seq_Len="+str(seq)+", Batch="+str(batch)+")", fontsize=12)
    plt.plot(keylist, l[keys[0]], lw=1, color=cols[0], marker='.', label= keys[0])
    plt.plot(keylist, l[keys[1]], lw=1, color=cols[1], marker='.', label= keys[1])
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
#    plt.show()
    plt.savefig(destfile, bbox_inches='tight', dpi=300)    
    plt.close(fig)
    return

def plot_traintest_accuracy(keys, l, xlab, ylab, seq, batch, best, destfile):
    # Plot the accuracy values for the different epochs in one trained model
    keylist = range(1, len(l[keys[0]])+1)      # x-axis for 30 epochs
    cols = ['r','g','b', 'c']
    print("Iteration and Accuracy Lists : ")
    print(keylist)
    print(l)
    fig = plt.figure(2)
    plt.title("Accuracy Vs Epoch (Seq_Len="+str(seq)+", Batch="+str(batch)+")", fontsize=12)
    plt.plot(keylist, l[keys[0]], lw=1, color=cols[0], marker='.', label= keys[0])
    plt.plot(keylist, l[keys[1]], lw=1, color=cols[1], marker='.', label= keys[1])
    plt.xlabel(xlab)
    plt.ylabel(ylab)
#    plt.axvline(x=best, color='r', linestyle='--')
    plt.legend()
    plt.ylim(bottom=0, top=1)
#    plt.show()
    plt.savefig(destfile, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return

def plot_accuracy(x, keys, l, xlab, ylab, fname):
    
#    keys = ["HOG", "HOOF", "OF Grid 20", "C3D $\mathit{FC7}$: $w_{c3d}=17$"]
#    l = {keys[0]: hog_acc, keys[1]: hoof_acc, keys[2]: of30_acc, keys[3]:accuracy_17_30ep}
    cols = ['r','g','b', 'c']        
    print(l)
    fig = plt.figure(2)
    plt.title("Accuracy Vs #Words", fontsize=12)
    for i in range(len(keys)):
        acc = l[keys[i]]
        plt.plot(x[(len(x)-len(acc)):], acc, lw=1, color=cols[i], marker='.', label=keys[i])
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(loc=4)
    plt.ylim(bottom=0, top=1)
    plt.show()
    fig.savefig(fname, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return

def plot_acc_of20(x, keys, l, xlab, ylab, fname):    
    cols = ['r','g','b', 'c']
    print(l)
    fig = plt.figure(2)
    plt.title("Accuracy Vs Sequence Length", fontsize=13)
    for i in range(len(keys)):
        acc = l[keys[i]]
        plt.plot(x[(len(x)-len(acc)):], acc, lw=1, color=cols[i], marker='.', label=keys[i])
    plt.xlabel(xlab, fontsize=12)
    plt.ylabel(ylab, fontsize=12)
    plt.axvline(x=24, color='r', linestyle='--')
    plt.legend(loc=4)
    plt.ylim(bottom=0, top=1)
    plt.show()
    fig.savefig(fname, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return


def plot_acc_of20_GRU_HA(x, keys, l, xlab, ylab, fname):
    cols = ['r','g','b', 'c']
    print(l)
    fig = plt.figure(2)
    plt.title("Accuracy Vs No. of Clusters(C)", fontsize=13)
    for i in range(len(keys)):
        acc = l[keys[i]]
        plt.plot(x[(len(x)-len(acc)):], acc, lw=1, color=cols[i], marker='.', label=keys[i])
    plt.xlabel(xlab, fontsize=12)
    plt.ylabel(ylab, fontsize=12)
    plt.legend(loc=4)
    plt.ylim(bottom=0, top=1)
    plt.show()
    fig.savefig(fname, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return

def plot_acc_diff_feats(x, keys, l, xlab, ylab, fname):
    cols = ['r','g','b', 'c']
    print(l)
    fig = plt.figure(2)
    plt.title("Accuracy Vs Sequence Length", fontsize=13)
    for i in range(len(keys)):
        acc = l[keys[i]]
        plt.plot(x[(len(x)-len(acc)):], acc, lw=1, color=cols[i], marker='.', label=keys[i])
    plt.xlabel(xlab, fontsize=12)
    plt.ylabel(ylab, fontsize=12)
    plt.axvline(x=20, color='r', linestyle='--')
    plt.axvline(x=2, color='g', linestyle='-')
    plt.axvline(x=2, color='b', linestyle='--')
    plt.axvline(x=32, color='c', linestyle='--')
    plt.legend(loc=4)
    plt.ylim(bottom=0, top=1)
    plt.show()
    fig.savefig(fname, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return

def plot_sample_dist():
    x = ['C-1', 'C-2', 'C-3', 'C-4', 'C-5']
    cat_wts = [2644.0, 14330.0, 7837.0, 3926.0, 9522.0]
    
    x_pos = [i for i, _ in enumerate(x)]
    
    plt.bar(x_pos, cat_wts, width=0.5)#, color='cyan')
    plt.xlabel("Stroke Categories", fontsize=12)
    plt.ylabel("#Samples", fontsize=12)
    plt.title("Number of samples per category", fontsize=16)
    plt.xticks(x_pos, x)
    plt.savefig("sampleDist.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    
    seq = list(range(2,41,2))
    OF20_HA_C1k_H256 = [0.7142857142857143, 0.7428571428571429, 0.7333333333333333, 
                   0.6761904761904762, 0.6857142857142857, 0.6952380952380952, 
                   0.6476190476190476, 0.7047619047619048, 0.7238095238095238, 
                   0.780952380952381, 0.6952380952380952, 0.7428571428571429, 
                   0.7333333333333333, 0.7142857142857143, 0.6952380952380952, 
                   0.7142857142857143, 0.6857142857142857, 0.7523809523809524, 
                   0.6571428571428571, 0.7142857142857143]
    
    OF20_HA_C2k_H256 = [0.6952380952380952, 0.7142857142857143, 0.7142857142857143, 
                   0.7333333333333333, 0.6571428571428571, 0.7142857142857143, 
                   0.638095238095238, 0.7047619047619048, 0.6285714285714286, 
                   0.7047619047619048, 0.6571428571428571, 0.6, 0.6571428571428571, 
                   0.6761904761904762, 0.6571428571428571, 0.7333333333333333, 
                   0.7238095238095238, 0.6761904761904762, 0.6952380952380952, 
                   0.6666666666666666]

    OF20_SA_C1k_H256 = [0.2761904761904762, 0.6571428571428571, 0.6952380952380952, 
                   0.7047619047619048, 0.6857142857142857, 0.6476190476190476, 
                   0.6952380952380952, 0.6761904761904762, 0.7523809523809524, 
                   0.6952380952380952, 0.7333333333333333, 0.7333333333333333, 
                   0.7047619047619048, 0.7238095238095238, 0.7619047619047619, 
                   0.6190476190476191, 0.6952380952380952, 0.6, 0.7523809523809524, 
                   0.5523809523809524]
    
    OF10_HA_C1k_H256 = [0.7047619047619048, 0.6952380952380952, 0.7523809523809524, 
                        0.6857142857142857, 0.7428571428571429, 0.7428571428571429, 
                        0.7047619047619048, 0.7047619047619048, 0.6857142857142857, 
                        0.6857142857142857, 0.7047619047619048, 0.6761904761904762, 
                        0.7142857142857143, 0.6666666666666666, 0.6857142857142857, 
                        0.6666666666666666, 0.6857142857142857, 0.6761904761904762, 
                        0.6761904761904762, 0.638095238095238]
    
    HOOF_B20_HA_C1k = [0.7142857142857143, 0.6571428571428571, 0.6190476190476191, 
                       0.6476190476190476, 0.6, 0.5714285714285714, 0.4857142857142857, 
                       0.5238095238095238, 0.5428571428571428, 0.5238095238095238, 
                       0.49523809523809526, 0.5714285714285714, 0.5619047619047619, 
                       0.5714285714285714, 0.580952380952381, 0.6476190476190476, 
                       0.5523809523809524, 0.5904761904761905, 0.5333333333333333, 
                       0.6285714285714286]
    
    GRU_OF20_H512 = [0.6857142857142857, 0.6476190476190476, 0.6761904761904762, 
                0.6952380952380952, 0.580952380952381, 0.6761904761904762, 
                0.5428571428571428, 0.6095238095238096, 0.6, 0.638095238095238, 
                0.5619047619047619, 0.6761904761904762, 0.6857142857142857, 
                0.6476190476190476, 0.6190476190476191, 0.6857142857142857, 
                0.6571428571428571, 0.5333333333333333, 0.7047619047619048, 
                0.6666666666666666]
    
    OF20_HA_C1k_H512 = [0.6857142857142857, 0.7428571428571429, 0.7047619047619048, 
                        0.7523809523809524, 0.7238095238095238, 0.7333333333333333, 
                        0.7142857142857143, 0.7428571428571429, 0.6952380952380952, 
                        0.7523809523809524, 0.6952380952380952, 0.7238095238095238, 
                        0.7047619047619048, 0.7238095238095238, 0.6952380952380952, 
                        0.7047619047619048, 0.6761904761904762, 0.6952380952380952, 
                        0.7047619047619048, 0.6857142857142857]  # Params 8915973 
    
    seq_cnn = list(range(2, 31, 2))
    CNN2D_HA_C1k_H256 = [0.47619047619047616, 0.44761904761904764, 0.41904761904761906, 
                    0.4380952380952381, 0.42857142857142855, 0.44761904761904764, 
                    0.45714285714285713, 0.42857142857142855, 0.42857142857142855, 
                    0.4095238095238095, 0.4095238095238095, 0.3904761904761905, 
                    0.41904761904761906, 0.4380952380952381, 0.3619047619047619, 
                    0.4380952380952381, 0.45714285714285713, 0.4, 0.41904761904761906, 
                    0.38095238095238093]
    
    CNN3D_HA_C1k_H256 = [0.4380952380952381, 0.4857142857142857, 0.44761904761904764, 
                         0.47619047619047616, 0.42857142857142855, 0.4857142857142857, 
                         0.47619047619047616, 0.4, 0.5047619047619047, 0.4380952380952381, 
                         0.41904761904761906, 0.37142857142857144, 0.42857142857142855]
    
    CNN3D_HA_C2k_H256 = [0.5047619047619047, 0.5142857142857142, 0.4666666666666667, 
                         0.4666666666666667, 0.4380952380952381, 0.44761904761904764, 
                         0.4666666666666667, 0.49523809523809526, 0.4380952380952381, 
                         0.41904761904761906, 0.42857142857142855, 0.45714285714285713, 
                         0.42857142857142855]
    
    ###########################################################################
    # Plot OF20 HA SA Comparison for C=1k and 2k 
    
#    keys = ["HA ; C=1000", "HA ; C=2000", "SA ; C=1000"]  # OFGrid20 Hidden=256
#    l = {keys[0] : OF20_HA_C1k_H256, keys[1] : OF20_HA_C2k_H256, keys[2] : OF20_SA_C1k_H256}
#    
#    fname = os.path.join("logs", "OF20_HA_SA.png")
#    plot_acc_of20(seq, keys, l, "Sequence Length", "Accuracy", fname)

    ###########################################################################
    # Plot OF20 GRU sequences Vs HA sequences Comparison for C=1k and 2k 
    
#    keys = ["OF Grid=20; HA; C=1000", "OF Grid=20 Sequences"]  # OFGrid20 Hidden=512 
#    l = {keys[0] : OF20_HA_C1k_H512, keys[1] : GRU_OF20_H512}
#    
#    fname = os.path.join("logs", "OF20_GRU_Vs_HA.png")
#    plot_acc_of20_GRU_HA(seq, keys, l, "Sequence Length", "Accuracy", fname)
    
    ###########################################################################
    # Plot Comparison for different features with HA and C=1k
    
#    keys = ["OF Grid=20", "HOOF Bins=20", "2D CNN", "3DCNN"]  # Hidden=256, C=1k 
#    l = {keys[0] : OF20_HA_C1k_H256, keys[1] : HOOF_B20_HA_C1k, 
#         keys[2] : CNN2D_HA_C1k_H256, keys[3] : CNN3D_HA_C1k_H256}
#    
#    fname = os.path.join("logs", "CompareFeats.png")
#    plot_acc_diff_feats(seq, keys, l, "Sequence Length", "Accuracy", fname)
    
    ###########################################################################
    # Plot comparing OF10 Vs OF20 with H256
    
#    keys = ["OF Grid=10; C=1000", "OF Grid=20; C=1000"]  # HA Hidden=256 
#    l = {keys[0] : OF10_HA_C1k_H256, keys[1] : OF20_HA_C1k_H256}
#    
#    fname = os.path.join("logs", "OF10_Vs_OF20_HA_C1k_H256.png")
#    plot_acc_of20_GRU_HA(seq, keys, l, "Sequence Length", "Accuracy", fname)
    
    ###########################################################################
#    # Plot Comparison for different cluster sizes with SA and HIDDEN=128, SEQ=22
#    
#    nclust_acc = [0.6285714285714286, 0.7047619047619048, 0.7238095238095238, 0.6571428571428571,
#           0.7333333333333333, 0.6476190476190476, 0.7142857142857143, 0.7142857142857143, 
#           0.6476190476190476, 0.7523809523809524, 0.5714285714285714, 0.7142857142857143,
#           0.7333333333333333, 0.7047619047619048, 0.7714285714285715, 0.6857142857142857,
#           0.7142857142857143, 0.7238095238095238, 0.6857142857142857, 0.7142857142857143]
#    keys = ["OF Grid=20 Hidden=128"]  # Hidden=256, C=1k 
#    l = {keys[0] : nclust_acc}
#    
#    fname = os.path.join("logs", "VaryingNWords.png")
#    plot_acc_of20_GRU_HA(list(range(10, 201, 10)), keys, l, 
#                         "No. of Clusters/Visual Words (C)", "Accuracy", fname)
    
#    ###########################################################################
#    # 2 Stream (OF20 + HOG SA C=1000) validation accuracies for seq=range(2, 41, 2)
#    nclust_acc = [0.5428571428571428, 0.7428571428571429, 0.6, 0.7142857142857143, 
#                  0.7238095238095238, 0.7047619047619048, 0.7142857142857143, 
#                  0.6857142857142857, 0.7238095238095238, 0.780952380952381, 
#                  0.7428571428571429, 0.780952380952381, 0.7428571428571429, 
#                  0.7428571428571429, 0.7333333333333333, 0.7428571428571429, 
#                  0.12380952380952381, 0.7047619047619048, 0.7142857142857143, 
#                  0.638095238095238]
#    
#    keys = ["2 Stream GRU Hidden=256"]
#    l = {keys[0] : nclust_acc}
#    fname = os.path.join("logs/plot_data", "2Stream_OF20_HOG_seq2_40.png")
#    plot_acc_of20(list(range(2, 41, 2)), keys, l, "Seq. Length", "Accuracy", fname)
#    
#    ###########################################################################
#    
#    # Plot the 2Stream GRU training (on HOG and OF20 feats SA) with Seq=24 and C=1000
#    # Val Accuracy : 0.780952380952381 
#    file = "logs/plot_data/2stream_seq24_C1000.txt" # 
#    train_loss, test_loss, train_acc, test_acc = [], [], [], []
#    with open(file, 'r') as fp:
#        lines = fp.readlines()
#    for line in lines: 
#        line = line.strip()
#        if 'train Loss:' in line:
#            t = re.findall("\d+\.\d+", line)
#            train_loss.append(float(t[0]))
#            train_acc.append(float(t[1]))
#        elif 'test Loss:' in line:
#            t = re.findall("\d+\.\d+", line)
#            test_loss.append(float(t[0]))
#            test_acc.append(float(t[1]))
#        elif 'SEQ_SIZE : ' in line:
#            t = re.findall("\d+", line)
#            seq = str(t[0])
#    
#    l1 = {"train loss" : train_loss, "test loss": test_loss}
#    l2 = {"train accuracy" : train_acc, "test accuracy" : test_acc}
#    best_ep = test_acc.index(max(test_acc)) + 1
#    loss_file = 'logs/plot_data/2stream_losses_seq'+str(seq)+'.png'
#    acc_file = 'logs/plot_data/2stream_acc_seq'+str(seq)+'.png'
#    plot_traintest_loss(["train loss", "test loss"], l1, "Epochs", "Loss", seq, 32, loss_file)
#    plot_traintest_accuracy(["train accuracy", "test accuracy"], l2, "Epochs", "Accuracy", seq, 
#                            32, best_ep, acc_file)
#    
    ###########################################################################
    # Plot the C3D finetuning losses (SEQ_SIZE = 16, STEP = 4, BATCH = 16, ITer=150/Ep)
    file = "logs/plot_data/C3DFine_seq16_SGD.txt" # 
    train_loss, test_loss, train_acc, test_acc = [], [], [], []
    seq = 16
    with open(file, 'r') as fp:
        lines = fp.readlines()
    for line in lines: 
        line = line.strip()
        if 'train Loss:' in line:
            t = re.findall("\d+\.\d+", line)
            train_loss.append(float(t[0]))
            train_acc.append(float(t[1]))
        elif 'test Loss:' in line:
            t = re.findall("\d+\.\d+", line)
            test_loss.append(float(t[0]))
            test_acc.append(float(t[1]))
            
    l1 = {"train loss" : train_loss, "test loss": test_loss}
    l2 = {"train accuracy" : train_acc, "test accuracy" : test_acc}
    best_ep = test_acc.index(max(test_acc)) + 1
    loss_file = 'logs/plot_data/C3DFine_seq16.png'
    acc_file = 'logs/plot_data/C3DFine_acc_seq16.png'
    plot_traintest_loss(["train loss", "test loss"], l1, "Epochs", "Loss", seq, 16, loss_file)
    plot_traintest_accuracy(["train accuracy", "test accuracy"], l2, "Epochs", "Accuracy", seq, 
                            16, best_ep, acc_file)
    
    ###########################################################################
    