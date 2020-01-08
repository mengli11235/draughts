"""auxiliary files to plot loss"""

import os
import numpy as np
from utils import *
from pickle import Pickler, Unpickler
import matplotlib.pyplot as plt
import copy
import torch
        
if __name__=="__main__":
    filename = os.path.join('/data/s2651513/draughts/temp-residual_baseline', str(False)+".loss")
    loss_arr = []
    with open(filename, "rb") as f:
        loss_dicts1 = Unpickler(f).load()
    f.closed
    for loss_dict in loss_dicts1:
        loss_arr.append(loss_dict['total_loss'])
    index0 = len(loss_arr) if len(loss_arr) < 50 else 50
    plt.plot(loss_arr[0:index0],label="CNN") 
    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss_v+Loss_pi")
    plt.savefig('/data/s2651513/draughts/temp-residual_baseline'+"/"+str(len(loss_arr)) + "totalLoss.jpg")
    plt.close()
    
    filename = os.path.join('/data/s2651513/draughts/temp-residual_mvp', str(False)+".loss")
    with open(filename, "rb") as f:
        loss_dicts = Unpickler(f).load()
    f.closed
    loss_arr1 = []
    loss_arr2 = []
    for loss_dict in loss_dicts:
        loss_arr1.append(loss_dict['total_loss'])
        loss_arr2.append(loss_dict['stotal_loss'])
    index0 = len(loss_arr1) if len(loss_arr1) < 50 else 50
    plt.plot(loss_arr1[0:index0],label="large CNN") 
    plt.plot(loss_arr2[0:index0],label="small CNN")
    plt.legend(loc=1,ncol=1)
    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss_v+Loss_pi")
    plt.savefig('/data/s2651513/draughts/temp-residual_mvp'+"/"+str(len(loss_arr1)) + "totalLoss.jpg")
    plt.close()
    
    filename = os.path.join('/data/s2651513/draughts/temp-residual_three', str(True)+".loss")
    with open(filename, "rb") as f:
        loss_dicts = Unpickler(f).load()
    f.closed
    loss_arr1 = []
    loss_arr2 = []
    for loss_dict in loss_dicts:
        loss_arr1.append(loss_dict['total_loss'])
        loss_arr2.append(loss_dict['s2total_loss'])
    index0 = len(loss_arr1) if len(loss_arr1) < 50 else 50
    plt.plot(loss_arr1[0:index0],label="large CNN") 
    plt.plot(loss_arr2[0:index0],label="small CNN")
    plt.legend(loc=1,ncol=1)
    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss_v+Loss_pi")
    plt.savefig('/data/s2651513/draughts/temp-residual_three'+"/"+str(len(loss_arr1)) + "totalLoss.jpg")
    plt.close()
    
    
    filename = os.path.join('/data/s2651513/draughts/temp-residual_mvp_three', str(True)+".loss")
    with open(filename, "rb") as f:
        loss_dicts = Unpickler(f).load()
    f.closed
    loss_arr1 = []
    loss_arr2 = []
    loss_arr3 = []
    for loss_dict in loss_dicts:
        loss_arr1.append(loss_dict['total_loss'])
        loss_arr2.append(loss_dict['stotal_loss'])
        loss_arr3.append(loss_dict['s2total_loss'])
    index0 = len(loss_arr1) if len(loss_arr1) < 50 else 50
    plt.plot(loss_arr1[0:index0],label="large CNN") 
    plt.plot(loss_arr2[0:index0],label="small CNN 1")
    plt.plot(loss_arr3[0:index0],label="small CNN 2")
    plt.legend(loc=1,ncol=1)
    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss_v+Loss_pi")
    plt.savefig('/data/s2651513/draughts/temp-residual_mvp_three'+"/"+str(len(loss_arr1)) + "totalLoss.jpg")
    plt.close()
    
    filename = os.path.join('/data/s2651513/draughts/temp-residual_three_large', str(True)+".loss")
    with open(filename, "rb") as f:
        loss_dicts = Unpickler(f).load()
    f.closed
    loss_arr1 = []
    loss_arr2 = []
    for loss_dict in loss_dicts:
        loss_arr1.append(loss_dict['total_loss'])
        loss_arr2.append(loss_dict['s2total_loss'])
    index0 = len(loss_arr1) if len(loss_arr1) < 50 else 50
    plt.plot(loss_arr1[0:index0],label="CNN 1") 
    plt.plot(loss_arr2[0:index0],label="CNN 2")
    plt.legend(loc=1,ncol=1)
    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss_v+Loss_pi")
    plt.savefig('/data/s2651513/draughts/temp-residual_three_large'+"/"+str(len(loss_arr1)) + "totalLoss.jpg")
    plt.close()
    
    
    filename = os.path.join('/data/s2651513/draughts/temp-residual_mvp_three_large', str(True)+".loss")
    with open(filename, "rb") as f:
        loss_dicts = Unpickler(f).load()
    f.closed
    loss_arr1 = []
    loss_arr2 = []
    loss_arr3 = []
    loss_arr4 = []
    for loss_dict in loss_dicts:
        loss_arr1.append(loss_dict['total_loss'])
        loss_arr2.append(loss_dict['stotal_loss'])
        loss_arr3.append(loss_dict['s2total_loss'])
        loss_arr4.append(loss_dict['s3total_loss'])
    index0 = len(loss_arr1) if len(loss_arr1) < 50 else 50
    plt.plot(loss_arr1[0:index0],label="large CNN 1") 
    plt.plot(loss_arr2[0:index0],label="small CNN 1")
    plt.plot(loss_arr3[0:index0],label="large CNN 2")
    plt.plot(loss_arr4[0:index0],label="small CNN 2")
    plt.legend(loc=1,ncol=1)
    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss_v+Loss_pi")
    plt.savefig('/data/s2651513/draughts/temp-residual_mvp_three_large'+"/"+str(len(loss_arr1)) + "totalLoss.jpg")
    plt.close()
