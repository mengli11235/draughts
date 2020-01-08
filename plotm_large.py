"""auxiliary files to plot results against the baseline model"""

import Arena
from MCTS import MCTS
from DraughtsGame import DraughtsGame, display
from DraughtsPlayers import *
from pytorch.NNet import NNetWrapper as nn
import os
import torch
import numpy as np
from utils import *
from pickle import Pickler, Unpickler
import matplotlib.pyplot as plt
from optparse import OptionParser
import copy

args = dotdict({
    'numIters': 1000,
    'numEps': 100,
    'tempThreshold': 10,
    'updateThreshold': 0.55,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 50,
    'arenaCompare': 20,
    'cpuct': 1,

    'checkpoint': '/data/s2651513/draughts/temp-',
    'load_model': False,
    'plot': True,
    'pretrain':False,
    'load_folder_file': ('/data/s2651513/draughts/examples','gp5.examples'),
    'load_folder_model': ('/data/s2651513/draughts/models','next.pth.tar'),
    'numItersForTrainExamplesHistory': 5,
    
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'model': 'mlp',
    'features': 8,
    'num_channels': 64,
    
    'budget_ratio': 1,
    'three_stages': False,
    'stage': 3,
    'large': False,
})

def loadLog(args):
    three_arr = []
    iter_arr = []
    mvp_three = []
    iteration = 0
    folder = args.checkpoint
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, "log.txt")
    if os.path.isfile(filename):
        with open(filename, "r") as f:
            iteration = int(f.read())
        f.closed
    print("Plot Elo scores up to the "+str(iteration)+" iteration...")
    filename = os.path.join(folder, "mat_plot_models_large.dat")
    if os.path.isfile(filename):
        with open(filename, "rb") as f:
            print("Plot history found. Read from:")
            temp = Unpickler(f).load()
            [three_arr, mvp_three, iter_arr] = temp
            print(iter_arr[-1])
        f.closed
    return iteration, three_arr, mvp_three, iter_arr

def saveArr(mat):
    folder = args.checkpoint
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, "mat_plot_models_large.dat")
    with open(filename, "wb+") as f:
        Pickler(f).dump(mat)
    f.closed
def calculate_elo(score, matches, np, opponent=0, k=32):
      #calculate ELO
      score -= matches*1/(1+10**((opponent-np)/400))
      return np + k*score

def getCheckpointFile(iteration, is_small = False, stage = 3):
    if is_small:
        return'checkpoint_s' + str(iteration) + '.pth.tar'
    elif stage == 1:
        return 'checkpoint_s2_' + str(iteration) + '.pth.tar'
    elif stage == 2:
        return 'checkpoint_s3_' + str(iteration) + '.pth.tar'
    return 'checkpoint_' + str(iteration) + '.pth.tar'
    
def getcheckpointList(i, args):
    filename_list = []
    for k in range(0, i):
        filename = os.path.join(args.checkpoint, getCheckpointFile(k))
        if os.path.isfile(filename):
            filename_list.append(k)
    return filename_list
        
if __name__=="__main__":
    temp = copy.deepcopy(args.checkpoint)
    filename_list = []
    args.model = "residual_baseline"
    args.checkpoint = temp + args.model
    args.three_stages = False
    i, three_arr, mvp_three, iter_arr = loadLog(args)
    g = DraughtsGame(10)
    n1 = nn(g,args)
    nnet1 = {}
    nnet1['n1'] = n1
    filename_list.append(getcheckpointList(i, args))

    args3 = copy.deepcopy(args)
    args3.model = "residual_three_large"
    args3.large = True
    args3.three_stages = True
    args3.checkpoint = temp + args3.model
    nnet3 = {}
    n3 = nn(g,args3)
    nnet3['n1'] = n3
    args3.stage = 1
    nnet3['s2'] = nn(g, args3)
    filename_list.append(getcheckpointList(i, args3))
    
    args4 = copy.deepcopy(args)
    args4.model = "residual_mvp_three_large"
    args4.large = True
    args4.numMCTSSims = 10
    args4.budget_ratio = 8
    args4.three_stages = True
    args4.checkpoint = temp + args4.model
    nnet4 = {}
    n4 = nn(g,args4)
    nnet4['n1'] = n4   
    args4.model = 'small'
    nnet4['s1'] = nn(g, args4)
    args4.stage = 1
    nnet4['s3'] = nn(g, args4)
    args4.model = 'residual'
    nnet4['s2'] = nn(g, args4)
    filename_list.append(getcheckpointList(i, args4))

    elo_scores_three = three_arr[-1] if len(three_arr) > 0 else 0
    elo_scores_mvp_three = mvp_three[-1] if len(mvp_three) > 0 else 0
    start_i = iter_arr[-1] + 1 if len(iter_arr) > 0 else 1
    index0 = len(filename_list[0]) if len(filename_list[0]) < 50 else 50
    for k in range(start_i, index0):
        # nnet players
        filename = os.path.join(args.checkpoint, getCheckpointFile(filename_list[0][k]))
        #filename lists of checkpoints
        nnet1['n1'].load_checkpoint(folder=args.checkpoint, filename=getCheckpointFile(filename_list[0][k]))
        mcts1 = MCTS(g, nnet1, args)
        n1p = lambda v,w,x,y,z: np.argmax(mcts1.getActionProb(v,w,x,y,z, temp=0))
        if k < len(filename_list[1]):
            mcts1 = MCTS(g, nnet1, args)
            n1p = lambda v,w,x,y,z: np.argmax(mcts1.getActionProb(v,w,x,y,z, temp=0))
            nnet3['n1'].load_checkpoint(folder=args3.checkpoint, filename=getCheckpointFile(filename_list[1][k]))
            nnet3['s2'].load_checkpoint(folder=args3.checkpoint, filename=getCheckpointFile(filename_list[1][k], False, 1))
            mcts3 = MCTS(g, nnet3, args3)
            n3p = lambda v,w,x,y,z: np.argmax(mcts3.getActionProb(v,w,x,y,z, temp=0))
            arena = Arena.Arena(n1p, n3p, g)
            _, n3wins, draws3 = arena.playGames(args3.arenaCompare)
            elo_scores_three = calculate_elo(n3wins+draws3*0.5, args.arenaCompare, elo_scores_three, 0)
            three_arr.append(elo_scores_three)
        if k < len(filename_list[2]):
          mcts1 = MCTS(g, nnet1, args)
          n1p = lambda v,w,x,y,z: np.argmax(mcts1.getActionProb(v,w,x,y,z, temp=0))
          nnet4['n1'].load_checkpoint(folder=args4.checkpoint, filename=getCheckpointFile(filename_list[2][k]))
          nnet4['s1'].load_checkpoint(folder=args4.checkpoint, filename=getCheckpointFile(filename_list[2][k], True))
          nnet4['s2'].load_checkpoint(folder=args4.checkpoint, filename=getCheckpointFile(filename_list[2][k], False, 1))
          nnet4['s3'].load_checkpoint(folder=args4.checkpoint, filename=getCheckpointFile(filename_list[2][k], False, 2))
          mcts4 = MCTS(g, nnet4, args4)
          n4p = lambda v,w,x,y,z: np.argmax(mcts4.getActionProb(v,w,x,y,z, temp=0))
          arena = Arena.Arena(n1p, n4p, g)
          _, n4wins, draws4 = arena.playGames(args4.arenaCompare)
          elo_scores_mvp_three = calculate_elo(n4wins+draws4*0.5, args.arenaCompare, elo_scores_mvp_three, 0)
          mvp_three.append(elo_scores_mvp_three)
        iter_arr.append(k)
        saveArr([three_arr,mvp_three,iter_arr]) 
    index3 = len(three_arr) if len(three_arr) < 50 else 50
    index4 = len(mvp_three) if len(mvp_three) < 50 else 50      
    plt.plot(three_arr[0:index3],"--",label="three-stage")
    plt.plot(mvp_three[0:index4],"+-",label="mpv plus three-stage")
    plt.legend(loc=1,ncol=1)
    plt.title("Elo scores of MCTS")
    plt.xlabel("Iteration")
    plt.ylabel("Elo score")
    plt.savefig(args.checkpoint +"/"+str(i) + "modelsEloScores_large.jpg")
    plt.close()