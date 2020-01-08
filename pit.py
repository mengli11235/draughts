import Arena
from MCTS import MCTS
from DraughtsGame import DraughtsGame, display
from DraughtsPlayers import *
from pytorch.NNet import NNetWrapper as NNet
import torch
import numpy as np
from utils import *

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

})

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = DraughtsGame(10)

# all players
rp = RandomPlayer(g).play
gp = GreedyDraughtsPlayer(g).play
hp = HumanDraughtsPlayer(g).play
ap3 = AlphaBetaDraughtsPlayer(g,2).play

# nnet players
n1 = NNet(g,args)
n1.load_checkpoint('/data/s2651513/draughts/temp-residual0/','temp.pth.tar')
mcts1 = MCTS(g, n1, args)
n1p = lambda v,w,x,y,z: np.argmax(mcts1.getActionProb(v, w,x,y,z, temp=0))
#
#
#n2 = NNet(g,args)
#n2.load_checkpoint('/data/s2651513/draughts/temp-residual0/','temp.pth.tar')
#args2 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
#mcts2 = MCTS(g, n1, args)
#n2p = lambda v,w,x,y,z: np.argmax(mcts2.getActionProb(v, w,x,y,z, temp=0))

arena = Arena.Arena(ap2, n1, g, display=display)
print(arena.playGames(2, verbose=True))
