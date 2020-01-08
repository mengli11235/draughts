""" Auxiliary files to generate examples for pretraining
    unused for final implementation"""


from collections import deque
import numpy as np
from DraughtsPlayers import *
from DraughtsGame import DraughtsGame as Game
from pickle import Pickler, Unpickler
from utils import *
import os
args = dotdict({
    'numEps': 100,
    'maxlenOfQueue': 200000,

    'load_folder_file': ('/data/s2651513/draughts/examples','gp5.examples'),
    'numItersForTrainExamplesHistory': 1,
})

def saveTrainExamples(iterationTrainExamples):
    folder = args.load_folder_file[0]
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, args.load_folder_file[1])
    with open(filename, "wb+") as f:
        Pickler(f).dump(iterationTrainExamples)
    f.closed

def executeEpisode(game, gp):
    trainExamples = []
    board = game.getInitBoard()
    curPlayer = 1
    episodeStep = 1
    noProgress = 0
    pCount = 0
    jump_valids = [0]*game.getActionSize()
    jump_valids[-1] = 1

    while True:
        canonicalBoard = game.getCanonicalForm(board,curPlayer)
        valids = game.getValidMoves(canonicalBoard,1) if jump_valids[-1] == 1 else jump_valids
        action = -1
        if sum(valids) == 1:
            action = np.where(valids==1)[0][0]
        else:
            action = gp(canonicalBoard, curPlayer, episodeStep, jump_valids, 3)
        pi = [0]*game.getActionSize()
        pi[action] = 1
        sym = game.getSymmetries(canonicalBoard, pi)
        for b,p,flip in sym:
            trainExamples.append([b, curPlayer, flip, p, episodeStep, noProgress, None])

        board, _ = game.getNextState(canonicalBoard, 1, action)
        if pCount == curPlayer*game.getScore(board,1):
            noProgress += 1
        else:
            noProgress = 0
            pCount = curPlayer*game.getScore(board,1)
        # flip board back
        board = game.getCanonicalForm(board, curPlayer)

        jump_valids = game.check_valid_jump(canonicalBoard, 1, action)
        if jump_valids[-1] == 1:
            curPlayer *= -1
            episodeStep += 1
        r = curPlayer*game.getGameEnded(game.getCanonicalForm(board, curPlayer), 1, episodeStep, noProgress)
        if r != 0:      
            if abs(r) == 2:
                # draw
                r = 0
            return [(game.getFeatureBoard(x[0],x[2],x[4],x[5]),x[3],x[2]*r*((-1)**(x[1]!=curPlayer))) for x in trainExamples]

if __name__=="__main__":
    g = Game(10)
    gp = AlphaBetaDraughtsPlayer(g).play
    iterationTrainExamples = []

    folder = args.load_folder_file[0]
    for eps in range(args.numEps):
        iterationTrainExamples += executeEpisode(g, gp)
    saveTrainExamples(iterationTrainExamples)