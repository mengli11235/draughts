from collections import deque
from Arena import Arena
from MCTS import MCTS
import numpy as np
from DraughtsPlayers import *
from DraughtsGame import DraughtsGame
from pytorch_classification.utils import Bar, AverageMeter
from pytorch.NNet import NNetWrapper as nn
import time, os, sys
from random import shuffle
import torch
from torch.backends import cudnn  
from pickle import Pickler, Unpickler
import matplotlib.pyplot as plt


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = {}
        self.pnet = {}
        self.nnet['n1'] = nnet
        self.pnet['n1'] = self.nnet['n1'].__class__(self.game, args)  # the competitor network
        # a small neural net to generate a small tree with budget_ratio times numMCTSsims
        if args.budget_ratio > 1:
            sargs = args
            sargs.model = 'small'
            self.nnet['s1'] = nn(game, sargs)
            self.pnet['s1'] = self.nnet['s1'].__class__(self.game, sargs)
        # two small neural nets for stage 1 (num of peices > 31), stage 2 (no kings, num of pieces > 8)
        if args.three_stages:
            sargs = args
            sargs.stage = 1
            if sargs.large:
                sargs.model = 'residual'
            else:
                sargs.model = 'small'
            self.nnet['s2'] = nn(game, sargs)
            self.pnet['s2'] = self.nnet['s2'].__class__(self.game, sargs)
        if args.three_stages and args.budget_ratio > 1 and args.large:
            sargs = args
            sargs.stage = 1
            sargs.model = 'small'
            self.nnet['s3'] = nn(game, sargs)
            self.pnet['s3'] = self.nnet['s3'].__class__(self.game, sargs)

        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.loss_dicts = []
        self.skipFirstSelfPlay = False # can be overriden in loadTrainExamples()

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (featureBoard,pi,v,...)
                           pi is the MCTS informed policy vector, v is from [-1, 1].
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 1
        noProgress = 0
        pCount = 0
        jump_valids = [0]*self.game.getActionSize()
        jump_valids[-1] = 1
        # track the three stages
        stage1 = 0
        stage2 = 0
        stage3 = 0

        while True:
            canonicalBoard = self.game.getCanonicalForm(board,self.curPlayer)
            stage = self.game.getStages(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard,1) if jump_valids[-1] == 1 else jump_valids
            action = -1
            if sum(valids) == 1:
                action = np.where(valids==1)[0][0]
            else:
                if stage == 1:
                    stage1 += 2
                    stage2 += 2
                    stage3 += 2
                elif stage == 2:
                    stage2 += 2
                    stage3 += 2
                elif stage == 3:
                    stage3 += 2
                temp = int(episodeStep < self.args.tempThreshold)
                # call the MCTS
                pi = self.mcts.getActionProb(canonicalBoard, self.curPlayer, episodeStep, jump_valids, noProgress, temp=temp)
                action = np.random.choice(len(pi), p=pi)
                # flip the color, double examples
                sym = self.game.getSymmetries(canonicalBoard, pi)
                for b,p,flip in sym:
                    trainExamples.append([b, self.curPlayer, flip, p, episodeStep, noProgress , None])

            board, _ = self.game.getNextState(canonicalBoard, 1, action)
            # track the number of no progress states
            if pCount == self.curPlayer*self.game.getScore(board,1):
                noProgress += 1
            else:
                noProgress = 0
                pCount = self.curPlayer*self.game.getScore(board,1)
            # flip board back
            board = self.game.getCanonicalForm(board,self.curPlayer)

            jump_valids = self.game.check_valid_jump(canonicalBoard, 1, action)
            if jump_valids[-1] == 1:
                self.curPlayer *= -1
                episodeStep += 1
            r = self.game.getGameEnded(self.game.getCanonicalForm(board, self.curPlayer), 1, episodeStep, noProgress)
            if r != 0:      
                if abs(r) == 2:
                    # draw
                    r = 0
                print(r)
                return ([(self.game.getFeatureBoard(x[0],x[2]*x[1],x[4],x[5]),x[3],x[2]*r*((-1)**(x[1]!=self.curPlayer))) for x in trainExamples], stage1, stage2, stage3)

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples.
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        iter_arr = []
        rp_arr = []
        loss_arr = []
        elo_scores_rp = 0
        start_i = self.loadLog()
        # examples of the iteration
        self.loadTrainExamples(start_i)
        for i in range(start_i, self.args.numIters+1):
            # bookkeeping
            print('------ITER ' + str(i) + '------')
            if not self.skipFirstSelfPlay or i>1:
                iterationTrainExamples1 = []
                iterationTrainExamples2 = []
                iterationTrainExamples3 = []
                iterationTrainExamples = []
    
                for eps in range(self.args.numEps):
                    self.mcts = MCTS(self.game, self.nnet, self.args)   # reset search tree
                    e, stage1, stage2, stage3 = self.executeEpisode()
                    iterationTrainExamples.extend(e)
                    # slice stages into example histories
                    for board, pi, r in e[0:stage1+1]:
                        compressed_pi = self.game.compressActions(pi)
                        iterationTrainExamples1.append((board, compressed_pi, r))
                    if stage2 > stage1:
                        for board, pi, r in e[stage1+1:stage2+1]:
                            compressed_pi = self.game.compressActions(pi)
                            iterationTrainExamples2.append((board, compressed_pi, r))
                    if stage3 > stage2:
                        iterationTrainExamples3.extend(e[stage2+1:stage3+1])

                # save the iteration examples to the history 
                self.trainExamplesHistory.append((iterationTrainExamples, iterationTrainExamples1, iterationTrainExamples2, iterationTrainExamples3))
                
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory), " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)
            
            # shuffle examples before training
            trainExamples = []
            trainExamples_stage1 = []
            for es0, es1, es2, es3 in self.trainExamplesHistory:
                trainExamples.extend(es0)
                trainExamples_stage1.extend(es1)
                trainExamples_stage1.extend(es2)
            shuffle(trainExamples)
            shuffle(trainExamples_stage1)

            # training new network, keeping a copy of the old one
            self.nnet['n1'].save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet['n1'].load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            if self.args.budget_ratio > 1:
                self.nnet['s1'].save_checkpoint(folder=self.args.checkpoint, filename='temps.pth.tar')
                self.pnet['s1'].load_checkpoint(folder=self.args.checkpoint, filename='temps.pth.tar')
            if self.args.three_stages:
                self.nnet['s2'].save_checkpoint(folder=self.args.checkpoint, filename='temps2.pth.tar')
                self.pnet['s2'].load_checkpoint(folder=self.args.checkpoint, filename='temps2.pth.tar')
            if self.args.three_stages and self.args.budget_ratio > 1 and self.args.large:
                self.nnet['s3'].save_checkpoint(folder=self.args.checkpoint, filename='temps3.pth.tar')
                self.pnet['s3'].load_checkpoint(folder=self.args.checkpoint, filename='temps3.pth.tar')
            nmcts = MCTS(self.game, self.nnet, self.args)
            pmcts = MCTS(self.game, self.pnet, self.args)
            loss_dict = {}
            if self.args.three_stages:
                loss_dict['total_loss'] = self.nnet['n1'].train(trainExamples)
                loss_dict['s2total_loss'] = self.nnet['s2'].train(trainExamples_stage1)
                if self.args.budget_ratio > 1:
                    loss_dict['stotal_loss'] = self.nnet['s1'].train(trainExamples)
                if self.args.budget_ratio > 1 and self.args.large:
                    loss_dict['s3total_loss'] = self.nnet['s3'].train(trainExamples_stage1)
            else:
                loss_dict['total_loss'] = self.nnet['n1'].train(trainExamples)
                if self.args.budget_ratio > 1:
                    loss_dict['stotal_loss'] = self.nnet['s1'].train(trainExamples)
            self.loss_dicts.append(loss_dict)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i-1)

            print('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda v, w,x,y,z: np.argmax(pmcts.getActionProb(v, w, x, y, z, temp=0)),
                          lambda v, w,x,y,z: np.argmax(nmcts.getActionProb(v, w, x, y, z, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins+nwins == 0 or float(nwins)/(pwins+nwins) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                self.nnet['n1'].load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
                if self.args.budget_ratio > 1:
                    self.nnet['s1'].load_checkpoint(folder=self.args.checkpoint, filename='temps.pth.tar')
                if self.args.three_stages:
                    self.nnet['s2'].load_checkpoint(folder=self.args.checkpoint, filename='temps2.pth.tar')
                    if self.args.budget_ratio > 1 and self.args.large:
                        self.nnet['s3'].load_checkpoint(folder=self.args.checkpoint, filename='temps3.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                self.saveLog(i)

            loss_arr.append(loss_dict['total_loss'])
            if self.args.plot and i%5 == 0:
                plt.plot([x for x in loss_arr])
                plt.title("Total loss")
                plt.xlabel("Iteration")
                plt.ylabel("pi_loss+v_loss")
                plt.savefig(self.args.checkpoint +"/"+str(i) + "loss.jpg")
                plt.close()      

    def getCheckpointFile(self, iteration, is_small = False, stage = 3):
        if is_small:
            return'checkpoint_s' + str(iteration) + '.pth.tar'
        elif stage == 1:
            return 'checkpoint_s2_' + str(iteration) + '.pth.tar'
        elif stage == 2:
            return 'checkpoint_s3_' + str(iteration) + '.pth.tar'
        return 'checkpoint_' + str(iteration) + '.pth.tar'
    
    # save the training examples
    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration)+".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory[-1])
        f.closed
        filename = os.path.join(folder, str(self.args.three_stages)+".loss")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.loss_dicts)
        f.closed
    
    # save the log and temp checkpoints
    def saveLog(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.nnet['n1'].save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(iteration))
        self.nnet['n1'].save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar') 
        if self.args.budget_ratio > 1:
            self.nnet['s1'].save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(iteration, True))
            self.nnet['s1'].save_checkpoint(folder=self.args.checkpoint, filename='bests.pth.tar')
        if self.args.three_stages:
            self.nnet['s2'].save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(iteration, False, 1))
            self.nnet['s2'].save_checkpoint(folder=self.args.checkpoint, filename='bests2.pth.tar')
            if self.args.budget_ratio > 1 and self.args.large:
                self.nnet['s3'].save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(iteration, False, 2))
                self.nnet['s3'].save_checkpoint(folder=self.args.checkpoint, filename='bests3.pth.tar')
        filename = os.path.join(folder, "log.txt")
        with open(filename, "w+") as f:
            f.write(str(iteration))
        f.closed
        
    # load the log and checkpoints
    def loadLog(self):
        iteration = 0
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, "log.txt")
        if os.path.isfile(filename):
            with open(filename, "r") as f:
                iteration = int(f.read())
            f.closed
            print("Resumed training from the "+str(iteration)+" iteration.")
            self.nnet['n1'].load_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(iteration))
            if self.args.budget_ratio > 1:
                self.nnet['s1'].load_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(iteration, True))
            if self.args.three_stages:
                self.nnet['s2'].load_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(iteration, False, 1))
                if self.args.budget_ratio > 1 and self.args.large:
                    self.nnet['s3'].load_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(iteration, False, 2))
        return iteration+1
        
    #load training examples
    def loadTrainExamples(self,i):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        examplesFiles = []
        if self.args.pretrain and i == 1:
            examplesFiles[0] = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
        elif i > 1 and self.args.numItersForTrainExamplesHistory>1:
            for k in range(max(0,i-self.args.numItersForTrainExamplesHistory), i-1):
                filename = os.path.join(folder, self.getCheckpointFile(k)+".examples")
                if os.path.isfile(filename):
                    examplesFiles.append(filename)
        for examplesFile in examplesFiles:
            if not os.path.isfile(examplesFile):
                print(examplesFile)
                r = input("File with trainExamples not found. Continue? [y|n]")
                if r != "y":
                    sys.exit()
            else:
                print("File with trainExamples found. Read it.")
                with open(examplesFile, "rb") as f:
                    self.trainExamplesHistory.append(Unpickler(f).load())
                f.closed
        filename = os.path.join(folder, str(self.args.three_stages)+".loss")
        if os.path.isfile(filename):
            with open(filename, "rb") as f:
                print("Loss history found. Read it.")
                self.loss_dicts = Unpickler(f).load()
            f.closed
    
    def calculate_elo(self, score, matches, np, opponent=0, k=32):
        #calculate ELO
        score -= matches*1/(1+10**((opponent-np)/400))
        return np + k*score
