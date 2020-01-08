import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys
sys.path.append('../')
from utils import *
from pytorch_classification.utils import Bar, AverageMeter
from NeuralNet import NeuralNet

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from .DraughtsNNet import MLP, CNN, SmallDraughtsNNet
from .DraughtsNNet import DraughtsNNet as onnet

# Neural Network Wrapper
class NNetWrapper(NeuralNet):
    def __init__(self, game, args):
        self.args = args
        # the model to create
        if args.model == 'mlp':
            print("Using MLP")
            self.nnet = MLP(game, args)
        elif args.model == 'cnn':
            print("Using CNN")
            self.nnet = CNN(game, args)
        elif args.model == 'small':
            print("Using Small Residual CNN")
            self.nnet = SmallDraughtsNNet(game, args)
        else:
            print("Using Residual CNN")
            self.nnet = onnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.board_y = int(self.board_y/2)
        self.action_size = game.getActionSize()

        if args.cuda:
            self.nnet.cuda()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        if self.args.model == 'mlp':
            optimizer = optim.SGD(self.nnet.parameters(),lr=0.01,momentum=0.0)
        else:
            optimizer = optim.Adam(self.nnet.parameters(), lr=0.001, weight_decay=10e-4)
        loss_sum = 0
        for epoch in range(self.args.epochs):
            print('EPOCH ::: ' + str(epoch+1))
            self.nnet.train()
            batch_idx = 0

            while batch_idx < int(len(examples)/self.args.batch_size):
                sample_ids = np.random.randint(len(examples), size=self.args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if self.args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v
                loss_sum += total_loss


                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                batch_idx += 1

        return loss_sum/len(examples)

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if self.args.cuda:
            board = board.contiguous().cuda()
        board = board.view(self.args.features, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets*outputs)/targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict' : self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        map_location = None  if self.args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
