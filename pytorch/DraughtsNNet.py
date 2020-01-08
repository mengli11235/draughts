import sys
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# large ResNet
class DraughtsNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.board_y = int(self.board_y/2)
        self.features = args.features
        self.action_size = game.getActionSize() if not args.three_stages or args.stage == 3 else 201
        self.args = args

        super(DraughtsNNet, self).__init__()
        self.conv1 = nn.Conv2d(self.features, args.num_channels, (3,3), stride=1, padding=1, bias=True)
        self.residual_blocks = self.__make_residual_blocks(args)
        self.value_head = Value_head(game, args, 128)
        self.policy_head = Policy_head(game, args)
        self.bn1 = nn.BatchNorm2d(args.num_channels)

    def __make_residual_blocks(self, args):
        blocks = []
        for _ in range(10):
            blocks.append(Residual_block(args))
        return nn.Sequential(*blocks)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, self.features, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)), inplace=True)                          # batch_size x num_channels x board_x x board_y
        s = self.residual_blocks(s)

        pi = self.policy_head(s)                                                                         # batch_size x action_size
        v = self.value_head(s)                                                                          # batch_size x 1
        return F.log_softmax(pi, dim=1), torch.tanh(v)

# small ResNet
class SmallDraughtsNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.board_y = int(self.board_y/2)
        self.features = args.features
        # the action size for small CNN
        self.action_size = game.getActionSize() if not args.three_stages or args.stage == 3 else 201
        self.args = args

        super(SmallDraughtsNNet, self).__init__()
        self.conv1 = nn.Conv2d(self.features, args.num_channels, (3,3), stride=1, padding=1, bias=True)
        self.residual_blocks = self.__make_residual_blocks(args)
        self.value_head = Value_head(game, args, 64)
        self.policy_head = Policy_head(game, args)
        self.bn1 = nn.BatchNorm2d(args.num_channels)

    def __make_residual_blocks(self, args):
        blocks = []
        for _ in range(5):
            blocks.append(Residual_block(args))
        return nn.Sequential(*blocks)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, self.features, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)), inplace=True)                          # batch_size x num_channels x board_x x board_y
        s = self.residual_blocks(s)
        # s = s.view(-1, self.args.num_channels*self.board_x*self.board_y)

        pi = self.policy_head(s)                                                                         # batch_size x action_size
        v = self.value_head(s)                                                                          # batch_size x 1
        return F.log_softmax(pi, dim=1), torch.tanh(v)

class Residual_block(nn.Module):
    def __init__(self, args):
        self.args = args
        super(Residual_block, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, args.num_channels, (3,3), stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, (3,3), stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, s):
        identity = s
        s = self.conv1(s)
        s = self.bn1(s)
        s = self.relu1(s)
        s = self.conv2(s)
        s = self.bn2(s)
        s += identity
        s = self.relu2(s)
        return s
       
class Policy_head(nn.Module):
    def __init__(self, game, args):
        self.board_x, self.board_y = game.getBoardSize()
        self.board_y = int(self.board_y/2)
        self.action_size = game.getActionSize() if not args.three_stages or args.stage == 3 else 201
        self.args = args
        super(Policy_head, self).__init__()
        self.conv = nn.Conv2d(args.num_channels, 
    					 		  out_channels=2,
    					 		  kernel_size=(1, 1),
    					 		  stride=1,
    					 		  padding=0,
    					 		  bias=True)
        self.bn = nn.BatchNorm2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_features=2*self.board_x*self.board_y, 
    							out_features=self.action_size,
    							bias=True)

    def forward(self, s):
        s = self.conv(s)
        s = self.bn(s)
        s = self.relu(s)
        s = self.fc(s.view(s.size(0), -1))
        return s


class Value_head(nn.Module):
    def __init__(self, game, args, num_features):
        self.board_x, self.board_y = game.getBoardSize()
        self.board_y = int(self.board_y/2)
        self.args = args
        super(Value_head, self).__init__()
        nb_filters_value_head = 1
        self.conv = nn.Conv2d(args.num_channels, 
    					 		  out_channels=1,
    					 		  kernel_size=(1, 1),
    					 		  stride=1,
    					 		  padding=0,
    					 		  bias=True)
        self.bn = nn.BatchNorm2d(1)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(in_features=1*self.board_x*self.board_y, 
    							 out_features=num_features, 
    							 bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=num_features, 
    							 out_features=1,
    							 bias=True)

    def forward(self, s):
        s = self.conv(s)
        s = self.bn(s)
        s = self.relu1(s)
        s = self.fc1(s.view(s.size(0), -1))
        s = self.relu2(s)
        s = self.fc2(s)
        return s

# CNN without residual blocks
class CNN(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.board_y = int(self.board_y/2)
        self.features = args.features
        self.action_size = game.getActionSize() if not args.three_stages or args.stage == 3 else 201
        self.args = args

        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(self.features, args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)

        self.fc1 = nn.Linear(args.num_channels*(self.board_x-4)*(self.board_y-4), 256)
        self.fc_bn1 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, self.action_size)

        self.fc4 = nn.Linear(256, 1)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, self.features, self.board_x, self.board_y)                
        s = F.relu(self.bn1(self.conv1(s)))                          
        s = F.relu(self.bn2(self.conv2(s)))                         
        s = F.relu(self.bn3(self.conv3(s)))                         
        s = F.relu(self.bn4(self.conv4(s)))                          
        s = s.view(-1, self.args.num_channels*(self.board_x-4)*(self.board_y-4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training) 


        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)

# MLP
class MLP(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.board_y = int(self.board_y/2)
        self.features = args.features
        self.action_size = game.getActionSize() if not args.three_stages or args.stage == 3 else 201
        self.args = args
        
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(self.features*self.board_x*self.board_y, 512)
        self.fc_bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, self.action_size)
        self.fc4 = nn.Linear(256, 1)

        
    def forward(self, s):
        s = s.view(-1, self.features*self.board_x*self.board_y)
        s = F.dropout(F.sigmoid(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.sigmoid(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512
        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)

        return F.log_softmax(pi, dim=1), torch.tanh(v)                                                                         