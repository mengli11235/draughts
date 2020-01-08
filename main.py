from Coach import Coach
from DraughtsGame import DraughtsGame as Game
from pytorch.NNet import NNetWrapper as nn
from utils import *
import torch
from optparse import OptionParser

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
    'model': 'residual',
    'features': 8,
    'num_channels': 64,
    
    'budget_ratio': 1,
    'three_stages': False,
    'stage': 3,
    'large': False

})

if __name__=="__main__":
    parser = OptionParser()
    parser.add_option("-m", "--model", action="store", type="string", dest="model")
    parser.add_option("-c", "--checkpoint", action="store", type="string", dest="checkpoint")
    parser.add_option("-n", "--numMCTSSims", action="store", type="int", dest="numMCTSSims")
    parser.add_option("-b", "--budget_ratio", action="store", type="int", dest="budget_ratio")
    parser.add_option("-p", "--pretrain", action="store_true", dest="pretrain")
    parser.add_option("-l", "--load", action="store_true", dest="load_model")
    parser.add_option("-g", "--large", action="store_true", dest="large")
    parser.add_option("-t", "--three_stages", action="store_true", dest="three_stages")
    (options, __) = parser.parse_args()
    args.checkpoint = options.checkpoint if options.checkpoint!=None else "/data/s2651513/draughts/temp-"
    args.model = options.model if options.model!=None else "residual"
    args.numMCTSSims = options.numMCTSSims if options.numMCTSSims!=None else 50
    args.budget_ratio = options.budget_ratio if options.budget_ratio!=None else 1
    args.checkpoint += args.model
    args.pretrain = options.pretrain if options.pretrain!=None else False
    args.three_stages = options.three_stages if options.three_stages!=None else False
    args.load_model = options.load_model if options.load_model!=None else False
    args.large = options.large if options.large!=None else False

    g = Game(10)
    print(args.cuda)
    nnet = nn(g, args)
    c = Coach(g, nnet, args)
    c.learn()
