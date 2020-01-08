import math
import numpy as np
import random
EPS = 1e-8
from DraughtsGame import display
from DraughtsLogic import Board
class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {'n1':{}, 's1':{}}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {'n1':{}, 's1':{}}      # stores #times edge s,a was visited
        self.Ns = {'n1':{}, 's1':{}}        # stores #times board s was visited
        self.Ps = {'n1':{}, 's1':{}}        # stores initial policy (returned by neural net)
        self.Rs = {'n1':{}, 's1':{}}        # stores initial values (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, curPlayer, curStep, jump_valids, noProgress, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        tree_ind = 'n1'
        if self.args.budget_ratio > 1:
            indices = self.args.numMCTSSims*(self.args.budget_ratio+1)
            tree_ind = 's1'
        else:
            indices = self.args.numMCTSSims
        for i in range(indices):
            tree = 'n1'
            if self.args.budget_ratio > 1 and i % self.args.budget_ratio != 0:
                tree = 's1'
            self.search(canonicalBoard, curPlayer, curStep, jump_valids, noProgress, tree)

        s = self.game.stringRepresentation(canonicalBoard, jump_valids[-1])
        counts = [self.Nsa[tree_ind][(s,a)] if (s,a) in self.Nsa[tree_ind] else 0 for a in range(self.game.getActionSize())]

        if temp==0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA]=1
            return probs

        counts = [x**(1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
        return probs


    def search(self, canonicalBoard, curPlayer, curStep, jump_valids, noProgress, sims, maxStep=150):
        """
        This function performs one iteration of MCTS until
        a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        states = []
        v = 0
        actions = []
        players = []
        curDepth = 0
        stage = 1
        while True:
            s = self.game.stringRepresentation(canonicalBoard, jump_valids[-1])
            pCount = curPlayer*self.game.getScore(canonicalBoard, 1)
            if s not in self.Es:
                self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
            if self.Es[s]!=0:
                # terminal node
                if abs(self.Es[s]) == 2:
                    # draw
                    v = 0
                else:
                    v = self.Es[s]
                break
            # exceeding number of maximal steps
            if curDepth >= 50 or noProgress > 21:
                v = curPlayer*pCount
                break
            if self.args.three_stages and stage != 3:
                stage = self.game.getStages(canonicalBoard)
            if s not in self.Ns[sims]:
                # leaf node
                if s not in self.Vs:
                    valids = self.game.getValidMoves(canonicalBoard, 1) if jump_valids[-1] == 1 else jump_valids
                    self.Vs[s] = valids
                valids = self.Vs[s]
                self.Ns[sims][s] = 0
                if sum(valids) != 1:
                    if self.args.three_stages and (stage == 1 or stage == 2):
                        if sims == 's1' and self.args.large:
                            temp_p, v = self.nnet['s3'].predict(self.game.getFeatureBoard(canonicalBoard,1,curStep,noProgress))
                        else:
                            temp_p, v = self.nnet['s2'].predict(self.game.getFeatureBoard(canonicalBoard,1,curStep,noProgress))
                        self.Ps[sims][s] = self.game.inflateActions(temp_p, canonicalBoard)

                    elif sims == 'n1':
                        self.Ps[sims][s], v = self.nnet['n1'].predict(self.game.getFeatureBoard(canonicalBoard,1,curStep,noProgress))
                    elif sims == 's1':
                        self.Ps[sims][s], v = self.nnet['s1'].predict(self.game.getFeatureBoard(canonicalBoard,1,curStep,noProgress))
                    self.Ps[sims][s] = self.Ps[sims][s]*valids      # masking invalid moves
                    sum_Ps_s = np.sum(self.Ps[sims][s])
                    if sum_Ps_s > 0:
                        self.Ps[sims][s] /= sum_Ps_s    # renormalize
                    else:
                        # if all valid moves were masked make all valid moves equally probable
                        
                        # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                        # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                        print("All valid moves were masked, do workaround.")
                        self.Ps[sims][s] = self.Ps[sims][s] + valids
                        self.Ps[sims][s] /= np.sum(self.Ps[sims][s])
                    self.Rs[sims][s] = v
                    # update shared values in MPV
                    if self.args.budget_ratio > 1 and (not self.args.three_stages or stage == 3 or self.args.large) and s in self.Ns['s1'] and s in self.Ns['n1']:
                        v = 0.5*self.Rs['s1'][s] + 0.5*self.Rs['n1'][s]
                        self.Rs['n1'][s] = v
                        self.Rs['s1'][s] = v
                        self.Ps['s1'][s] = self.Ps['n1'][s]
                    # update simmultaneously if stage 1 or 2 in hybrid method
                    elif self.args.three_stages and self.args.budget_ratio > 1 and stage != 3 and not self.args.large:
                        self.Rs['n1'][s] = v
                        self.Rs['s1'][s] = v
                        self.Ns['n1'][s] = 0
                        self.Ns['s1'][s] = 0
                        self.Ps['n1'][s] = self.Ps[sims][s]
                        self.Ps['s1'][s] = self.Ps[sims][s]
                else:
                    v = curPlayer*pCount
                break
      
            valids = self.Vs[s]
            cur_best = -float('inf')
            best_act = -1
            if sum(valids) == 1:
                best_act = np.where(valids==1)[0][0]
            else:
                if self.args.budget_ratio > 1 and sims == 'n1' and s in self.Ns['s1'] and self.Ns['s1'][s] > 0:
                    # pick the leaf with highest visited count in Ts that is not evaluated by Tl
                    for a in range(self.game.getActionSize()):
                        if valids[a]:
                            if (s,a) in self.Nsa['s1'] and (s,a) not in self.Nsa['n1']:
                                u = self.Nsa['s1'][(s,a)]
                                if u > cur_best:
                                    cur_best = u
                                    best_act = a                      
                if best_act == -1:
                    # pick the action with the highest upper confidence bound
                    for a in range(self.game.getActionSize()):
                        if valids[a]:
                            if (s,a) in self.Qsa[sims]:
                                u = self.Qsa[sims][(s,a)] + self.args.cpuct*self.Ps[sims][s][a]*math.sqrt(self.Ns[sims][s])/(1+self.Nsa[sims][(s,a)])
                            else:
                                u = self.args.cpuct*self.Ps[sims][s][a]*math.sqrt(self.Ns[sims][s] + EPS)     # Q = 0 ?
              
                            if u > cur_best:
                                cur_best = u
                                best_act = a
                actions.append(best_act)
                states.append(s)
                players.append(curPlayer)
      
            a = best_act
            next_s, _ = self.game.getNextState(canonicalBoard, 1, a)
            if pCount ==  curPlayer*self.game.getScore(next_s, 1):
                noProgress += 1
            else:
                noProgress = 0
            # flip board back after move
            next_s = self.game.getCanonicalForm(next_s,curPlayer)
            jump_valids = self.game.check_valid_jump(canonicalBoard, 1, a)
            if jump_valids[-1] == 1:
                # next player
                curPlayer = curPlayer*-1
                curStep += 1
                curDepth += 1
            # new canonical board
            canonicalBoard = self.game.getCanonicalForm(next_s, curPlayer)
      
        for i in range(len(states)):
            # update Q-values according to backpropagated v
            current_v = v*((-1)**(players[i]!=curPlayer))
            s = states[i]
            a = actions[i]
            if (s,a) in self.Qsa[sims]:
                self.Qsa[sims][(s,a)] = (self.Nsa[sims][(s,a)]*self.Qsa[sims][(s,a)] + current_v)/(self.Nsa[sims][(s,a)]+1)
                self.Nsa[sims][(s,a)] += 1
      
            else:
                self.Qsa[sims][(s,a)] = current_v
                self.Nsa[sims][(s,a)] = 1
      
            self.Ns[sims][s] += 1