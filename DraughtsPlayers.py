import numpy as np

# the random player
class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board, curPlayer, curStep, jump_valids, noProgress):
        curPlayer = 1
        valids = self.game.getValidMoves(board, 1) if jump_valids[-1] == 1 else jump_valids
        a = np.random.randint(self.game.getActionSize())
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a

# the human player
class HumanDraughtsPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board, curPlayer, curStep, jump_valids, noProgress):

        valid = self.game.getValidMoves(board, 1) if jump_valids[-1] == 1 else jump_valids

        for i in range(len(valid)):
            if valid[i]:
                (x,y,new_x,new_y) = self.game.restoreMove([int(i/self.game.action2), i%self.game.action2])
                if curPlayer == -1:
                    x = self.game.n-1-x
                    y = self.game.n-1-y
                    new_x = self.game.n-1-new_x
                    new_y = self.game.n-1-new_y
                print((x,y,new_x,new_y))
        while True:
            a = input()

            x,y,new_x,new_y = [int(x) for x in a.split(' ')]
            if curPlayer == -1:
                x = self.game.n-1-x
                y = self.game.n-1-y
                new_x = self.game.n-1-new_x
                new_y = self.game.n-1-new_y
            [[reshape,reshape_new]] = self.game.reshapeMoves([[x,y,new_x,new_y]])
            a = self.game.action2*reshape+reshape_new if x!= -1 else self.game.action1*self.game.action2
            if valid[a]:
                break
            else:
                print('Invalid')

        return a

# the greedy player
class GreedyDraughtsPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board, curPlayer, curStep, jump_valids, noProgress):
        curPlayer = 1
        valids = self.game.getValidMoves(board, curPlayer) if jump_valids[-1] == 1 else jump_valids
        while sum(valids) == 1:
                a = np.where(valids==1)[0][0]
                board, _ = self.game.getNextState(board, curPlayer, a)
                jump_valids = self.game.check_valid_jump(canonicalBoard, 1, action)
                if jump_valids[-1] == 1:
                    curPlayer *= -1
                valids = self.game.getValidMoves(board, curPlayer) if jump_valids[-1] == 1 else jump_valids
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a]==0:
                continue
            nextBoard, _ = self.game.getNextState(board, curPlayer, a)
            score = self.game.getScore(nextBoard, curPlayer)
            if jump_valids[-1] == 1:
                score *= -1
            candidates += [(score, a)]
        candidates.sort()
        return candidates[0][1]

# Alpha-Beta Player
class AlphaBetaDraughtsPlayer():
    def __init__(self, game, depth=5):
        self.game = game
        self.depth = depth
    # wrapper
    def play(self, canonicalBoard, curPlayer, curStep, jump_valids, noProgress, alpha = -1000, beta = 1000):
        curPlayer = 1
        depth = self.depth
        return self.alphaBeta(canonicalBoard, curPlayer, curStep, jump_valids, depth, alpha, beta)[1]
    
    def alphaBeta(self, canonicalBoard, curPlayer, curStep, jump_valids, depth, alpha, beta):
        best_act = -1
        end = self.game.getGameEnded(canonicalBoard, 1)*curPlayer
        if end != 0:
            if abs(end) == 2:
                return (0, best_act)
            return (end, best_act)
        if depth == 0:
            return (self.game.getScore(canonicalBoard, 1)*curPlayer, -1)
        valids = self.game.getValidMoves(canonicalBoard, 1) if jump_valids[-1] == 1 else jump_valids
        move_list = np.where(valids==1)[0]
        for i in range(len(move_list)):
            a = move_list[i]
            board, _ = self.game.getNextState(canonicalBoard, 1, a)
            # flip board back
            board = self.game.getCanonicalForm(board, curPlayer)
            jump_valids = self.game.check_valid_jump(canonicalBoard, 1, a)
            new_depth = depth
            new_curPlayer = curPlayer
            new_curStep = curStep
            # if no valid jump
            if jump_valids[-1] == 1:
                new_curPlayer *= -1
                new_curStep += 1
                new_depth -= 1
            new_canonicalBoard = self.game.getCanonicalForm(board, new_curPlayer)
            temp = self.alphaBeta(new_canonicalBoard, new_curPlayer, new_curStep, jump_valids, new_depth, alpha, beta)[0]
            if curPlayer == 1 and temp > alpha:
                best_act = a
                alpha = temp
            if curPlayer == -1 and temp < beta:
                best_act = a
                beta = temp
            if (beta <= alpha):
                break
        if curPlayer == 1:
            return (alpha, best_act)
        if curPlayer == -1:
            return (beta, best_act)