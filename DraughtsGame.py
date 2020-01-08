""" the game engine """

from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from DraughtsLogic import Board
import numpy as np


class DraughtsGame(Game):
    def __init__(self, n, max_steps=150):
        self.n = n
        self.n_direc = 4
        self.max_steps = max_steps
        self.features = 8
        self.action1 = int(self.n**2/2)
        self.action2 = int(self.n**2/2)

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.action1*self.action2+1

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.action1*self.action2:
            print("Invalid Action")
            return (board, player)
        b = Board(self.n)
        b.pieces = np.copy(board)
        move = self.restoreMove([int(action/self.action2), action%self.action2])
        b.execute_move(move, player)
        return (b.pieces, -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)
        # get possible jumps
        legalMoves =  b.get_legal_moves(player, True)
        has_capture = True
        # if no jump, get all moves
        if len(legalMoves)==0:
            legalMoves = b.get_legal_moves(player)
            has_capture = False
        MaxLegalMoves = []
        # no legal move
        if len(legalMoves)==0:
            valids[-1]=1
            return np.array(valids)
        popMoves = []
        # must choose moves with the highest amount of captures
        if len(legalMoves) > 1 and has_capture:
            maximal_jumps = []
            for move in legalMoves:
                maximal_jumps.append(self.getMaximalJumps(b.pieces, player, move))
            maximal_ind = np.where(maximal_jumps==np.max(maximal_jumps))[0]
            for i in maximal_ind:
                MaxLegalMoves.append(legalMoves[i])
        else:
            MaxLegalMoves = legalMoves
        reshaped_legalMoves = self.reshapeMoves(MaxLegalMoves)
        for x, y in reshaped_legalMoves:
            valids[self.action2*x+y]=1
        return np.array(valids)
    
    def getStages(self, board):
        # get the current stage
        oneKings = 0
        twoKings = 0
        onePawns = 0
        twoPawns = 0
        for j in board:
            for i in j:
                if i == 2:
                    oneKings += 1
                if i == -2:
                    twoKings += 1
                if i == 1:
                    onePawns += 1
                if i == -1:
                    twoPawns += 1
        total = oneKings + twoKings + onePawns + twoPawns
        if total > 31:
            return 1
        if oneKings + twoKings == 0 and total > 8:
            return 2
        return 3
    
    def getGameEnded(self, board, player, steps=0, noprogress=0):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # or return heuristic values
        b1 = Board(self.n)
        b1.pieces = np.copy(board)
        if steps > self.max_steps or noprogress > 20:
            x = self.getScore(board, player)
            if x == 0:
                x = 2
            return x
        elif not b1.has_legal_moves(player):
            return -player

        return 0

    def getGameRealEnded(self, board, player, steps=0, noprogress=0):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        b1 = Board(self.n)
        b1.pieces = np.copy(board)
        if steps > self.max_steps or noprogress > 20:
            return 2
        elif not b1.has_legal_moves(player):
            return -player

        return 0

    def getCanonicalForm(self, board, player):
        if player == -1:        
            return np.rot90(board, 2)*-1
        return board
    
    def checkCapture(self, board, player, action):
        b = Board(self.n)
        b.pieces = np.copy(board)
        move = self.restoreMove([int(action/self.action2), action%self.action2])
        if len(b._get_captures(move))>0:
            return True
        return False
    
    def check_valid_jump(self, board, player, action):
        # find out if consecutive jumps are valid
        b = Board(self.n)
        b.pieces = np.copy(board)
        moves = set()
        move = self.restoreMove([int(action/self.action2), action%self.action2])
        (x, y, new_x, new_y) = move
        if len(b._get_captures(move))>0:
            # promoted pieces has king right only after next turn
            old_pos = b[x][y]
            b.execute_move(move, player)
            b[new_x][new_y] = old_pos
            newmoves = b.get_moves_for_square((new_x,new_y), player, True)
            if len(newmoves)>0:
                moves.update(newmoves)
        legalMoves = []
        move_list = list(moves)
        valids = [0]*self.getActionSize()
        if len(move_list)==0:
            valids[-1]=1
            return np.array(valids)
        if len(move_list) > 1:
            maximal_jumps = []
            for move in move_list:
                maximal_jumps.append(self.getMaximalJumps(b.pieces, player, move))
            maximal_ind = np.where(maximal_jumps==np.max(maximal_jumps))[0]
            for i in maximal_ind:
                legalMoves.append(move_list[i])
        else:
            legalMoves = move_list
        reshaped_legalMoves = self.reshapeMoves(legalMoves)
        for x, y in reshaped_legalMoves:
            valids[self.action2*x+y]=1
        return np.array(valids)
    
    def getMaximalJumps(self, board, player, move):
        # called by check_valid_jump(), record the maximal jump
        maximal_jump = 0
        move_list = [[move]]
        board_list = [board]
        while len(move_list)>0:
            next_move_list = []
            next_board_list = []
            for i in range(len(move_list)):
                k = move_list[i]
                board = board_list[i]
                for move in k:
                    moves = set()
                    b = Board(self.n)
                    b.pieces = np.copy(board)
                    (x, y, new_x, new_y) = move
                    b.execute_move(move, player)
                    newmoves = b.get_moves_for_square((new_x,new_y), player, True)
                    if len(newmoves)>0:
                        moves.update(newmoves)
                    if len(list(moves))>0:
                        next_move_list.append(list(moves))
                        next_board_list.append(b.pieces)
            move_list = next_move_list
            board_list = next_board_list
            maximal_jump += 1
        return maximal_jump
    
    def flipPolicy(self, pi):
        # flip the policy accordingly when flip the color
        newPi = np.zeros(len(pi))
        for i in range(len(pi)-1):
            x,y,new_x,new_y = self.restoreMove((int(i/self.action2), i%self.action2))
            x = self.n-1-x
            new_x = self.n-1-new_x
            y = self.n-1-y
            new_y = self.n-1-new_y
            [[reshape, reshape_new]] = self.reshapeMoves([(x,y,new_x,new_y)])
            newPi[self.action2*reshape+reshape_new] = pi[i]      
        return newPi

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.action1*self.action2+1)
        newPi = self.flipPolicy(pi)
        newB = np.rot90(board, 2)*-1
        l = []
        l += [(board, pi, 1)]
        l += [(newB, newPi, -1)]
        return l

    def stringRepresentation(self, board, is_not_iump):
        # reshape board first, attach jump flag
        reshaped_board = np.array(self.reshape_board(board))
        return reshaped_board.tostring()+np.array([is_not_iump]).tostring()

    def getScore(self, board, player):
        # heuristics
        b = Board(self.n)
        b.pieces = np.copy(board)
        return b.countDiff(player)
    
    def getFeatureBoard(self, board, curPlayer, totalMoves, NoProgress):
        # get the feature board of 8 channels
        featue_board = [None]*self.features
        for i in range(self.features):
            featue_board[i] = [None]*self.n
            for j in range(self.n):
                featue_board[i][j] = [0]*self.n
        b1 = Board(self.n)
        b1.pieces = np.copy(board)
        for i in range(self.n):
            for j in range(self.n):
                if b1.pieces[i][j] == 1:
                    featue_board[0][i][j] = 1
                if b1.pieces[i][j] == -1:
                    featue_board[1][i][j] = 1
                if b1.pieces[i][j] == 2:
                    featue_board[2][i][j] = 1
                if b1.pieces[i][j] == -2:
                    featue_board[3][i][j] = 1
                if abs(b1.pieces[i][j]) == 3:
                    featue_board[4][i][j] = 1
                if curPlayer == 1:
                    featue_board[5][i][j] = 1
                featue_board[6][i][j] = totalMoves
                featue_board[7][i][j] = NoProgress
                    
        reshaped_featue_board = [None]*self.features
        for i in range(self.features):
            reshaped_featue_board[i] = self.reshape_board(featue_board[i])
        return np.array(reshaped_featue_board)
        
    def reshapeMoves(self, moves):
    # reshape the moves into smaller with board size 10*5
        reshaped_moves = []
        for x, y, new_x, new_y in moves:
            reshape = 0
            reshape_new = 0
            if x % 2 == 0:
                assert y%2 == 1
                reshape = x*5 + int((y-1)/2)
            else:
                assert y%2 == 0
                reshape = x*5 + int(y/2)
            if new_x % 2 == 0:
                assert new_y%2 == 1
                reshape_new = new_x*5 + int((new_y-1)/2)
            else:
                assert new_y%2 == 0
                reshape_new = new_x*5 + int(new_y/2)
            reshaped_moves.append([reshape,reshape_new])
        return reshaped_moves

    def compressActions(self, probs):
        # compress the probability vector by 10*5*4
        probs = probs[:-1]
        actions = [0] * 201
        for move in range(len(probs)):
            reshape = int(move / self.action2)
            reshape_new = move % self.action2
            x = int(reshape/5)
            y = 0
            new_x = int(reshape_new/5)
            new_y = 0
            if x % 2 == 0:
                y = 2*(reshape%5)+1
            else:
                y = reshape%5*2
            if new_x % 2 == 0:
                new_y = 2*(reshape_new%5)+1
            else:
                new_y = reshape_new%5*2
            __, __, direc, __ = self.coor_to_direc((x, y, new_x, new_y))
            
            actions[reshape*self.n_direc+direc] = probs[move]
        return actions

    def restoreMove(self, move):
        # restore the move within larger board 10*10
        reshape, reshape_new = move
        x = int(reshape/5)
        y = 0
        new_x = int(reshape_new/5)
        new_y = 0
        if x % 2 == 0:
            y = 2*(reshape%5)+1
        else:
            y = reshape%5*2
        if new_x % 2 == 0:
            new_y = 2*(reshape_new%5)+1
        else:
            new_y = reshape_new%5*2
        return x,y,new_x,new_y
        
    def inflateActions(self, probs, pieces):
        # reshape the probability vector into 10*5^2
        probs = probs[:-1]
        actions = [0]*self.getActionSize()
        for move in range(len(probs)):
            reshape = int(move / self.n_direc)
            direc = move % self.n_direc
            x = int(reshape/5)
            y = 0
            dist = 1
            if x % 2 == 0:
                y = 2*(reshape%5)+1
            else:
                y = reshape%5*2
            __, __, new_x, new_y = self.direc_to_coor((x, y, direc, dist))
            if 0 <= new_x < self.n and 0 <= new_y < self.n:
                if pieces[new_x][new_y] != 0:
                    dist = 2
                    __, __, new_x, new_y = self.direc_to_coor((x, y, direc, dist))
                if 0 <= new_x < self.n and 0 <= new_y < self.n:
                    reshape_new = 0
                    if new_x % 2 == 0:
                        assert new_y%2 == 1
                        reshape_new = new_x*5 + int((new_y-1)/2)
                    else:
                        assert new_y%2 == 0
                        reshape_new = new_x*5 + int(new_y/2)
                    actions[self.action2*reshape+reshape_new] = probs[move]
        return actions
    
    def reshape_board(self, pieces):
        # reshape the board into 10*5
        reshaped_pieces = [None]*self.n
        for i in range(self.n):
            reshaped_pieces[i] = [0]*int(self.n/2)
        for x in range(self.n):
            for y in range(self.n):
                if x % 2 == 0 and y % 2 == 1:
                    reshaped_pieces[x][int((y-1)/2)] = pieces[x][y]
                elif  x % 2 == 1 and y % 2 == 0:
                    reshaped_pieces[x][int(y/2)] = pieces[x][y]

        return reshaped_pieces

    def restore_board(self, reshaped_pieces):
        # restore the board into 10*10
        pieces = [None]*self.n
        for i in range(self.n):
            pieces[i] = [0]*self.n
        for x in self.n:
            for y in self.n:
                if x % 2 == 0 and y % 2 == 1:
                    pieces[x][y] = reshaped_pieces[x][int((y-1)/2)]
                elif  x % 2 == 1 and y % 2 == 0:
                    pieces[x][y] = reshaped_pieces[x][int(y/2)]

        return pieces
        
    def direc_to_coor(self, move):
        # change direction representation into coordinates
        (x, y, direc, dist) = move
        direc_x = dist
        direc_y = dist
        if direc == 0:
            direc_x *= -1
            direc_y *= -1
        elif direc == 1:
            direc_x *= -1
        elif direc == 2:
            direc_y *= -1
        new_x = x+direc_x
        new_y = y+direc_y

        return (x, y, new_x, new_y)
        
    def coor_to_direc(self, move):
        # change coordinates representation into direction 
        (x, y, new_x, new_y) = move
        direc = -1
        dist_x = new_x - x
        dist_y = new_y - y
        if abs(dist_x) != abs(dist_y):
            assert("Distance must be equal")
        if dist_x < 0 and dist_y < 0:
            direc = 0
        elif dist_x < 0 and dist_y > 0:
            direc = 1
        elif dist_x > 0 and dist_y < 0:
            direc = 2
        elif dist_x > 0 and dist_y > 0:
            direc = 3
        else:
            assert("No direction")
        return (x, y, direc, abs(dist_x))

def display(board):
    # display the board
    n = board.shape[0]

    for y in range(n):
        print (y,"|",end="")
    print("")
    print(" -----------------------")
    for y in range(n):
        print(y, "|",end="")    # print the row #
        for x in range(n):
            piece = board[y][x]    # get the piece to print
            if piece == -1: print("b ",end="")
            elif piece == -2: print("bK ",end="")
            elif piece == 1: print("W ",end="")
            elif piece == 2: print("WK ",end="")
            elif abs(piece) == 3: print("G ",end="")
            else:
                if x==n:
                    print("-",end="")
                else:
                    print("- ",end="")
        print("|")

    print("   -----------------------")
