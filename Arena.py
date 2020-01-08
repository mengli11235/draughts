import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """
    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it. Is necessary for verbose
                     mode.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 1
        noProgress = 0
        pCount = 0
        jump_valids = [0]*self.game.getActionSize()
        jump_valids[-1] = 1
        while True:
            if verbose:
                assert(self.display)
                print("Turn ", str(it), "Player ", str(curPlayer))
                if jump_valids[-1] != 1:
                    print("Keep jumping!")
                self.display(board)
            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer),1) if jump_valids[-1] == 1 else jump_valids
            if sum(valids) == 1:
                if verbose:
                    print("Mandatory jump:")
                action = np.where(valids==1)[0][0]
            else:
                action = players[curPlayer+1](self.game.getCanonicalForm(board, curPlayer), curPlayer, it, jump_valids, noProgress)
            if verbose:
                (x,y,new_x,new_y) = self.game.restoreMove([int(action/self.game.action2), int(action%self.game.action2)])
                if curPlayer == -1:
                    x = self.game.n-1-x
                    y = self.game.n-1-y
                    new_x = self.game.n-1-new_x
                    new_y = self.game.n-1-new_y
                print((x,y,new_x,new_y))
            if valids[action]==0:
                print(action)
                assert valids[action] > 0
            canonicalBoard = self.game.getCanonicalForm(board, curPlayer)
            board, _ = self.game.getNextState(canonicalBoard, 1, action)
            if pCount == curPlayer*self.game.getScore(board,1):
                noProgress += 1
            else:
                noProgress = 0
                pCount = curPlayer*self.game.getScore(board,1)
            # flip board back
            board = self.game.getCanonicalForm(board, curPlayer)
            jump_valids = self.game.check_valid_jump(canonicalBoard, 1, action)
            if jump_valids[-1] == 1:
                curPlayer *= -1
                it += 1
            if self.game.getGameEnded(self.game.getCanonicalForm(board, curPlayer), 1, it, noProgress)!=0:
                break
        if verbose:
            assert(self.display)
            print("Game over: Turn ", str(it), "Result ", str(curPlayer*self.game.getGameEnded(self.game.getCanonicalForm(board, curPlayer), 1, it, noProgress)))
            self.display(board)
        return curPlayer*self.game.getGameEnded(self.game.getCanonicalForm(board, curPlayer), 1, it, noProgress)

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num/2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in range(num):
            gameResult = self.playGame(verbose=verbose)
            print(gameResult)
            if abs(gameResult) == 2:
                draws+=1
            elif gameResult==1:
                oneWon+=1
            elif gameResult==-1:
                twoWon+=1
            else:
                draws+=1

        self.player1, self.player2 = self.player2, self.player1
        
        for _ in range(num):
            gameResult = self.playGame(verbose=verbose)
            print(gameResult)
            # count the game result
            if abs(gameResult) == 2:
                draws+=1
            elif gameResult==-1:
                oneWon+=1                
            elif gameResult==1:
                twoWon+=1
            else:
                draws+=1

        return oneWon, twoWon, draws
