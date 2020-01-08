# list of all move directions on the board, from offsets
NORTHWEST, NORTHEAST, SOUTHWEST, SOUTHEAST  = 0, 1, 2, 3
DIRECTIONS = [NORTHWEST, NORTHEAST, SOUTHWEST, SOUTHEAST]
# coordinate changes
COORS = [[-1, -1], # NorthWest
         [-1,  1], # NorthEast
         [ 1, -1], # SouthWest
         [ 1,  1]] # SouthEast
                   
class Board():

    def __init__(self, n):
        "sets up initial board configuration."

        self.n = n
        # Create the empty board array.
        self.pieces = [None]*self.n
        for i in range(self.n):
            self.pieces[i] = [0]*self.n

        # Set up the initial pieces.
        rows = range(int(self.n/2+1), n)
        for row in rows:
            cols = range(0, self.n, 2) if row % 2 == 1 else range(1, self.n, 2)
            for col in cols:
                self.pieces[row][col] = 1
        rows = range(0, int(self.n/2-1))
        for row in rows:
            cols = range(0, self.n, 2) if row % 2 == 1 else range(1, self.n, 2)
            for col in cols:
                self.pieces[row][col] = -1

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.pieces[index]

    def countDiff(self, color):
        """counts the diffrences of pieces of the given color
        by heuristics and return a value [-1, 1]"""
        count = 0
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y]*color > 0 and abs(self[x][y]) != 3:
                    count += abs(self[x][y])
                elif self[x][y]*color < 0  and abs(self[x][y]) != 3:
                    count -= abs(self[x][y])
                elif abs(self[x][y]) == 3:
                    count += 1
        return count/((self.n/2-1)*(self.n/2)*2)

    def get_legal_moves(self, curPlayer, is_jump=False):
        """returns all the legal moves for the given color.
        """
        moves = set()  # stores the legal moves.

        # Get all the squares with pieces of the given color..
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] > 0:
                    newmoves = self.get_moves_for_square((x,y), curPlayer, is_jump)
                    if len(newmoves)>0:
                        moves.update(newmoves)
        return list(moves)

    def has_legal_moves(self, curPlayer):
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] > 0:
                    newmoves = self.get_moves_for_square((x,y), curPlayer)
                    if len(newmoves)>0:
                        return True
        return False

    def get_moves_for_square(self, square, curPlayer, is_jump=False):
        """returns all the legal moves that use the given square as a base.
        """
        (x,y) = square

        # skip empty source squares.
        if curPlayer==0:
            return None
        
        # search all possible directions.
        directions = []
        moves = []
        if abs(self[x][y]) == 2:
            directions = DIRECTIONS
        elif is_jump:
            directions = DIRECTIONS
        elif curPlayer == -1:
            directions = [SOUTHWEST, SOUTHEAST]
        else:
            directions = [NORTHWEST, NORTHEAST]
    
        for direction in directions:
            move = self._discover_move(square, direction)
            if move:
                if isinstance(move, list):
                    for single_move in move:
                        if not (is_jump and len(self._get_captures(single_move)) == 0):
                            moves.append(single_move)
                else:
                    if not (is_jump and len(self._get_captures(move)) == 0):
                        moves.append(move)

        # return the generated move list
        return moves

    def execute_move(self, move, curPlayer):
        """executes the given move on the board; captures pieces as necessary.
        """

        # Add the piece to the empty square.
        # print(move)
        (x, y, new_x, new_y) = move
        has_jump = False
        captures = self._get_captures(move)
        self[new_x][new_y] = self[x][y]
        self[x][y] = 0
        for x, y in captures:
            self[x][y] = self[new_x][new_y]/abs(self[new_x][new_y])*3
        if len(captures)>0:
            newmoves = self.get_moves_for_square((new_x,new_y), curPlayer, True)
            if len(newmoves)>0:
                has_jump = True
        if not has_jump:
            #if self[new_x][new_y] == self.check_crowned(move, curPlayer):
            for y in range(self.n):
                for x in range(self.n):
                    if abs(self[x][y]) == 3:
                        self[x][y] = 0
            self[new_x][new_y] = self.check_crowned(move, curPlayer)

    def _discover_move(self, origin, direc):
        """ finds moves given a starting square"""
        x, y = origin
        direction = COORS[direc]
        color = self[x][y]
        move = []
        
        ## If the piece is king
        if abs(color) == 2:
            for row in range(1, self.n):
                x_direc = row*direction[0]
                y_direc = row*direction[1]
                new_x = x+x_direc
                new_y = y+y_direc
                if (0 <= new_x < self.n and 0 <= new_y < self.n):
                    # Blocked by pieces, stop
                    if self[new_x][new_y]*color > 0:
                        break
                    elif self[new_x][new_y]*color < 0:
                        new_x += direction[0]
                        new_y += direction[1]
                        while 0 <= new_x < self.n and 0 <= new_y < self.n and self[new_x][new_y] == 0:
                            move.append((x, y, new_x, new_y))
                            new_x += direction[0]
                            new_y += direction[1]
                        break                            
                    else:
                        move.append((x, y, new_x, new_y))
            return move
        else:
            ## If the piece is not king
            new_x = x+direction[0]
            new_y = y+direction[1]
            # Failure, out of bound or already taken
            if not (0 <= new_x < self.n and 0 <= new_y < self.n) or self[new_x][new_y] * color > 0:
                return None
            # Success, empty space
            elif self[new_x][new_y] == 0:
                return (x, y, new_x, new_y)
            # Capture
            else:
                new_x += direction[0]
                new_y += direction[1]
                # Failure, out of bound or no space
                if not (0 <= new_x < self.n and 0 <= new_y < self.n) or self[new_x][new_y] != 0:
                    return None
                # Success
                else:
                    return (x, y, new_x, new_y)

    def _get_captures(self, move):
        """ gets the list of captures for a given move """
        #initialize variables
        (x, y, new_x, new_y) = move
        color = self[x][y]
        captures = []
        distance = abs(x-new_x)
        if distance == 1:
            return captures
        direc_x = int((new_x - x) / distance)
        direc_y = int((new_y - y) / distance)

        while distance > 1:
            x += direc_x
            y += direc_y
            if self[x][y] * color < 0:
                captures.append((x, y))
            distance -= 1

        return captures

    def check_crowned(self, move, curPlayer):
        # check whether the men is crowned
        x, y, new_x, new_y = move
        color = self[new_x][new_y]
        if new_x == 0 and curPlayer == 1:
            return int(2*color/abs(color))
        elif new_x == self.n-1 and curPlayer == -1:
            return int(2*color/abs(color))
        else:
            return color