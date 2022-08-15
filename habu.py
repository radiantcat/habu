import time
from ctypes import *

# Load NNUE probe and init weights
nnue = cdll.LoadLibrary('./libnnueprobe.so')
nnue.nnue_init(b'./nn.nnue')

#
# Chess constants
#

PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6
EMPTY = 13

A1, B1, C1, D1, E1, F1, G1, H1 = 91, 92, 93, 94, 95, 96, 97, 98
A2, H2 = 81, 88
A8, H8 = 21, 28
last_rank = [28, 27, 26, 25, 24, 23, 22, 21]

fen_pieces = 'PNBRQKpnbrqk'

# Directions for generating moves on a 10x12 board
N, E, S, W = -10, 1, 10, -1

directions = {
    PAWN: (N, N+N, N+W, N+E),
    KNIGHT: (N+N+E, E+N+E, E+S+E, S+S+E, S+S+W, W+S+W, W+N+W, N+N+W),
    BISHOP: (N+E, S+E, S+W, N+W),
    ROOK: (N, E, S, W),
    QUEEN: (N, E, S, W, N+E, S+E, S+W, N+W),
    KING: (N, E, S, W, N+E, S+E, S+W, N+W)
}
directions_isatt = {
    BISHOP + 6: (N+E, S+E, S+W, N+W),
    ROOK + 6: (N, E, S, W),
    KNIGHT + 6: (N+N+E, E+N+E, E+S+E, S+S+E, S+S+W, W+S+W, W+N+W, N+N+W),
    PAWN + 6: (N+E, N+W),
    KING + 6: (N, E, S, W, N+E, S+E, S+W, N+W)
}

# 10x12 board
mailbox = [
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, 56, 57, 58, 59, 60, 61, 62, 63, -1,
    -1, 48, 49, 50, 51, 52, 53, 54, 55, -1,
    -1, 40, 41, 42, 43, 44, 45, 46, 47, -1,
    -1, 32, 33, 34, 35, 36, 37, 38, 39, -1,
    -1, 24, 25, 26, 27, 28, 29, 30, 31, -1,
    -1, 16, 17, 18, 19, 20, 21, 22, 23, -1,
    -1,  8,  9, 10, 11, 12, 13, 14, 15, -1,
    -1,  0,  1,  2,  3,  4,  5,  6,  7, -1, 
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
]

mailbox64 = [
    98, 97, 96, 95, 94, 93, 92, 91,
    88, 87, 86, 85, 84, 83, 82, 81,
    78, 77, 76, 75, 74, 73, 72, 71,
    68, 67, 66, 65, 64, 63, 62, 61,
    58, 57, 56, 55, 54, 53, 52, 51,
    48, 47, 46, 45, 44, 43, 42, 41,
    38, 37, 36, 35, 34, 33, 32, 31,
    28, 27, 26, 25, 24, 23, 22, 21
]

square_name = [
    'a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1',
    'a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2',
    'a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'h3',
    'a4', 'b4', 'c4', 'd4', 'e4', 'f4', 'g4', 'h4',
    'a5', 'b5', 'c5', 'd5', 'e5', 'f5', 'g5', 'h5',
    'a6', 'b6', 'c6', 'd6', 'e6', 'f6', 'g6', 'h6',
    'a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7',
    'a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8'
]

char_to_piece = {
    'K' : KING,
    'Q' : QUEEN,
    'R' : ROOK,
    'B' : BISHOP,
    'N' : KNIGHT,
    'P' : PAWN,
    'k' : KING + 6,
    'q' : QUEEN + 6,
    'r' : ROOK + 6,
    'b' : BISHOP + 6,
    'n' : KNIGHT + 6,
    'p' : PAWN + 6
}

#
# Search constants
#

# Only for ordering captures
piece_vals = {PAWN: 1, KNIGHT: 2, BISHOP: 3, ROOK: 5, QUEEN: 9, KING: 0}

MATE_SCORE = 200000

HIST_MAX = 10000

TT_MAX = 1e6

TT_EXACT = 1
TT_UPPER = 2
TT_LOWER = 3

FUTILITY_MARGIN = 180
RAZOR_MARGIN = 500
ASPIRATION_DELTA = 30

#
# Misc functions
#

def my_piece(piece):
    return piece >= 1 and piece <= 6
def opp_piece(piece):
    return piece >= 7 and piece <= 12
def is_pnk(piece):
    return piece == PAWN or piece == KNIGHT or piece == KING

def invert_sides(board):
    for i, piece in enumerate(board):
        if opp_piece(piece):
            board[i] -= 6
        elif my_piece(piece):
            board[i] += 6

#
# Game class and chess logic
#

class Game():
    def __init__(self, brd, s, wc, wlc, bc, blc, eps, pos):
        self.board = brd # 10x12 board
        self.side = s # White/Black

        self.w_castle = wc #Castle rights
        self.w_lcastle = wlc
        self.b_castle = bc
        self.b_lcastle = blc

        self.enp = eps # En Passant square
        
        self.positions = pos

    def copy(self):
        return Game(self.board.copy(), self.side, self.w_castle, self.w_lcastle, self.b_castle, self.b_lcastle, self.enp, self.positions)

    def initial_pos(self):
        self.parse_fen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
        
    def uci_move(self, move):
        move_str = square_name[mailbox[move[0]]] + square_name[mailbox[move[1]]] if self.side else \
            square_name[mailbox[119 - move[0]]] + square_name[mailbox[119 - move[1]]] 
        if self.board[move[0]] == PAWN and move[1] in last_rank:
            move_str += 'q'
        return move_str

    def parse_fen(self, fen):
        self.board = [0] * 120
        self.positions = []
        self.side = True
        self.enp = 0

        fen_split = fen.split()
        
        idx = 63
        for c in fen_split[0]:
            if c in '12345678':
                for i in range(int(c)):
                    self.board[mailbox64[idx]] = EMPTY
                    idx -= 1
            elif c == '/': continue
            else:
                self.board[mailbox64[idx]] = char_to_piece[c]
                idx -= 1
        if fen_split[1] == 'b':
            self.rotate()

        self.w_castle = 'K' in fen_split[2]
        self.w_lcastle = 'Q' in fen_split[2]
        self.b_castle = 'k' in fen_split[2]
        self.b_lcastle = 'q' in fen_split[2]

        self.positions.append(self.to_fen())

    def to_fen(self): # For NNUE evaluation
        fen = ''
        cpy = self.copy()
        if not self.side:
            cpy.rotate()
        cnt = 0
        empties = 0
        breaks = 0
        for piece in cpy.board:
            if piece == 0: continue
            if piece == EMPTY:
                empties += 1
            else:
                if empties != 0: 
                    fen += str(empties)
                    empties = 0
                fen += fen_pieces[piece - 1]
            cnt += 1
            if cnt % 8 == 0:
                breaks += 1
                if empties != 0: 
                    fen += str(empties)
                    empties = 0
                if breaks != 8: fen += '/'
        fen += ' w ' if self.side else ' b '
        fen += '- -'

        return fen

    def rotate(self):
        invert_sides(self.board)
        self.board.reverse()

        self.w_castle, self.w_lcastle, self.b_castle, self.b_lcastle = \
            self.b_castle, self.b_lcastle, self.w_castle, self.w_lcastle

        self.enp = 119 - self.enp
        self.side = not self.side

    def movegen(self): # Move generation using directions
        movelist = []
        for sq, piece in enumerate(self.board):
            if not my_piece(piece): continue
            for d in directions[piece]:
                next_sq = sq
                while True:
                    next_sq += d
                    # If next square is our piece or off the board
                    if self.board[next_sq] <= 6: break

                    if piece == PAWN:
                        # Pawns cannot capture when going directly forward
                        if d == N and self.board[next_sq] != EMPTY: break
                        if d == N + N and (self.board[next_sq] != EMPTY or self.board[next_sq - N] != EMPTY \
                            or sq < A2 or sq > H2) : break
                        if (d == N + W or d == N + E) and not (opp_piece(self.board[next_sq]) or self.enp == next_sq): break
                    movelist.append((sq, next_sq))
                    if opp_piece(self.board[next_sq]): break
                    
                    # Break for a non sliding piece
                    if is_pnk(piece): break
                    
                    #Castling rights
                    if sq == A1 and self.w_lcastle:
                        if self.board[next_sq + E] == KING and not (self.is_attacked(next_sq + E) or self.is_attacked(next_sq) or self.is_attacked(next_sq + W)):
                            movelist.append((next_sq + E, next_sq + W))
                    elif sq == H1 and self.w_castle:
                        if self.board[next_sq + W] == KING and not (self.is_attacked(next_sq + E) or self.is_attacked(next_sq) or self.is_attacked(next_sq + W)):
                            movelist.append((next_sq + W, next_sq + E))

        return movelist

    def is_attacked(self, sq):
        for piece, dirs in directions_isatt.items():
            for d in dirs:
                next_sq = sq
                while True:
                    next_sq += d
                    if self.board[next_sq] <= 6: break
                    if self.board[next_sq] == piece: return True
                    if (piece == BISHOP + 6 or piece == ROOK + 6) and self.board[next_sq] == QUEEN + 6: return True
                    if opp_piece(self.board[next_sq]): break
                    if is_pnk(piece - 6): break
        return False

    def is_capture(self, move):
        return opp_piece(self.board[move[1]])

    def in_check(self):
        for i in range(120):
            if self.board[i] == KING:
                if self.is_attacked(i): return True
        return False
    
    def make_move(self, move):
        fr, to = move
        enpcpy = self.enp
        self.enp = 0

        if self.w_lcastle and fr == A1:
            self.w_lcastle = False
        elif self.w_castle and fr == H1:
            self.w_castle = False

        if self.board[fr] == KING:
            self.w_castle = self.w_lcastle = False
            if fr == E1:
                if to == C1:
                    self.board[A1] = EMPTY
                    self.board[D1] = ROOK
                elif to == G1:
                    self.board[H1] = EMPTY
                    self.board[F1] = ROOK
            elif fr == D1:
                if to == B1:
                    self.board[A1] = EMPTY
                    self.board[C1] = ROOK
                elif to == F1:
                    self.board[H1] = EMPTY
                    self.board[E1] = ROOK
        self.board[to] = self.board[fr]

        if self.board[fr] == PAWN:
            if to in last_rank:
                self.board[to] = QUEEN
            if to == fr + N + N:
                self.enp = fr + N
            if to == enpcpy:
                self.board[to + S] = EMPTY

        self.board[fr] = EMPTY

    def make_uci_move(self, move):
        for m in self.movegen():
            if move == self.uci_move(m):
                self.make_move(m)
                self.rotate()
                self.positions.append(self.to_fen())
                break

    def uci_position(self, command):
        is_move = False
        for part in command.split()[1:]:
            if is_move:
                self.make_uci_move(part)
            elif part == 'startpos':
                self.initial_pos()
            elif part == 'fen': 
                self.parse_fen(command.split('fen')[1].split('moves')[0])
            elif part == 'moves':
                is_move = True

    def evaluate_nnue(self):
        return nnue.nnue_evaluate_fen(bytes(self.positions[-1], encoding='utf-8'))

    def has_non_pawns(self):
        return any(p in self.board for p in [KNIGHT, BISHOP, ROOK, QUEEN])

#
# Search algorithm
#

class Searcher:
    def __init__(self):
        self.nodes = 0
        self.b_move = 0
        self.finish_time = 0

        self.history = {} # History table for move ordering
        self.counter_hist = {}
        self.counter_move = {}
        self.killer = {} # Killer move for each ply
        self.killer2 = {}
        self.tt = {} # Transposition table

        self.start_depth = 0

    def reset_timer(self, tt):
        self.finish_time = time.perf_counter() + tt

    def move_values(self, game, movelist, ply, opp_move, hash_move):
        # Give every move a score for ordering
        killer_move = self.killer.get(ply)
        killer_move2 = self.killer2.get(ply)
        counter = self.counter_move.get(opp_move)
        scores = {}
        for move in movelist:
            if move == hash_move:
                scores[move] = 2 * HIST_MAX + 5000
            elif game.is_capture(move):
                scores[move] = 2 * HIST_MAX + 2000 + piece_vals[game.board[move[1]] - 6] - piece_vals[game.board[move[0]]]
            else:
                if move == killer_move:
                    scores[move] = 2 * HIST_MAX + 3
                elif move == killer_move2:
                    scores[move] = 2 * HIST_MAX + 2
                elif move == counter:
                    scores[move] = 2 * HIST_MAX + 1
                else:
                    scores[move] = self.history.get(move, 0)
                    scores[move] += self.counter_hist.get((opp_move, move), 0)

        return scores

    def qsearch(self, game, alpha, beta):
        val = game.evaluate_nnue()

        self.nodes += 1

        if val >= beta:
            return val
        if alpha < val:
            alpha = val

        movelist = [m for m in game.movegen() if game.is_capture(m)]
        scores = self.move_values(game, movelist, 0, None, None)
        movelist.sort(key = lambda x: scores[x], reverse=True)

        for move in movelist:
            
            cpy = game.copy()
            cpy.make_move(move)
            if cpy.in_check():
                continue
            cpy.rotate()
            cpy.positions.append(cpy.to_fen())
            self.nodes += 1

            score = -self.qsearch(cpy, -beta, -alpha)

            cpy.positions.pop()
            if score > val:
                val = score
                if score > alpha:
                    alpha = score
                    if score >= beta:
                        break

        return val

    def search(self, game, depth, alpha, beta, ply, do_pruning, opp_move, root = False):

        repetitions = 0
        if not root:
            for pos in game.positions:
                if pos == game.positions[-1]:
                    repetitions += 1
                if repetitions > 1: return 0

        if depth <= 0:
            return self.qsearch(game, alpha, beta)
        
        self.nodes += 1

        hash_move = None

        is_pv_node = beta - alpha != 1

        # Probe transposition table
        tt_entry = self.tt.get(game.positions[-1])
        if tt_entry:
            hash_move = tt_entry[1]
            if tt_entry[2] >= depth and not is_pv_node:
                if tt_entry[3] == TT_EXACT or \
                 (tt_entry[3] == TT_LOWER and tt_entry[0] >= beta) or \
                 (tt_entry[3] == TT_UPPER and tt_entry[0] <= alpha):
                    return tt_entry[0]

        if self.nodes % 10000 == 0: 
            if len(self.tt) > TT_MAX: # Clear transposition table if too many entries
                self.tt.clear()

        in_check = game.in_check()
        evalu = game.evaluate_nnue()

        if not is_pv_node and do_pruning and not in_check:
            # Razoring
            if depth <= 3 and evalu + RAZOR_MARGIN < beta:
                score = self.qsearch(game, alpha, beta)
                if score < beta:
                    return score

            # Futility Pruning
            if depth <= 6 and evalu >= beta + FUTILITY_MARGIN * depth:
                return evalu

            # Null move pruning 
            if depth >= 2 and evalu >= beta and game.has_non_pawns():
                cpy = game.copy()
                cpy.enp = 0
                cpy.rotate()
                reduction = 3
                cpy.positions.append(cpy.to_fen())
                score = -self.search(cpy, depth - reduction, -beta, -beta + 1, ply + 1, False, None)
                cpy.positions.pop()
                if score >= beta:
                    return score

        best_move = None
        best_score = -MATE_SCORE
        legal_moves = 0
        tt_flag = TT_UPPER

        # Generate and sort moves
        movelist = game.movegen()
        scores = self.move_values(game, movelist, ply, opp_move, hash_move)
        movelist.sort(key = lambda x: scores[x], reverse=True)

        for move in movelist:            
            if self.nodes % 50000 == 0 and self.start_depth > 1 and time.perf_counter() > self.finish_time:
                break
            # Copy-Make
            cpy = game.copy()
            cpy.make_move(move)
            if cpy.in_check():
                continue
            legal_moves += 1
            cpy.rotate()
            cpy.positions.append(cpy.to_fen())

            check_move = cpy.in_check()
            # Check extension
            ext = 1 if check_move and not root else 0
            
            # Principal Variation Search
            score = None
            if legal_moves == 1:
                score = -self.search(cpy, depth - 1 + ext, -beta, -alpha, ply + 1, True, move)
            else:
                # Late Move Reductions for quiet moves
                reduction = 0
                if depth >= 3 and legal_moves > 3 and not in_check and not check_move and not game.is_capture(move):
                    reduction = 1 

                score = -self.search(cpy, depth - 1 - reduction + ext, -alpha - 1, -alpha, ply + 1, True, move)
                if score > alpha and reduction != 0:
                    score = -self.search(cpy, depth - 1 + ext, -alpha - 1, -alpha, ply + 1, True, move)
                if score > alpha and score < beta:
                    score = -self.search(cpy, depth - 1 + ext, -beta, -alpha, ply + 1, True, move)
            cpy.positions.pop()
            if score > best_score:
                best_score = score
                best_move = move
                if score > alpha:
                    tt_flag = TT_EXACT
                    alpha = best_score       
                    if score >= beta:
                        if not game.is_capture(move): # Update history tables
                            self.history[move] = min(HIST_MAX, self.history.get(move, 0) + depth * depth)
                            if opp_move: 
                                self.counter_move[opp_move] = move
                                self.counter_hist[(opp_move, move)] = min(HIST_MAX, self.counter_hist.get((opp_move, move), 0) + depth * depth)
                            k, k2 = self.killer.get(ply), self.killer2.get(ply)
                            if k != k2:
                                self.killer2[ply] = k
                                self.killer[ply] = move
                            else: self.killer[ply] = move
                        tt_flag = TT_LOWER
                        for m in movelist:
                            if m == move: break
                            self.history[m] = max(-HIST_MAX, self.history.get(m, 0) - depth * depth)
                            if opp_move:
                                self.counter_hist[(opp_move, m)] = max(-HIST_MAX, self.counter_hist.get((opp_move, m), 0) - depth * depth)
                        break

        if legal_moves == 0:
            if game.in_check():
                return -MATE_SCORE + ply
            else:
                return 0

        if root:
            self.b_move = best_move

        # Update transposition table
        self.tt[game.positions[-1]] = (best_score, best_move, depth, tt_flag)

        return best_score

    def search_iterative(self, game):
        start_time = time.perf_counter()

        # Reset history tables
        self.history = {}
        self.counter_hist = {}
        self.counter_move = {}
        self.killer = {}
        self.killer2 = {}
        self.tt = {}

        move = None
        self.nodes = 0
        time_1 = time.perf_counter()

        alpha, beta = -MATE_SCORE, MATE_SCORE
        d = 1
        asp_cnt = 0

        # Iterative Deepening with Aspiration Windows
        while d < 80:
            self.start_depth = d
            score = self.search(game, d, alpha, beta, 0, True, None, True)

            if time.perf_counter() > self.finish_time and d > 1:
                break

            time_2 = time.perf_counter()
            nps = int(self.nodes / (time_2 - time_1))
            time_elapsed = int((time.perf_counter() - start_time) * 1000)
            
            if score <= alpha:
                asp_cnt += 1
                alpha -= ASPIRATION_DELTA * 2**asp_cnt
                print('info depth {} time {} nodes {} nps {} score cp {}'.format(d, time_elapsed, self.nodes, nps, score))
                continue
            elif score >= beta:
                asp_cnt += 1
                beta += ASPIRATION_DELTA * 2**asp_cnt
                print('info depth {} time {} nodes {} nps {} score cp {}'.format(d, time_elapsed, self.nodes, nps, score))
                continue

            move = self.b_move

            print('info depth {} time {} nodes {} nps {} score cp {} pv {}'.format(d, time_elapsed, self.nodes, nps, score, game.uci_move(move)))
            if d >= 4:
                alpha = score - ASPIRATION_DELTA
                beta = score + ASPIRATION_DELTA

            asp_cnt = 0
            d += 1

        print(f'bestmove {game.uci_move(move)}')


def main():
    g = Game(None, True, True, True, True, True, 0, [])
    g.initial_pos()
    s = Searcher()

    while True:
        command = input()

        if command == 'uci':
            print('id name Habu')
            print('uciok')

        elif command == 'isready':
            print('readyok')

        elif command.startswith('go'):
            move_time = 10
            c_split = command.split()
            for idx, val in enumerate(c_split):
                if val == 'wtime' and g.side:
                    move_time = int(c_split[idx + 1]) / 40000
                elif val == 'btime' and not g.side:
                    move_time = int(c_split[idx + 1]) / 40000
            s.reset_timer(move_time)
            s.search_iterative(g)

        elif command.startswith('position'):
            g.uci_position(command)

        elif command == 'quit':
            return

if __name__ == '__main__':
    main()


