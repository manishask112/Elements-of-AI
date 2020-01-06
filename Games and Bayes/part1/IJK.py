#!/usr/local/bin/python3

"""
@author: Abhilash Kuhikar, October 2019
./ijk.py : Interface to an IJK game.
Usage:
    ./ijk.py player1 player2 mode
where:
    player1 and player2 are either "ai" or "human"
    mode is either "det" for deterministic or "nondet" for nondetermistic
You may want to modify this file for debugging purposes, but all
of your AI logic should be in solve_IJK.py.
"""

import time, copy
import random
import sys
import ai_IJK
import human_IJK

from logic_IJK import Game_IJK, initialGame

def IJK(player1, player2, deterministic, timeout=60, max_moves=30000):
    game = initialGame(6, '+', deterministic)

    game.printGame()

    moves = []
    while not game.state() != 0 and len(moves) < max_moves:
        start = time.time()
        move = None
        
        for result in player1(copy.deepcopy(game),True):
            end = time.time()
            if end-start > timeout:
                break
            else:
                move = result
        if not move:
            print ('player + forfeits the game')
            break
        else:
            print ("%s played: %s" % (game.getCurrentPlayer(), move))

            moves.append((game.getCurrentPlayer(), move))
            game = game.makeMove(move)
            
            game.printGame()

        start = time.time()
        move = None
        for result in player2(copy.deepcopy(game),False):
            end = time.time()
            if end-start > timeout:
                break
            else:
                move = result
        if not move:
            print ('player - forfeits the game')
            break
        else:
            print ("%s played: %s" % (game.getCurrentPlayer(), move))
            
            moves.append((game.getCurrentPlayer(), move))
            game = game.makeMove(move)
            
            game.printGame()

    print('Winner was: ', game.state())
    return moves

if __name__ == "__main__":
    if(len(sys.argv) != 4):
        raise Exception('Please provide three commandline arguments.')
    
    (p1, p2, mode) = sys.argv[1:]
    
    logics = { "human" : human_IJK.next_move, "ai" : ai_IJK.next_move }
    deterministic = { "det" : True, "nondet" : False }

    IJK(logics[p1], logics[p2], deterministic[mode])
