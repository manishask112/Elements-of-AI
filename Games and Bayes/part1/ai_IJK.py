#!/usr/local/bin/python3

"""
This is where you should write your AI code!
Authors: Manisha Suresh Kumar(msureshk), Deepthi Raghu(draghu), Uma Maheswari Gollapudi(ugollap)
Based on skeleton code by Abhilash Kuhikar, October 2019
"""

from logic_IJK import Game_IJK
import random
from numpy.core.numeric import inf
import copy
# Suggests next move to be played by the current player given the current game
#
# inputs:
#     game : Current state of the game 
#
# This function should analyze the current state of the game and determine the 
# best move for the current player. It should then call "yield" on that move.

DEPTH=6
PLAYER_=True

#FUNCTION THAT RETURNS A LIST OF COORDINATES OF FREE SPACES
def count_free_spaces(board):
    coordinates_of_free_space=[]
    for i in range(0,6):
        for j in range(0,6):
            if board[i][j]==' ':
                coordinates_of_free_space.append((i,j))            
    return coordinates_of_free_space

#FUNCTION TO CALCULATE SCORES FOR HIGHEST LETTER IN THE CORNER FOR A GIVEN CONFIGURATION
def edge_check(mat):
    i=0
    j=0
    highest_letter='a'
    highest_letter_opponent='a'
    score=0
    letter={'a':1,'b':2,'c':3,'d':4,'e':7,'f':8,'g':9,'h':10,'i':11,'j':20,'k':25}
    vertices=[(0,0),(0,5),(5,0),(5,5)]
    player={True:mat[i][j].isupper(),False:mat[i][j].islower()} 
    for i in range(0,6):
        for j in range(0,6): 
            if player[PLAYER_] and mat[i][j].lower()>highest_letter:
                highest_letter=mat[i][j].lower()
            elif not player[PLAYER_] and mat[i][j].lower()>highest_letter_opponent:
                highest_letter_opponent=mat[i][j].lower()
    for i in range(0,6):
        for j in range(0,6):
            if  mat[i][j]!=' ' and player[PLAYER_]:
                if mat[i][j].lower()==highest_letter and (i,j) in vertices:
                    score=score+(2*letter[mat[i][j].lower()])
            elif mat[i][j]!=' ' and not player[PLAYER_]:
                if mat[i][j].lower()==highest_letter_opponent and (i,j) in vertices:
                    score=score-(2*letter[mat[i][j].lower()])
    return score            

#FUNCTION TO CALCULATE EVALUATION FUNCTION
def Result(game):
    letter={'a':1,'b':2,'c':3,'d':4,'e':7,'f':8,'g':9,'h':10,'i':11,'j':20,'k':22,' ':0}
    score=0
    i=0
    j=0
    mat=game.getGame()
    player={True:mat[i][j].isupper(),False:mat[i][j].islower()}   
    highest_letter='a'
    highest_letter_opponent='a' 
    
    score=score+len(count_free_spaces(mat))
    score=score+edge_check(mat)
    
    #CODE FOR EVALUATING SCORE FOR MERGES FOR A GIVEN CONFIGURATION
    i=0
    while i<=5:
        j=0
        while j<=5:
            count=0
            count_opponent=0
            if mat[i][j]!=' ':
                if 0<=(j-1)<=5:
                    if mat[i][j-1]!=' ' and mat[i][j].lower()==mat[i][j-1].lower():
                        if player[PLAYER_]:                 
                            count=count+1
                        elif mat[i][j].lower()>'c':
                            k=j
                            j=j-1
                            if not player[PLAYER_]:
                                count_opponent=count_opponent+2
                            else:
                                count_opponent=count_opponent+1
                            j=k
                        else: 
                            count_opponent=count_opponent+1
                if 0<=(j+1)<=5:
                    if mat[i][j+1]!=' ' and  mat[i][j].lower()==mat[i][j+1].lower():
                        if player[PLAYER_]:
                            count=count+1
                        elif mat[i][j].lower()>'c':
                            k=j
                            j=j+1
                            if not player[PLAYER_]:
                                count_opponent=count_opponent+2
                            else: 
                                count_opponent=count_opponent+1
                            j=k    
                        else:  
                            count_opponent=count_opponent+1
                if 0<=(i+1)<=5:
                    if mat[i+1][j]!=' ' and mat[i][j].lower()==mat[i+1][j].lower():
                        if player[PLAYER_]:
                            count=count+1
                        elif mat[i][j].lower()>'c':
                            k=i
                            i=i+1
                            if not player[PLAYER_]:
                                count_opponent=count_opponent+2
                            else:
                                count_opponent=count_opponent+1
                            i=k
                        else:
                            count_opponent=count_opponent+1
                if 0<=(i-1)<=5:
                    if mat[i-1][j]!=' ' and mat[i][j].lower()==mat[i-1][j].lower():
                        if player[PLAYER_]:
                            count=count+1
                        elif mat[i][j].lower()>'c':
                            k=i
                            i=i-1
                            if not player[PLAYER_]:
                                count_opponent=count_opponent+2
                            else:
                                count_opponent=count_opponent+1
                            i=k       
                        else:
                            count_opponent=count_opponent+1      
                score=score+(count*letter[mat[i][j].lower()])-(count_opponent*letter[mat[i][j].lower()])
            j=j+1
        i=i+1   
    return score


def makeMove(game,move):
        if move == 'L':
            game._Game_IJK__left(game.getGame())
        if move == 'R':
            game._Game_IJK__right(game.getGame())
        if move == 'D':
            game._Game_IJK__down(game.getGame())
        if move == 'U':
            game._Game_IJK__up(game.getGame())
        return game          

    
#CODE FOR EXPECTIMINIMAX STARTS    
def MIN_EXPECTIMINIMAX(game_board,MAX,MIN,DEPTH):
    if DEPTH==1:
        return Result(game_board)
    d=DEPTH-1    
    FRINGE=[]
    game_copy1=copy.deepcopy(game_board)
    game_copy2=copy.deepcopy(game_board)
    game_copy3=copy.deepcopy(game_board)
    game_copy4=copy.deepcopy(game_board)
    FRINGE.append(makeMove(game_copy1,'U'))
    FRINGE.append(makeMove(game_copy2,'D'))
    FRINGE.append(makeMove(game_copy3,'R'))
    FRINGE.append(makeMove(game_copy4,'L'))  
    for successor in FRINGE:
        DEPTH=d
        MIN=min(MIN,CHANCE(successor,MAX,MIN,DEPTH,'MIN'))
    return MIN    

def MAX_EXPECTIMINMAX(game_board,MAX,MIN,DEPTH):
    if DEPTH==1:
        return Result(game_board)
    d=DEPTH-1
    FRINGE=[]
    game_copy1=copy.deepcopy(game_board)
    game_copy2=copy.deepcopy(game_board)
    game_copy3=copy.deepcopy(game_board)
    game_copy4=copy.deepcopy(game_board)
    FRINGE.append(makeMove(game_copy1,'U'))
    FRINGE.append(makeMove(game_copy2,'D'))
    FRINGE.append(makeMove(game_copy3,'R'))
    FRINGE.append(makeMove(game_copy4,'L'))
    for successor in FRINGE:
        DEPTH=d
        MAX=min(MAX,CHANCE(successor,MAX,MIN,DEPTH,'MAX'))
    return MAX

def CHANCE(successor,MAX,MIN,DEPTH,MIN_or_MAX):
    free_spaces=count_free_spaces(successor.getGame())
    sum_of_heuristics=0
    if len(free_spaces)==0:
        return sum_of_heuristics 
    for (i,j) in free_spaces:
        if MIN_or_MAX=='MAX':
            successor._Game_IJK__game[i][j]='a'
            sum_of_heuristics=sum_of_heuristics+MIN_EXPECTIMINIMAX(successor,MAX,MIN,DEPTH)
        else:
            successor._Game_IJK__game[i][j]='A'
            sum_of_heuristics=sum_of_heuristics+MAX_EXPECTIMINMAX(successor,MAX,MIN,DEPTH)    
    return float(sum_of_heuristics)/float(len(free_spaces))      
#CODE FOR EXPECTIMINIMAX ENDS

    
#CODE FOR ALPHA BETA PRUNING STARTS
def MAX_Value(game,alpha,beta,DEPTH):
    if DEPTH==1:
       return Result(game) 
    d=DEPTH-1
    FRINGE=[]
    
    game_copy1=copy.deepcopy(game)
    game_copy2=copy.deepcopy(game)
    game_copy3=copy.deepcopy(game)
    game_copy4=copy.deepcopy(game)
    FRINGE.append(game_copy1.makeMove('U'))
    FRINGE.append(game_copy2.makeMove('D'))
    FRINGE.append(game_copy3.makeMove('R'))
    FRINGE.append(game_copy4.makeMove('L')) 
    for successor in FRINGE:
        DEPTH=d
        alpha=max(alpha,MIN_Value(successor,alpha,beta,DEPTH))
        if alpha>=beta:
            return alpha
    return alpha

def MIN_Value(game,alpha,beta,DEPTH):
    if DEPTH==1:
        return Result(game) 
    d=DEPTH-1
    FRINGE=[]
    game_copy1=copy.deepcopy(game)
    game_copy2=copy.deepcopy(game)
    game_copy3=copy.deepcopy(game)
    game_copy4=copy.deepcopy(game)
    FRINGE.append(game_copy1.makeMove('U'))
    FRINGE.append(game_copy2.makeMove('D'))
    FRINGE.append(game_copy3.makeMove('R'))
    FRINGE.append(game_copy4.makeMove('L')) 
    for successor in FRINGE:
        DEPTH=d
        beta=min(beta,MAX_Value(successor,alpha,beta,DEPTH))
        if alpha>=beta:
            return beta
    return beta
#CODE FOR ALPHA BETA PRUNING ENDS


def next_move(game: Game_IJK,player_)-> None:

    '''board: list of list of strings -> current state of the game
       current_player: int -> player who will make the next move either ('+') or -'-')
       deterministic: bool -> either True or False, indicating whether the game is deterministic or not
    '''
    FRINGE=[]
    board = game.getGame()
    player = game.getCurrentPlayer()
    deterministic = game.getDeterministic()
    PLAYER_=player_

    game_copy1=copy.deepcopy(game).makeMove('U')
    game_copy2=copy.deepcopy(game).makeMove('D')
    game_copy3=copy.deepcopy(game).makeMove('R')
    game_copy4=copy.deepcopy(game).makeMove('L')
    FRINGE.append(('U',game_copy1))
    FRINGE.append(('D',game_copy2))
    FRINGE.append(('R',game_copy3))
    FRINGE.append(('L',game_copy4)) 
    MAX=-inf
    move=None
    
    #CODE FOR DETERMINISTIC MODE
    if deterministic:
        for successor in FRINGE:
            DEPTH=6
            # print("1")
            new_MAX=MIN_Value(successor[1],-inf,inf,DEPTH)
            if MAX<new_MAX:
                move=successor[0]
                MAX=new_MAX
                
    #CODE FOR NON-DETERMINISTIC MODE
    else:
#         game_copy5=copy.deepcopy(game).makeMove('U')
#         game_copy6=copy.deepcopy(game).makeMove('D')
#         game_copy7=copy.deepcopy(game).makeMove('R')
#         game_copy8=copy.deepcopy(game).makeMove('L')
#         FRINGE.append(('U',game_copy5))
#         FRINGE.append(('D',game_copy6))
#         FRINGE.append(('R',game_copy7))
#         FRINGE.append(('L',game_copy8))
        for successor in FRINGE:
            DEPTH=2
            # print("1")
            new_MAX=CHANCE(successor[1],-inf, inf,DEPTH,'MAX')
            if MAX<new_MAX:
                move=successor[0]
                MAX=new_MAX
                
    yield move
#     yield random.choice(['U', 'D', 'L', 'R', 'S'])
