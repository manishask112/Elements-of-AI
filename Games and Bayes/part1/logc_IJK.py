#!/usr/local/bin/python3

"""
This file implements the IJK board and rules. 
Please do not modify anything in this file.
@author: Abhilash Kuhikar, October 2019
"""

'''
logic code inspired from https://github.com/yangshun/2048-python
'''

'''
Game API Documentation:
game.makeMove(move)->Game_IJK:
    Input argument move: one of 'U', 'L', 'R', 'D', 'S'
    Returns deepcopy of the game object after the 'move' is applied
game.printGame()->None:
    Prints the current game in user readable format
    
game.getGame()->List[List[int]]:
    Returns current game board as list of list of int
    
game.getCurrentPlayer()->int:
    Returns '+', if currentPlayer is uppercase
    Returns '-', if currentPlayer is lowercase
game.getDeterministic()->bool:
    Returns True if the game is in deterministic mode
    Returns False if the game is in non-deterministic mode
game.state()->string:
    Returns 
        '+' : uppercase has won
        '-' : lowercase has won
        '' : Game is still on
'''

import random
import copy

class InvalidMoveException(Exception):
    pass

class GameFullException(Exception):
    pass

'''Gives an object of Game_IJK with initial empty game and first player as uppercase
'''
def initialGame(size = 6, player = '+', deterministic = True):
    game = [[' ' for _ in range(size)]for _ in range(size)]
    game[0][0] = 'A' if player == '+' else 'a'
    return Game_IJK(game, player, deterministic)

class Game_IJK:
    def __init__(self, game, currentPlayer, deterministic):
        self.__game = game
        self.__current_player = +1 if currentPlayer == '+' else -1
        self.__previous_game = self.__game
        self.__new_piece_loc = (0,0)
        self.__deterministic = deterministic

    def __switch(self):
        self.__current_player = -self.__current_player
    
    def isGameFull(self):
        for i in range(len(self.__game)):
            for j in range(len(self.__game[0])):
                if self.__game[i][j] == ' ':
                    return False
        return True
    
    def __game_state(self,mat):
        highest = {'+': 'A', '-': 'a'}
        
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                if (mat[i][j]).isupper():
                    highest['+'] = chr(max(ord(mat[i][j]), ord(highest['+'])))
                if (mat[i][j]).islower():
                    highest['-'] = chr(max(ord(mat[i][j]), ord(highest['-'])))
        
        if highest['+'] == 'K' or highest['-'] == 'k' or self.isGameFull():
            if highest['+'].lower() != highest['-']:
                return highest['+'] if highest['+'].lower()>highest['-'] else highest['-']
            return 'Tie'

        return 0

    def __reverse(self,mat):
        new = []
        for i in range(len(mat)):
            new.append([])
            for j in range(len(mat[0])):
                new[i].append(mat[i][len(mat[0])-j-1])
        return new
    
    def __transpose(self,mat):
        new = []
        for i in range(len(mat[0])):
            new.append([])
            for j in range(len(mat)):
                new[i].append(mat[j][i])
        return new
    
    def __cover_up(self,mat):
        new = [[' ' for _ in range(len(self.__game))]for _ in range(len(self.__game))]

        done = False
        for i in range(len(self.__game)):
            count = 0
            for j in range(len(self.__game)):
                if mat[i][j] != ' ':
                    new[i][count] = mat[i][j]
                    if j != count:
                        done = True
                    count += 1
        return (new, done)
    
    def __merge(self,mat):
        global current_player

        done = False
        for i in range(len(self.__game)):
            for j in range(len(self.__game)-1):
                if mat[i][j] == mat[i][j+1] and mat[i][j] != ' ':
                    mat[i][j] = chr(ord(mat[i][j])+ 1)
                    mat[i][j+1] = ' '
                    done = True
                elif mat[i][j].upper() == mat[i][j+1].upper() and mat[i][j] != ' ':
                    mat[i][j] = chr(ord(mat[i][j])+ 1)
                    mat[i][j] = mat[i][j].upper() if self.__current_player > 0 else mat[i][j].lower()
                    mat[i][j+1] = ' '
                    done = True
        return (mat, done)
    
    def __up(self,game):
        #print("up")
        # return matrix after shifting up
        game = self.__transpose(game)
        game, done = self.__cover_up(game)
        temp = self.__merge(game)
        game = temp[0]
        done = done or temp[1]
        game = self.__cover_up(game)[0]
        game = self.__transpose(game)
        if done == True:
            self.__game = copy.deepcopy(game)
        return (game, done)
    
    def __down(self,game):
        #print("down")
        game = self.__reverse(self.__transpose(game))
        game, done = self.__cover_up(game)
        temp = self.__merge(game)
        game = temp[0]
        done = done or temp[1]
        game = self.__cover_up(game)[0]
        game = self.__transpose(self.__reverse(game))
        if done == True:
            self.__game = copy.deepcopy(game)
        return (game, done)
    
    def __left(self,game):
        #print("left")
        # return matrix after shifting left
        game, done = self.__cover_up(game)
        temp = self.__merge(game)
        game = temp[0]
        done = done or temp[1]
        game = self.__cover_up(game)[0]
        if done == True:
            self.__game = copy.deepcopy(game)
        return (game, done)
    
    def __right(self,game):
        #print("right")
        # return matrix after shifting right
        game = self.__reverse(game)
        game, done = self.__cover_up(game)
        temp = self.__merge(game)
        game = temp[0]
        done = done or temp[1]
        game = self.__cover_up(game)[0]
        game = self.__reverse(game)
        if done == True:
            self.__game = copy.deepcopy(game)
        return (game, done)
    
    def __skip(self):
        x, y = self.__new_piece_loc
        self.__game[x][y] = self.__game[x][y].swapcase()
    '''
    Expose this method to client to print the current state of the board
    '''
    def printGame(self):
        str_game = [['______' for _ in range(len(self.__game))] for _ in range(len(self.__game))]
        
        for i in range(len(self.__game)):
            for j in range(len(self.__game)):
                str_game[i][j] = "_"+self.__game[i][j]+"_"
        
        for i in range(len(self.__game)):
            print("|".join(str_game[i]))
        print("\n")

    def __add_piece(self):
        if self.__deterministic:
            for i in range(len(self.__game)):
                for j in range(len(self.__game)):
                    if self.__game[i][j] == ' ':
                        self.__game[i][j] = 'A' if self.__current_player>0 else 'a'
                        self.__new_piece_loc = (i,j)
                        return
        else:
            open=[]
            for i in range(len(self.__game)):
                for j in range(len(self.__game)):
                    if self.__game[i][j] == ' ':
                        open += [(i,j),]

            if len(open) > 0:
                r = random.choice(open)
                self.__game[r[0]][r[1]] = 'A' if self.__current_player>0 else 'a'
                self.__new_piece_loc = r

                
    def makeMove(self,move):
        if move not in ['U','L','D','R']:
            raise InvalidMoveException

        self.__previous_game = self.__game
        if move == 'L':
            self.__left(self.__game)
        if move == 'R':
            self.__right(self.__game)
        if move == 'D':
            self.__down(self.__game)
        if move == 'U':
            self.__up(self.__game)
        if move == 'S':
            self.__skip()
        
        '''
        Switch player after the move is done
        '''
        self.__switch()
        if move != 'S':
            self.__add_piece()
        #self.printGame()
        
        return copy.deepcopy(self)

    def getDeterministic(self):
        return self.__deterministic
    
    def getGame(self):
        return copy.deepcopy(self.__game)
    
    '''player who will make the next move'''
    def getCurrentPlayer(self):
        return '+' if self.__current_player > 0 else '-'

    ''' '+' : '+' has won
       '-1' : '-' has won
       '' : Game is still on
    '''
    def state(self):
        return self.__game_state(self.__game)
