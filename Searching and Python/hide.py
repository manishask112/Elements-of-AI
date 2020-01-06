#!/usr/local/bin/python3
#
# hide.py : a simple friend-hider
#
# Submitted by : [Manisha Suresh Kumar msureshk]
#
# Based on skeleton code by D. Crandall and Z. Kachwala, 2019
#
# The problem to be solved is this:
# Given a campus map, find a placement of F friends so that no two can find one another.
#

import sys

# Parse the map from a given filename
def parse_map(filename):
    with open(filename, "r") as f:
        return [[char for char in line] for line in f.read().split("\n") if line !=""]

# Count total # of friends on board
def count_friends(board):
    
    return sum([ row.count('F') for row in board ] )

# Return a string with the board rendered in a human-friendly format
def printable_board(board):
    
    return "\n".join([ "".join(row) for row in board])

# Add a friend to the board if friend is at a valid position and return it or an empty board otherwise(doesn't change original)
def add_friend(board, row, col):
    for i in range(col-1,-1,-1):
        if board[row][i]=='&':
            break
        if board[row][i]=='F':
            return[]
    for i in range(col+1,len(board[0])):    
        if board[row][i]=='&'or board[row][i]=='@':
            break
        if board[row][i]=='F':
            return[]
    for i in range(row-1,-1,-1): 
        if board[i][col]=='&'or board[i][col]=='@':
            break
        if board[i][col]=='F':
            return[]       
    for i in range(row+1,len(board)):    
        if board[i][col]=='&'or board[i][col]=='@':
            break
        if board[i][col]=='F':
            return[]
    return board[0:row] + [board[row][0:col] + ['F',] + board[row][col+1:]] + board[row+1:]

# Get list of successors of given board state
def successors(board):
    s=[]
    for r in range(0, len(board)): 
        for c in range(0,len(board[0])): 
            if board[r][c] == '.':
                temp=add_friend(board, r, c)
                if temp!=[]:
                    s.append(temp)   
    return s
   
# check if board is a goal state
def is_goal(board):
    return count_friends(board) == K 

# Solve n-rooks!
def solve(initial_board):
    fringe = [initial_board]
    # A list that stores the nodes that are being expanded so as to not visit them again
    expand=[]
    while len(fringe) > 0:
        board=fringe.pop()
        if board not in expand:
            expand.append(board)
            for s in successors(board):
                if is_goal(s):
                    return(s)
                fringe.append(s)
    return False

# Main Function
if __name__ == "__main__":
    IUB_map=parse_map(sys.argv[1])
    # This is K, the number of friends
    K = int(sys.argv[2])
    print ("Starting from initial board:\n" + printable_board(IUB_map) + "\n\nLooking for solution...\n")
    solution = solve(IUB_map)
    print ("Here's what we found:")
    print (printable_board(solution) if solution else "None")
