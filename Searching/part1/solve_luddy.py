#!/usr/local/bin/python3
# solve_luddy.py : Sliding tile puzzle solver
#
# Code by: [Manisha Suresh Kumar msureshk, Uma Maheswari Gollapudi ugollap]
#
# Based on skeleton code by D. Crandall, September 2019
#
from queue import PriorityQueue 
import sys

MOVES = { "R": (0, -1), "L": (0, 1), "D": (-1, 0), "U": (1,0) }
GOAL_STATE=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0]
REVISITED=[]
LUDDY_MOVES={"A":(2,1), "B":(2,-1), "C":(-2,1), "D":(-2,-1), "E":(1,2), "F":(1,-2), "G":(-1,2), "H":(-1,-2)}

def rowcol2ind(row, col):
    return row*4 + col

def ind2rowcol(ind):
    return (int(ind/4), ind % 4)

def valid_index(row, col):
    return 0 <= row <= 3 and 0 <= col <= 3

def swap_ind(list, ind1, ind2):
    a= list[0:ind1] + (list[ind2],) + list[ind1+1:ind2] + (list[ind1],) + list[ind2+1:]
    return a

def swap_tiles(state, row1, col1, row2, col2):
    return swap_ind(state, *(sorted((rowcol2ind(row1,col1), rowcol2ind(row2,col2)))))

def printable_board(row):
    return [ '%3d %3d %3d %3d'  % (row[j:(j+4)]) for j in range(0, 16, 4) ]

#To check if board is solvable
def permutation_inversion(board):
    sum=0
    for i in range(0,len(board)):
        if board[i]==0:
            continue
        count=0
        for j in range(i+1,len(board)):
            if board[j]==0:
                continue
            if board[j]<board[i]:
                count=count+1
        sum=sum+count
    row = ind2rowcol(board.index(0))
    n=sum+row[0]+1    
    return n%2    

#HEURISTIC:Returns sum of Manhattan distances of each tile from its goal state
def heuristic_number_of_tiles(board):
    manhattan_dist=0
    for i in range(0,len(board)):
        if board[i]==0:
            continue
        (board_row,board_col)=ind2rowcol(board.index(board[i]))
        (goal_row,goal_col)=ind2rowcol(GOAL_STATE.index(board[i]))
        manhattan_dist=manhattan_dist+abs(board_row-goal_row)+abs(board_col-goal_col)
    return manhattan_dist

#Returns a list of possible successor states for original variant
def successors(distance_traversed,state):
    successors=[]
    (empty_row, empty_col) = ind2rowcol(state.index(0))
    for (c, (i, j)) in MOVES.items():
        if valid_index(empty_row+i, empty_col+j):
            board=swap_tiles(state, empty_row, empty_col, empty_row+i, empty_col+j)
            priority=heuristic_number_of_tiles(board)
            priority=priority+distance_traversed+1
            if board in REVISITED:
                continue       
            successors.append((priority,distance_traversed+1,board, c))
    
    return successors       

#Returns a list of possible successor states for circular variant
def successors_circular(distance_traversed,state):
    successors=[]
    (empty_row, empty_col) = ind2rowcol(state.index(0))
    for (c, (i, j)) in MOVES.items():
        if valid_index(empty_row+i, empty_col+j):
            board=swap_tiles(state, empty_row, empty_col, empty_row+i, empty_col+j)
            priority=heuristic_number_of_tiles(board)
            priority=priority+distance_traversed+1
            if board in REVISITED:
                continue
            successors.append((priority,distance_traversed+1,board, c))
    if empty_row==3:
        board=swap_tiles(state,empty_row,empty_col,0,empty_col)
        priority=heuristic_number_of_tiles(board)
        priority=priority+distance_traversed+1
        if board not in REVISITED:
            successors.append((priority,distance_traversed+1,board, 'U')) 
               
    if empty_col==3:
        board=swap_tiles(state,empty_row,empty_col,empty_row,0) 
        priority=heuristic_number_of_tiles(board)
        priority=priority+distance_traversed+1
        if board not in REVISITED:
            successors.append((priority,distance_traversed+1,board,'L'))
              
    if empty_row==0:
        board=swap_tiles(state,empty_row,empty_col,3,empty_col)      
        priority=heuristic_number_of_tiles(board)
        priority=priority+distance_traversed+1
        if board not in REVISITED:
            successors.append((priority,distance_traversed+1,board,'D'))
        
    if empty_col==0:
        board=swap_tiles(state,empty_row,empty_col,empty_row,3)  
        priority=heuristic_number_of_tiles(board)
        priority=priority+distance_traversed+1
        if board not in REVISITED:
            successors.append((priority,distance_traversed+1,board,'R'))                    
    return successors

#Returns a list of possible successor states for luddy variant
def successors_luddy(distance_traversed,state):
    successors=[]
    (empty_row, empty_col) = ind2rowcol(state.index(0))
    for (c, (i, j)) in LUDDY_MOVES.items():
        if valid_index(empty_row+i, empty_col+j):
            board=swap_tiles(state, empty_row, empty_col, empty_row+i, empty_col+j)
            priority=heuristic_number_of_tiles(board)
            priority=priority+distance_traversed+1
            if board in REVISITED:
                continue
            successors.append((priority,distance_traversed+1,board, c))
    return successors        

# check if we've reached the goal
def is_goal(state):
    return sorted(state[:-1]) == list(state[:-1]) and state[-1]==0
    
# The solver! - for original variant
def solve_original(initial_board):
    fringe = PriorityQueue()
    priority=heuristic_number_of_tiles(initial_board)
    fringe.put((priority,0,initial_board, ""))
    while not fringe.empty():
        (priority,distance_traversed,state,route_so_far) = fringe.get()
        REVISITED.append(state)
        for (priority,distance_traversed,succ,move) in successors(distance_traversed,state ):
            if is_goal(succ):
                return( route_so_far + move )
            fringe.put((priority, distance_traversed, succ, route_so_far + move ) )
            
    return False

# The solver! - for circular variant
def solve_circular(initial_board):
    goal_state_parity=permutation_inversion(GOAL_STATE)
    if permutation_inversion(initial_board)!=goal_state_parity:
        return False
    fringe = PriorityQueue()
    priority=heuristic_number_of_tiles(initial_board)
    distance_traversed=0
    fringe.put((priority,distance_traversed,initial_board, ""))
    count=0
    while not fringe.empty():
        (priority,distance_traversed,state,route_so_far) = fringe.get()
        count=count+1
        REVISITED.append(state)
        for (priority,distance_traversed,succ,move) in successors_circular(distance_traversed,state ):
            if is_goal(succ):
                return( route_so_far + move )
            fringe.put((priority, distance_traversed, succ, route_so_far + move ) )
    return False

# The solver! - for luddy variant
def solve_luddy(initial_board):
    goal_state_parity=permutation_inversion(GOAL_STATE)
    if permutation_inversion(initial_board)!=goal_state_parity:
        return False
    fringe = PriorityQueue()
    priority=heuristic_number_of_tiles(initial_board)
    distance_traversed=0
    fringe.put((priority,distance_traversed,initial_board, ""))
    count=0
    while not fringe.empty():
        (priority,distance_traversed,state,route_so_far) = fringe.get()
        count=count+1
        REVISITED.append(state)
        for (priority,distance_traversed,succ,move) in successors_luddy(distance_traversed,state ):
            if is_goal(succ):
                return( route_so_far + move )
            fringe.put((priority, distance_traversed, succ, route_so_far + move ) )
    return False

if __name__ == "__main__":
    if(len(sys.argv) != 3):
        raise(Exception("Error: expected 2 arguments"))
    start_state = []
    with open(sys.argv[1], 'r') as file:
        for line in file:
            start_state += [ int(i) for i in line.split() ]
    if len(start_state) != 16:
        raise(Exception("Error: couldn't parse start state file"))
    print("Start state: \n" +"\n".join(printable_board(tuple(start_state))))
    print("Solving...")
    goal_state_parity=permutation_inversion(GOAL_STATE)
    if permutation_inversion(start_state)!=goal_state_parity:
        route=False
    else:
        if(sys.argv[2] == "original"):
            route = solve_original(tuple(start_state))
        if(sys.argv[2] == "circular"):
            route=solve_circular(tuple(start_state))
        if(sys.argv[2] == "luddy"):
            route = solve_luddy(tuple(start_state))
    if not route:
        print("Inf")
    else:
        print("Solution found in " + str(len(route)) + " moves:" + "\n" + route)
