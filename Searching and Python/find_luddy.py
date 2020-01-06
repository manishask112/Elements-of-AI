#!/usr/local/bin/python3
#
# find_luddy.py : a simple maze solver
#
# Submitted by : [Manisha Suresh Kumar msureshk]
#
# Based on skeleton code by Z. Kachwala, 2019
#

import sys
import json

def parse_map(filename):
    with open(filename, "r") as f:
        return [[char for char in line] for line in f.read().split("\n") if line !=""]

def valid_index(pos, n, m):
    return 0 <= pos[0] < n  and 0 <= pos[1] < m

# Find the possible moves from position (row, col)
def moves(maps, row, col):
    moves=((row+1,col), (row-1,col), (row,col-1), (row,col+1))

    # Return only moves that are within the board and legal (i.e. on the sidewalk ".")
    return [ move for move in moves if valid_index(move, len(maps), len(maps[0])) and (maps[move[0]][move[1]] in ".@" ) ]

#If the current move is a shorter move, then popping previous elements from 'navigation'
#Having greater distance to create this shorter path 
def create_path(navigation,curr_move,curr_dist):
    if curr_dist<1:
        navigation.append((curr_move,curr_dist))
        return
    top_dist=navigation[-1]
    if curr_dist<=top_dist[1]: 
        while(top_dist[1]>=curr_dist):
            top_dist=navigation.pop()
            top_dist=navigation[-1]
    navigation.append((curr_move, curr_dist))    
    
#Function to figure out the navigation
def create_solution_path(navigation):
    solution_path=""
    (prev_mov,distance)=navigation[0]
    for i in range(1,len(navigation)):
        (next_mov,distance)=navigation[i]
        if prev_mov[1]==next_mov[1]:
            if prev_mov[0]-next_mov[0]>0:
                solution_path=solution_path+"N"
            else:
                solution_path=solution_path+"S"
        else:
            if prev_mov[1]-next_mov[1]>0:
                solution_path=solution_path+"W"
            else:
                solution_path=solution_path+"E"
        prev_mov=next_mov    
    return solution_path

def search1(IUB_map):
    # Find my start position
    smallest_distance=0
    you_loc=[(row_i,col_i) for col_i in range(len(IUB_map[0])) for row_i in range(len(IUB_map)) if IUB_map[row_i][col_i]=="#"][0]
    fringe=[(you_loc,0)]
    #A dictionary that stores expanded/explored nodes and distance travelled that gets updated with shortest distance
    expand={you_loc:0}
    #Stack that stores the current traversal and finally contains the final solution path
    navigation=[]
    flag=0
    while fringe:
        (curr_move, curr_dist)=fringe.pop()
        #Don't execute while loop if the popped node has already been expanded/explored or 
        #Has travelled a distance greater than shortest distance yet
        if (curr_move in expand and curr_dist>expand[curr_move])  or (flag==1 and (curr_dist+1)>=smallest_distance):
            continue
        #Call create_path() to add new move in current path
        create_path(navigation,curr_move, curr_dist)
        for move in moves(IUB_map, *curr_move):
            if IUB_map[move[0]][move[1]]=="@":
                if flag ==0:
                    smallest_distance=(curr_dist+1)
                    navigation.append(((move[0],move[1]),smallest_distance))
                    solution_path=create_solution_path(navigation)
                    flag=1
                else:
                    if (curr_dist+1)<smallest_distance:
                        navigation.append(((move[0],move[1]),smallest_distance))                
                        solution_path=create_solution_path(navigation)
                        smallest_distance=curr_dist+1
            else: 
                fringe.append((move, curr_dist + 1))
                expand.update({curr_move:curr_dist})
    if smallest_distance==0:
        solution_path="Inf"             
    return (solution_path,smallest_distance)       
            
if __name__ == "__main__":
    IUB_map=parse_map(sys.argv[1])
    print("Shhhh... quiet while I navigate!")
    solution = search1(IUB_map)
    print("Here's the solution I found:")
    if(solution[1]==0):
        print(solution[0])
    else:
        print(str(solution[1]) + " " + solution[0])
