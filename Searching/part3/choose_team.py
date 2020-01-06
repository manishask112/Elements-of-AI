#!/usr/local/bin/python3
#
#Code by: [Uma Maheswari Gollapudi(ugollap),Manisha Suresh Kumar(msureshk)]

import sys
import time
import heapq
import copy 
from copy import deepcopy as create_copy

#Just a timer, to keep track of the time passed.
start_time = time.time()
def load_people(filename):
    people={}
    #with open(r"C:\Users\umamg\OneDrive\Desktop\ppl_small.txt", "r") as file:
    with open(filename, "r") as file:
        for line in file:
            l = line.split()
            people[l[0]] = [ float(i) for i in l[1:] ] 
    #print("People",people)
    return people

#Defined a class Node to encapsulate the properties of the node, which will be used for the creation of the dummy, left and right nodes.
class Node:
    def __init__(self, name, skill, cost, bound):
        self.name = name #names of robots
        self.skill = skill
        self.cost = cost
        self.bound = bound #limit for selection of robot
        self.item = 0

#Queue to store the final list of robots, along with both the left and right trees.
class PQ:
    def __init__(self):
        self.heap = []
        self.count = 0
    
    def empty(self):
        return len(self.heap)==0
    
    def insert(self, item, priority):
        c = (priority*-1, self.count ,item)
        heapq.heappush(self.heap,c)
        self.count+=1
    
    def remove(self):
        (_,_,item) = heapq.heappop(self.heap)
        self.count -=1
        return item
#The above queue was created based on code fragments from geeksforgeeks.com and stackexchange.com.

#Function to calculate the bound of every robot for every existing combination of paths    
#Based on explanation available on youtube - link in readme file.
def bound_function(p,b,n):
    if n.cost> b: #check if cost is greater than bound; if true, no solution
        return 0
    else:
        #initialize skill, cost and index/item numbers.
        weight = n.cost
        value = n.skill
        item = n.item
        #check if number of items are lesser than the length of the total items and if total cost>budget
        #this portion adds the whole values of item to bound
        while item<len(p) and weight+p[item][1][0]>b :
            weight_iter = p[item][1][1]
            value_iter = p[item][1][0]
            weight += weight_iter
            value += value_iter
            item+=1
        #adding remaining fractional values of item to the bound
        if b-weight > 0:
            weight_iter = p[item][1][1]
            #print("Weight",weight_iter)
            value_iter = p[item][1][0]
            fvalue = (b-weight) * (value_iter/weight_iter)
            value += fvalue
    return value    
            
# This function implements a non-greedy solution (0/1 knapsack with branch and bound) to the problem:
#  It adds people in decreasing order of "skill per dollar,"
#  until the maximum possible budget is exhausted, but it does not exactly exhaust the entire budget
#  and instead, fits as many whole robots as can be assigned, while maximizing skill.
 
def approx_solve(people, budget): 
    sort_dict = {}
    counter= 0
    pointer = 0
    solution =[]
    q = PQ()
    #print("Sorted Dictionary",sort_dict)
    #reverse = True, descending order sort
    for (person, (skill, cost)) in sorted(people.items(), key=lambda x: x[1][0]/x[1][1],reverse = True):
        sort_dict[counter]  = (person,(skill,cost))
        counter += 1 
    #Create a dummy node, which is just used to branch the values in the dictionary. Create a cound for the node and enqueue it.
    dummy = Node([0 for i in range(0,len(people))],0,0,0)
    dummy.bound = bound_function(sort_dict, budget,dummy)
    q.insert(dummy,dummy.bound)
    
    #Create node to include the final robots that are a part of the solution
    #-999999999999999999 is an impossible value for skill to be at; if true then solution is infinite.
    final = Node([0 for i in range(0,len(people))],-999999999999999999,0,0)
    
    if q.empty()!=0: #just for checking, the while condition takes care of this anyways
        return 0
    while not q.empty():
        initial = q.remove() #gives the values of the initial/current node
        ib = initial.bound
        ii = initial.item
        #print("Initial", initial)
        if ib > final.skill:
            if ii < len(sort_dict)-1:
            #The right nodes of the tree - includes all the nodes except the current element
                right = create_copy(initial)
                right.name[ii]=-1
                right.item +=1 #go to first/next item
                right.bound = bound_function(sort_dict, budget,right)
                if right.bound>final.skill: #determine wether or not robot is valuable
                    q.insert(right, right.bound)
            #The left nodes of the tree - includes all the nodes with the current element       
                left = create_copy(initial)
                left.item +=1 #go to first/next item
                left.name[ii] =1
                left.skill += sort_dict[ii][1][0]
                left.cost += sort_dict[ii][1][1]
                left.bound = bound_function(sort_dict, budget,left)
                if left.cost <=budget:      #determine wether or not robot is valuable
                    if left.skill > final.skill:
                        final = create_copy(left)
                    if left.bound > final.skill:
                        q.insert(left,left.bound)
    
    #Print solution in required format,i.e, to work for given output.
    for iteration in final.name:
        if iteration==1:
            solution.append(sort_dict[pointer]) #gets the name,skill and cost of the optimal robots
        pointer +=1
    return(solution, final.skill,final.cost)
    if skill == -999999999999999999:
        print("Inf")
    """
        if budget - cost > 0:
            solution += ( ( person, 1), )
            budget -= cost
        else:
           return solution + ( ( person, budget/cost ), )
    return solution
    """

if __name__ == "__main__":

    if(len(sys.argv) != 3):
        raise Exception('Error: expected 2 command line arguments')
    budget = float(sys.argv[2])
    people = load_people(sys.argv[1])
    l =()
    #budget = 200.0
    #people = load_people("ppl_small.txt")
    (person,skill,cost) = approx_solve(people, budget)
    number = len(person)
    #To set a check on code running time
    if(time.time()-start_time > 850):
        print("Please wait, a solution will be generated")
    if skill == -999999999999999999:
        print("Inf")
    else:
        print("I've found a group with %d people costing %f with total skill %f!" % \
               #( len(solution), sum(people[p][1]*f for p,f in solution), sum(people[p][0]*f for p,f in solution)))
               (number, cost, skill))
        for robot in person: #to list out the values of the optimal robots
            l +=((robot[0],1),) 
        for p in l:
            print("%s %f" % p)
