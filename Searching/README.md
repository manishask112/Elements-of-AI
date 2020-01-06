# Part 1: The Luddy puzzle  

Part 1 requires to solve the 15-puzzle problem of three variants.
## Search Problem
In the given code, BFS was implemented for the search problem leading to unnecessary expansion of nodes thereby taking a long time to find an optimum solution. So in our implementation, we have used A* search using Best First Search and a heuristic. The search abstraction is:
* Valid State: Any possible 15-puzzle board configuration that is solvable
* Successor Function: Calculates all possible allowable moves for a particular variant that can be made from a given configuration
* Cost Function: moves made thus far + sum of Manhattan distance of every 		     tile from its position in goal state
* Goal State: The correct configuration of a 15-puzzle board

 The heuristic used here is:
 h(s)=sum of Manhattan distances of every tile in the given configuration from its goal state positions.
Consider the board configurations that were given "board4" and "board6":
```
1 2 3 4

5 0 6 7

9 10 11 8

13 14 15 12
```
h(s)=4
h*(s)=4
```
0 2 3 4

1 5 6 7

9 10 11 8

13 14 15 12
```
h(s)=6
h*(s)=6

In both cases, h(s)<=h*(s) proving it to be admissible.
## Working

Since the only difference among the three variants is in the allowable moves and since the goal state is the same across the three variants, we used A* search for all three, implementing Best First Search with Priority Queue for the Fringe and a suitable heuristic function. 
The Fringe consists of cost, current board configuration and the moves made so far. For a particular state(configuration), the successor function computes the allowed moves from that state along with their cost and these are added the Fringe. Being a Priority Queue, the Fringe ensures that paths with higher costs aren't explored. So the optimum path to the Goal State is found faster than a simple BFS.

**Visited Nodes**
Since all possible next moves from a state will also include previous state, we use a data structure called 'REVISITED' to keep track of already explored states(nodes) so that same paths aren't created repeatedly thereby avoiding entering an infinite loop. 

**Permutation Inversion**
Certain 15-puzzle board configurations are unsolvable. Our A*search is unable to recognize this and creates an endless search tree to find a solution path that doesn't exist. So we pass the initial board  and goal state board through the permutation inversion function to compare their parities. If parities are not equal, the board is unsolvable and the execution ends.
# Part 2: Road Trip
We need to find an optimum path between two cities based on the cost function.
## Search Problem
Similar to Part 1, A* search was used with  the Bidirectional Search for "segments" cost function.
* Valid State: A city reachable from start city 
* Successor Function: Cities reachable from current city
* Cost Function: Number of segments moved/distance traveled/time traveled/mpg
* Goal State: End City

## Working

The dataset provides us with major road segments, their distance in miles and speed limit in mph. So finding optimal path given cost functions "distance", "time" and "mpg" could be done with simple A* search with Best First Search and the Fringe as a Priority Queue. The Fringe has have the city, cost till that city, value of the other parameters till this city, and the path taken till that city.
 
**Bi-directional Search**
For the "segments" cost function, initially, we planned on using the latitude and longitude of a city (that were provided in city-gps) and implement a heuristic such that a path can be found closest to the end city. However, the dataset had many cities that missed values for latitude and longitude, including values for cities in an entire state. We then decided to use a different variant of A*, namely the bi-directional strategy of implementation. Here, the search tree is divided in half, one starting from the start city towards the end city and the other starting from the end-city towards start city and when a common city is encountered by both trees, a path is found. So, we had two Fringes(PQs) and when a new city is popped from each Fringe, it is checked for in the other Fringe. The cost function is the number of segments moved and the Priority Queue ensures that no path with larger number segments are chosen.
This strategy cannot be applied on the other cost functions as cost isn't consistent across all edges and an optimum solution is not guaranteed.

**Visited Nodes**
Here too, all possible next moves from a state will also include previous state. Hence, we use 'REVISITED' to keep track of already explored states(nodes).

# Part 3: Choose Teams
We need to find the robots for a given budget, that will have complete the task with the optimal cost and skill.
## The Problem
The state space for the problem can be defined by:
* Valid State: Any state that includes 0 to all the robots that will be included
* Successor Function: Determines wether the next robot will be included or not.
* Cost Function: 1
* Goal State: Optimal outcome, which implies the combination of all robots with the best skill for the best cost that satisfy the given budget.

## Working

The given initial code performs a good, but not optimal task of choosing robots. Furthermore, it also chooses partial robots, which we cannot do. Basically,the problem with the code is that it choses non optimal robots and includes fractional values of the robots to accomplish the task. Hence, we remove the fractional segment from the code entirely. The implementation of the solution was based on the concept of 0/1 knapsack, which we have come across previously It would have been infeasible to use the dynamic programming variant of this algorithm, as it cannot handle non integer weights/costs. Hence, we solved this question via the branch and bound method. 
The solution involves three steps:
* Creation of a class to encapsulate a dummy node in the tree and the definition of a priority queue.
* The calculation of the maximum bound for each node in the tree.
* The implementation of the branch and bound algorithm for each side of the tree- I.e the left and right nodes.(Note: The left nodes are those that will include the current element, while the right nodes are those that check for combinations of elements not including the current element.)


Essentially, the values from the dictionary are stored into a queue, from where they are split into left and right nodes and the bound for each node is calculated, so that the best robots can be determined. 
For example, if the first robot in the sorted dictionary was the robot Sam, with a given skill and cost, then the first left node after the dummy node would calculate the skill and costs taking Sam into consideration, while the first right node after the dummy node would calculate the skill and cost without taking David into account. If the next robot was say, Edna, then the left node of the first left node would take Edna into consideration, while the right node of the first left node would not, and so on. This is the same logic used for the right node too. This will ensure that all combinations of robots are checked before the best is chosen. 

In order to understand the logic behind the algorithm, we looked at Abdul Bari's youtube video(which can be found at: https://www.youtube.com/watch?v=yV1d-b_NeK8). Furthermore, the idea of using the deepcopy function (in order to recursively copy all the nodes upto the given element) came up during a discussion with Deepthi Raghu (dgrahu), prior to which, we had written a longer, buggier snippet that copied the elements manually. 

During the initial phase, we also worked on using bfs to navigate the sorted dictionary and then compare the skill/cost values, such that everytime we encountered a better robot, we would put that into the final solution.
