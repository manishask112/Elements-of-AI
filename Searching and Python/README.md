# Part 1: Finding your way 

Part 1 requires to find the shortest path from where we stand to Luddy Hall walking only on sidewalks one step at a time and only up, down,left or right.
## Given Code
The given code implements DFS algorithm, where elements in the fringe are tuples containing coordinates of a particular state and the distance traveled to get there. The search abstraction is:
* Valid State: If the state is either a '.' or '@'>
* Successor Function: Among all four principle coordinates, every state that is either a '.' or a '@' can be a traversable path
* Cost Function: 1, for every move made
* Goal State: When the current coordinate is '@', the first time it is encountered, and may not be optimal
* Initial State: The coordinates of '#' and distance traveled as 0.
 
 This search abstraction fails in finding the optimum distance between you and Luddy Hall because the algorithm explores nodes that have already been explored previously, constantly adding nodes again and again to fringe therefore entering an infinite loop. Also no logic is implemented to find the shortest path. 
## New Algorithm Implementation

Retaining DFS algorithm and the structure of Fringe of existing code, the new code adds more conditions before adding new nodes to fringe and updates every time a shorter path to goal node is found.

**Visited Nodes**
First, another data structure (a list) called 'expand' was made that keeps track of the nodes that have been already explored starting from '#'. It is a list of coordinates. This will ensure that a node isn't unnecessarily explored again as the popped node is checked against 'expand' before exploring it. However, this meant that for a case when a particular node appears again for a shorter traveled distance than the previous appearance, it will not be explored thereby terminating exploration of a possible shortest path. 
So, the structure was changed to a python dictionary to hold both coordinates and smallest distance from '#' to it. A dictionary was chosen since searching and updating is quicker for it. This way if a newly popped node from fringe already exists in the dictionary, and exists at a smaller traveled distance, then this new path need not be explored else, the new shorter path is explored and the shorter distance is updated in the dictionary. 
To further reduce the chances of unnecessary exploration of nodes, at any point if the traveled distance is greater than the latest found optimum path, this new path is not explored further and backtracking is done.

**Storing Solution Path**
In order to display the path in the final output of the code, we need to keep a record of the path, every time a possible optimum path is found. This is done with the help of 'navigation' data structure. When a new optimal path is found, 'navigation' is updated to reflect this. The function create_path() does this task. While 'expand' will have the shortest path from '#' to it for even nodes that might not be in the optimal path, 'navigation' only has the path of current optimal path. So at the end of the program, when the actual optimal path is found, 'navigation' will reflect this.
# Part 2: Hide-and-seek
We need to place k number of friends on sidewalks such that they don't face each either in the same row or column, If a building comes between them, then they can be placed on either side of the building. 
## Given Code
The given code implements DFS algorithm, where elements in the fringe are entire map representations with a newly added 'F' in a particular position. The search abstraction is:
* Valid State: A board which has 'F's at positions that are sidewalks 
* Successor Function: If the current position is a sidewalk,  a friend can be added there to create a new board/state
* Cost Function: 1, for every new friend that is placed
* Goal State: a board with n Friends placed with no friend facing the other
* Initial State: A board with no friends placed.
The successor function for this code allows friends to be placed at a location if its a side walk. So friends end up adjacent to each other.
## New Algorithm Implementation

Retaining DFS algorithm and the structure of Fringe of existing code, the new code will check certain conditions before adding a friend to that position thereby reducing search space. In particular, for every position that is being sent to add_friend(), before the friend can be added to the board, the row and column of that position is checked to see if it meets the conditions required to add a friend there. 
This change worked well in giving an output that meets all the conditions, however, for the below map and for n=9 friends, the code took over 120s to find an arrangement. 
```
....&&&
.&&&... 
....&..
.&.&...
.&.&.&.
#&...&@
```
This meant further optimization needed to be done. After debugging the code, it was found that certain board representations were being repeated in the fringe. Which meant that boards that were found to be inconsequential will repeat some time later only to be proved inconsequential again. Referring to the usage of 'expand' data structure that would store visited nodes in Part 1 of the assignment, the same logic was applied here to prevent visiting nodes repeatedly. After this modification, the code ran for n=9 friends in approximately 5s. The valid state and successor function now becomes:
* Valid State: A board which has 'F's at valid positions that are sidewalks (not facing any friend)
* Successor Function: If the current position is a sidewalk,  and there is no friend along this sidewalk in all four directions, a friend can be added there to create a new board/state
