# Part 1: IJK 

Part 1 requires to implement the AI for a 2-player game in two modes: deterministic and non-deterministic. This is an Adversarial Search problem.
## Deterministic Mode
	
After every player's turn in the game, an 'a' or 'A'(depending on the player) gets added in the first free space in the board. This implies that for any move that is made on the board, we can say where the As will be placed. A good way to implement search tree for the game will then be Alpha Beta Pruning on Minimax algorithm to a depth 'd'(here, d=6).

## Non-Deterministic Mode

 Here the placement is done randomly after each player's turn .In regular Minimax algorithm, we can work down the game tree adding 'A's or 'a's in predetermined positions and create successors for those boards. Although this can be done for non-deterministic mode as well, the element of randomness here means that we may get board configurations in the actual game that are  different from the ones created in the game tree. For this reason, we have used Expectiminimax Algorithm. There will be 4 chance nodes representing the four moves('L', 'R', 'U', 'D) between every MIN and MAX nodes. Each chance node calculates the average value of evaluation function on boards that have 'A's or 'a's in all possible locations for that particular move.
 Now this means that your game tree will be a lot bigger than the game tree in regular Minimax so we go up to a depth of 2 only.
 
## Evaluation Function
To come up with evaluation functions, after noticing the similarities between IJK and the popular 2048, we took to our phones to look for patterns in the way we play the game. We noticed that all of us follow the strategy of arranging tiles in an increasing or decreasing order with the largest tile in the corners. This creates an order in the board and makes merges easier. According to Nie et al. 2016 <sup> [1] </sup>, this is called 'monotonicity' and the number of merges are maximised when the tiles are arranged in a snake-like shape. Along with this, we also kept a count of number of free spaces in the board (each free space gets a value 1) because more number of free spaces facilitates more number of moves and helps prevent the board from filling up too fast.

Being a competitive game, we also incorporated the number and type of merges that happen in each board for MIN and MAX.

Depending on the type of merge:
1. **Letters of same case**
Assign value 1 for smaller case or upper case, if MAX player is - or + respectively.
Assign value 2 for smaller or upper case if MAX player is + or - respectively.
2. **Letters of different case**
Assign value 1.

Then take the difference between the weighted sum of these values. Weights are assigned for each letter and increases for each letter with larger weights assigned to letters greater than 'D' and even larger weights for 'J' and 'K'. 

With these two heuristics, we noticed that the AI prioritised monotonicity over the merges which meant that the opponent could make merges and create higher alphabets while the AI is concerned with maintaining monotonicity in the board. So, we increased the weights assigned for merges. Although this change meant that the AI could now tell when a move is favorable to the opponent and when its favorable to itself, monotonicity of the board was compromised. This led us to infer that  unlike 2048 which is a single player game, in IJK, when our moves and strategy at each stage are challenged by the adversary, there are other parameters that take precedence over monotonicity in the need to maximise our chances and minimise the opponent's. We added a final heuristic by replacing monotonicity with a value of 1 for each player if they have their highest letter achieved so far at any of the corners multiplied by the weight of that letter. The weights are same as before.
Our final evaluation function is as below:
```
e(s)=Number of free spaces+ 
     sum of differences of values for favorable merges between each player multiplied by weight of that letter+
     sum of difference of values for highest letter at a corner between each player multiplied by weight of that letter
Weights={'a':1,'b':2,'c':3,'d':4,'e':7,'f':8,'g':9,'h':10,'i':11,'j':20,'k':22,' ':0}     
```

# Part 2: Horizon finding

The horizon finding problem is split into three parts:
## Part 2.1 
This is a simple approach to estimate the following:

<b>s<sub>i</sub><sup>*</sup> = arg max s<sub>i</sub> P(S<sub>i</sub>=s<sub>i</sub>|w<sub>1</sub>,..,w<sub>m</sub>)</b>

i.e, for each column in the grey-scale image, the corresponding row which has the maximum pixel value (since the color white has maximum pixel value). 

This gives us a list of row indices in the image with an assumption that the point in each column which has maximum pixel value will lie on the ridge line. However, while drawing a line using the row values estimated using this method, it is observed that the image might also have points which are not the horizon, but have a higher pixel value (like a person wearing a white shirt, or white buildings and walls). In this case, this method results in a scattered estimate, which includes many outliers scattered randomly away from the horizon boundary line (refer images below).

| ![](https://lh3.googleusercontent.com/5eSuF5XiO7zE91SZTx4bn03Hl9EqRzO_Zbo_GAQvQSYlWjqxKJza5AQRSIb71R427aYSJ1k9_PIgWg)| ![](https://lh3.googleusercontent.com/sDgK-EXaU0ZuTWx-wlfyyPmmTaXTs_S9Rt8gz4Y0h4O86ornj3mZe08qEvrM5qIqfclSfJKLyz0dzA)
|:---:|:---:|

## Part 2.2 
This approach involves using Viterbi algorithm to estimate the following maximum a posteriori path:

<b>arg max s<sub>1</sub>,..,s<sub>m</sub> P(S<sub>1</sub>=s<sub>1</sub>,..,S<sub>m</sub>=s<sub>m</sub>|w<sub>1</sub>,..,w<sub>m</sub>)</b>

The difference between 2.1 and 2.2 is that instead of finding the maximum pixel in each column, we use Viterbi algorithm to find the most probable sequence of row values. 
We define the following for our Viterbi algorithm:
* <b>States:</b> All possible row indices in the image (i.e, 0 to n)
* <b>Observed variables:</b> Row indices corresponding to the maximum pixel value in each column 
* <b>Initial probability:</b> 1
* <b> Emission probability:</b> Since we need an emission probability such that it is high near a strong edge and low otherwise, this is given by the grey-scale image itself. We normalize the grey-scale image (which essentially has values between 0 and 255) to have values between 0 and 1, and use this as our emission probability. 
* <b>Transition probability:</b> This is defined in a way to establish smoothness when the line is drawn from one row to another across the columns in the image. For a transition between two row whose indices have a difference of 20 or below, we assign a probability of 1 and for a difference greater than 20, we assign a transition probability of 0.  

After applying Viterbi algorithm on the image, it is observed that the most probable boundary line is found if there is a clear distinction between the sky, mountain and the rest of the image. However, if there is a large variation in intensity values (i.e, more of white in the image), this method does not find the perfect boundary line (refer images below). Another observation is that the most outliers, whose nearby values have been found to lie on the ridge line, have been eliminated by Viterbi algorithm since it is driven by the emission and transition probabilities (it estimates current row values based on previously estimated row values) . The outliers that exist after applying Viterbi algorithm are observed to occur in a sequence of columns in a particular area of the image where there is a spike in pixel intensity. This has been rectified by making use of the human input in Part 2.3

| ![](https://lh3.googleusercontent.com/EtuAZlAdwM2Op0SM1HiSX0E30tLy9XY8TSIC35U1di7e1sODpSF5UsbKx6N6gcn8ZWKUJhR6LfRN0A)| ![](https://lh3.googleusercontent.com/WqhGap2X-3mmVnZnTIOwqJxZv4ZFTpOTzVHQhyzBFsI_Ov_FlXRWnQE3QrrB68eNUS-S04OckWxMdw)
|:---:|:---:|


## Part 2.3 

In Part 2.3, a human input is provided which is a row and column corresponding to a point that lies on the horizon line. Using this new information, we modify our probabilities to help the Viterbi algorithm make a better estimate on the horizon line. 
We define the following for our Viterbi algorithm:
* <b>States:</b> Same as in Part 2.2 
* <b>Observed variables:</b> Same as in Part 2.2 
* <b>New Initial probability:</b> The pixel value corresponding to the row and column value given by human, normalized between 0 and 1
* <b>New Emission probability:</b> Since we need an emission probability such that it is high near a strong edge and low otherwise, this is given by the grey-scale image itself. We normalize the grey-scale image (which essentially has values between 0 and 255) to have values between 0 and 1, and use this as our emission probability. Additionally, based on the pixel value given by human, we assume that all rows below the human provided row value + 10 can be ignored. For example, if the human gives 20 as input, we ignore all rows below 30. This assumption is made based on another assumption that the human looks at the image and gives the lowest point on the horizon line. Since we are looking at estimating a horizon boundary between mountain and sky, it is assumed that the boundary line cannot go below the lowest point of the mountain. To incorporate this assumption, we make the emission probabilities of all the cells below the human provided row + 10 as 0. We also make the emission probability of human provided pixel as 1, since we know this point definitely lies on the horizon line.

* <b>Transition probability:</b> Same as in Part 2.2

After incorporating these changes, it is observed that the algorithm estimates a better horizon boundary line, when compared to Part 2.3 (refer image below).

| ![](https://lh3.googleusercontent.com/ZmhgWz0IzHgJ0lMzuiyHPXAlbn4qWDtenNTII98j_fHJ2glMBSEFC0SXI51AEs5JR54xaWxKWfz1dg)| ![](https://lh3.googleusercontent.com/k6yT-UKte28mnsJQ75kvbUychPnOjxzjhV-nHCbp-XjbCMBquy4pjD0qfwe9XPH36vQcuxRoXnhQDg)
|:---:|:---:|

## Illustration of improved estimate of horizon line:
Output of Part 2.1 is marked by a blue ridge line, Part 2.2 with a red line and Part 2.3 with a green line

| ![](https://lh3.googleusercontent.com/XqytJyCapBNIyJfFnXhdzaN-vPaT4uNfp3aItWgaiig8uAmt3IHCDPN0g4dvT8t_db1_DnhXuPz7sA) | ![](https://lh3.googleusercontent.com/NyQbt70ns74VLP-PBVCikHM-qzFIl6F05-CQ38W1HNM_5I8wsWL9ro1Wn7biFCH8FmQSa1paO8FoyA) | ![](https://lh3.googleusercontent.com/O4W-9fGoRmP9cd6dax15jsepcMny2N9pZfuSwlE8u13Az-N6W6vAm68_e6SAwZxvC0RWNaN8WQ8Cbg)|
|:---:|:---:|:---:|

**References**
1. Yun Nie, Wenqi Hou, Yicheng An., 2016, AI Plays 2048, [http://cs229.stanford.edu/proj2016/report/NieHouAn-AIPlays2048-report.pdf](http://cs229.stanford.edu/proj2016/report/NieHouAn-AIPlays2048-report.pdf)
