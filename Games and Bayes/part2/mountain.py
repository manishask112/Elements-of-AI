#!/usr/local/bin/python3
#
# Authors: Deepthi Raghu(draghu), Manisha Suresh Kumar(msureshk), Uma Maheswari Gollapudi(ugollap)

#
# Mountain ridge finder
# Based on skeleton code by D. Crandall, Oct 2019
#

from PIL import Image
from numpy import *
from scipy.ndimage import filters
import sys
import imageio
import numpy as np
# calculate "Edge strength map" of an image
#
def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))
    filtered_y = zeros(grayscale.shape)
    filters.sobel(grayscale,0,filtered_y)
    return sqrt(filtered_y**2)

# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
#
def draw_edge(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range( int(max(y-int(thickness/2), 0)), int(min(y+int(thickness/2), image.size[1]-1 )) ):
            image.putpixel((x, t), color)
    return image

# main program
#
(input_filename, gt_row, gt_col) = sys.argv[1:]
# load in image 
input_image = Image.open(input_filename)

# compute edge strength mask
edge_strength = edge_strength(input_image)
imageio.imwrite('edges.jpg', uint8(255 * edge_strength / (amax(edge_strength))))

# You'll need to add code here to figure out the results! For now,
# just create a horizontal centered line.

# count number of rows and columns in the image
col_count = (len(np.transpose(edge_strength)))
row_count = (len(edge_strength))

# Part 2.1: Simple approach - for each column in the grey-scale image, find the corresponding row which has the maximum pixel value
ridge = []
max_edge_list = []
edge_strength_transpose = np.transpose(edge_strength)
for i in range (len(edge_strength_transpose)):
    max_strength = (np.where(edge_strength_transpose[i]==max(edge_strength_transpose[i])))
    ridge.append(max_strength[0][0])
    max_edge_list.append(max(edge_strength_transpose[i]))

# draw blue line using the row indices which have maximum pixel values
imageio.imwrite("output_simple.jpg", draw_edge(input_image, ridge, (0, 0, 255), 5))

# viterbi algorithm implementation
def viterbiAlgorithm(observations, states, start_probability, transition_probability, emission_probability):

    ViterbiTable = [{}]
    stateList = {}
    
    # calculate initial probabilities (t=0) using emission_probability(0th observation)*initial probability
    for i in states:
        ViterbiTable[0][i] = start_probability * emission_probability[i][0]
        stateList[i] = [i]
    
    # calculate probabilities for all other cells (t = 1 to number of observations) 
    for i in range(1, len(observations)):
        ViterbiTable.append({})
        tempStateList = {}
        # calculate max(each of previous viterbi cell value * transition_probability) and multiply with emission_probability(ith observation)
        for j in states:
            (probability, state) = max((ViterbiTable[i-1][k] * transition_probability[k][j] * emission_probability[j][i], k) for k in states)
            ViterbiTable[i][j] = probability
            tempStateList[j] = stateList[state] + [j]
        stateList = tempStateList
    
    # after filling viterbi table, find the maximum probability of each column in the table and store the corresponding state value (row index of the image)
    (probability, state) = max((ViterbiTable[i][j], j) for j in states)
    # retrun a list of states (row indices of the image) corresponding to the maximum viterbi table value of each column
    return (stateList[state])


# Part 2.2: Viterbi without human input

input_image = Image.open(input_filename)

# emission probability - normalize the grey-scale image to have values between 0 and 1 to use it as the emission probability
edge_strength_normalized = (edge_strength - np.min(edge_strength))/(np.max(edge_strength)-np.min(edge_strength))
emission_probability = edge_strength_normalized

# transition probability - if two row whose indices have a difference of 20 or below, assign a probability of 1, else assign probability 0
transition_probability = []
for i in range(row_count):
    transition_probability.append([])
    for j in range(row_count):
        if(abs(i-j)<=20):
            transition_probability[i].append(1)
        else:
            transition_probability[i].append(0)

# states - All possible row indices in the image (i.e, 0 to n)
states = list(range(0,row_count))

# observations - Row indices corresponding to the maximum pixel value in each column
observations = ridge

start_probability = 1

# draw red line using estimated row indices returned by the viterbi algorithm 
ridge_red = (viterbiAlgorithm(observations,states,start_probability,transition_probability,emission_probability))
imageio.imwrite("output_map.jpg", draw_edge(input_image, ridge_red, (255, 0, 0), 5))


# Part 2.3: Viterbi with human input

input_image = Image.open(input_filename)

# emission probability - normalize the grey-scale image to have values between 0 and 1 to use it as the emission probability
edge_strength_normalized = (edge_strength - np.min(edge_strength))/(np.max(edge_strength)-np.min(edge_strength))
emission_probability = edge_strength_normalized

# Make emission probability of human provided pixel as 1 (since we know this point definitely lies on the horizon line)
for i in range(len(emission_probability)):
    for j in range(len(np.transpose(emission_probability))):
        if i == int(gt_row) and j == int(gt_col):
            emission_probability[i][j] = 1

# Assuming that the boundary line cannot go below the lowest point of the mountain, 
# make the emission probabilities of all the cells below the human provided row + 10 as 0
for i in range(int(gt_row)+10,row_count):
    for j in range(len(np.transpose(emission_probability))):
        emission_probability[i][j]=0
                
# transition probability - if two row whose indices have a difference of 20 or below, assign a probability of 1, else assign probability 0
transition_probability = []
for i in range(row_count):
    transition_probability.append([])
    for j in range(row_count):
        if(abs(i-j)<=20):
            transition_probability[i].append(1)
        else:
            transition_probability[i].append(0)

# start probability - The pixel value corresponding to the row and column value given by human, normalized between 0 and 1
start_probability = edge_strength_normalized[int(gt_row)][int(gt_col)]

# draw green line using estimated row indices returned by the viterbi algorithm 
ridge_green = (viterbiAlgorithm(observations,states,start_probability,transition_probability,emission_probability))
imageio.imwrite("output_human.jpg", draw_edge(input_image, ridge_green, (0, 255, 0), 5))
