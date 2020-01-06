# 
# Code to apply replacement and rearrangement code.
#
# Please don't modify this code -- make your changes to break_code.py instead.
#
# D Crandall, 11/2019
#

import random


# This function takes a string str, and encodes it using two
#  strategies: a replacement table, where every instance of each
#  character is replaced with another character (e.g. all a's 
#  replaced with b's), and a rearrangement table, in which
#  groups of n characters are rearranged (e.g., hello -> olhel)
#
# replace_table : This should be a has mapping characters to characters,
#                 e.g. { 'a':'q', 'b':'l', 'c':'m', ... }
#
# rearrange_table : This should be a permutation of the integers from
#                   0 to n-1. For example, (3, 2, 1, 0) would
#                   reverse every sequence of 4 characters, e.g.
#                   "csci is easy" -> "icsc si ysae"
#
def encode(str, replace_table, rearrange_table):
    # apply replace table
    str2 = str.translate({ ord(i):ord(replace_table[i]) for i in replace_table })

    # pad with spaces to even multiple of rearrange table
    str2 +=  ' ' * (len(rearrange_table)-(len(str2) %  len(rearrange_table)))

    # and apply rearrange table
    return "".join(["".join([str2[rearrange_table[j] + i] for j in range(0, len(rearrange_table))]) for i in range(0, len(str), len(rearrange_table))])


# read in encoded text, convert to lowercase, remove other characters
#  (except spaces)
def read_clean_file(filename):
    with open(filename, "r") as file:
        return "".join([ ("".join( [ i if i.islower() or i == ' ' else '' for i in line ] ) + " ") for line in file ] )
