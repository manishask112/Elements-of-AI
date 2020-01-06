#!/usr/local/bin/python3
# CSCI B551 Fall 2019
#
# Authors: Uma Maheswari Gollapudi (ugollap), Manisha Suresh (msureshk), Deepthi Raghu (draghu)
#
# based on skeleton code by D. Crandall, 11/2019
#
# ./break_code.py : attack encryption
#

import random
import numpy as np
import math
import copy 
import sys
import encode
import string
import time
start_time = time.time()

# put your code here!
alphabets = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
ip = {}
for i in alphabets:
    ip[i] = 0.000001
    
tp = {}
for i in alphabets:
    for j in alphabets:
        tp[(i, j)] = 0.000001
        
denom = {}
for i in alphabets:
    denom[i] = 0.000001

count = 0
nwords=0

#Compute initial probabilties and transition probabilties for words in the corpus
def probabilities(corpus):
    #print("9999999")
    nwords=0
    count = 0
    for word in corpus:
        nwords+=1
        if (word == " "):
            continue
        if ip[word[0]] == 0.000001:
            ip[word[0]]=1
        else:    
            ip[word[0]]+=1
            
    for i in alphabets:
        ip[i] = ip[i]/nwords
    #print("IP",ip) 
    
    #To count words where letter occurs
    for i in range(1, len(corpus)):
        if (corpus[i] == " "):
            continue
        if corpus[i-1]!=' ' and denom[corpus[i]] == 0.000001:
            denom[corpus[i]]=1
        elif corpus[i-1]!=' ':
            denom[corpus[i]] = denom[corpus[i]]+1 
    #print("Denom2",denom)        

    #To calculate transitions        
    for i in range(1, len(corpus)):
        if (corpus[i] == " "):
            continue
        if corpus[i-1]!=' ' and tp[(corpus[i-1],corpus[i])] == 0.000001:
            tp[(corpus[i-1],corpus[i])]=1
        elif corpus[i-1]!=' ':
            tp[(corpus[i-1],corpus[i])]+=1   
    # print('Transprob',tp)

    for k,v in tp.items():
        tp[k] = v/float(denom[k[0]])
    #print('Transprob',tp)           
    #print(transition_probability)
    #print("10101010101")
    return ip, tp

#Function to shuffle the order of a list
def shuffle_order(list1):
    list2 = copy.copy(list1)
    return list2, random.shuffle(list1)

#Function to generate random string
#Based on https://pynative.com/python-generate-random-string/
def randomString(stringLength):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

#Function to convert text into lists
def Convert(string): 
    out = list(string.split()) 
    return out     

#Function to define intial guesses of T and T'
def table_guess():   
    asciival=[]
    replace_table = {}
    rept={}
    rearrange_table = [0,1,2,3]
    reat = copy.copy(rearrange_table)
    
    asciival = [x for x in range(97,123)]
    asciival_org, non = shuffle_order(asciival)
    for i in range(0, len(alphabets)):
        replace_table.update({alphabets[i]:chr(asciival[i])})
        
    #print("REPLACE_TABLE",replace_table)   
    rlist,non1 = shuffle_order(rearrange_table)
    #print("rlist", rlist)
    #print("Rearrange Table",rearrange_table)
    #print("7777777")
    
    letters = randomString(2)
    for k in replace_table:
        v = replace_table[k]
        n = len(letters)
        for i in range(0,len(letters)):
            if k == letters[i]:
                rept.update({k:replace_table[letters[n-1-i]]})
            if k not in letters:
                rept.update({k:v})
                
    #print("Rept",rept) 
    #print("888888")
#     pos = [0,1]
#     n = len(pos)
#     for j in rearrange_table:
#         for i in range(0,len(pos)-1):
#             if rearrange_table[j] in pos:
#                 reat[pos[i+1]], reat[pos[i]] = rearrange_table[pos[i]], rearrange_table[pos[i+1]] 
    shuffle_order(reat)
    #print("reat",reat)    
    return rearrange_table, reat, replace_table, rept

#Function to create new replace and rearrange tables for updation
def changes(rearrange,replace):
    #New replace function
    rept_next = {}
    rear_next = rearrange
    letters = randomString(2)
    #print("yolo")
    for k in replace:
        v = replace[k]
        n = len(letters)
        for i in range(0,len(letters)):
            if k == letters[i]:
                rept_next.update({k:replace[letters[n-1-i]]})
            if k not in letters:
                rept_next.update({k:v})
    #New rearrange function            
#     pos = [0,1]
#     #print("lolo")
#     n = len(pos)
#     for j in rearrange:
#         for i in range(0,len(pos)-1):
#             if rearrange[j] in pos:
#                 rear_next[pos[i+1]], rear_next[pos[i]] = rearrange[pos[i]], rearrange[pos[i+1]]
    shuffle_order(rear_next)
    
    return rearrange, rear_next, replace, rept_next

#Function to calculate word probabilities
def word_prob(Doc):
    (ip, tp) = probabilities(corpus)
    PDoc = 0
    for ele in Doc:
        pw=1
        for i in range(0,len(ele)):
            if i == 0:
                pw=pw*ip[ele[i]]
            else:
                pw=pw*tp[(ele[i-1],ele[i])]
                
    for k,v in tp.items():
        if v==0.0 or v==0:
            print("key",k)
            
        #print("5555555")
        #print("pword is:",pw)
        PDoc = PDoc + np.log(pw)
        #print("66666")
        #print("pd",PDoc)
        return PDoc

#Function to iterate over updations and checks    
def break_code(string, corpus):
    threshold =0
    (rear_prev, rear_next, rept_prev, rept_next) = table_guess()
    #print("PD",PD)
    #print("PDdash",PDdash) 
    #print("111111111")
    D=encode.encode(encoded, rept_prev, rear_prev)
    for i in range(1,15000):
       # print("ITERATION",i)
        Ddash = encode.encode(D, rept_prev, rear_prev)
        #print("Doc1",D)
#         Ddash = encode.encode(encoded, rept_next, rear_next)
        #print("Doc2",Ddash)
        Document = Convert(D)
        DocumentDash = Convert(Ddash)
        PD = word_prob(Document)
        PDdash = word_prob(DocumentDash)
        if np.exp(PDdash - PD) > threshold:
            D=Ddash
            rear_prev = rear_next
            rept_prev = rept_next
         #Comment the below two lines in case you want to run for more than 10 minutes.
            if (time.time() - start_time)> 570:
                break   
        (rear_prev, rear_next, rept_prev, rept_next)= changes(rear_prev,rept_prev)    
    #Ddash = encode.encode(encoded, rept_next, rear_next)
    #print("heloooo")
    #print("Document",Ddash)
    return Ddash 

if __name__== "__main__":
    if(len(sys.argv) != 4):
        raise Exception("usage: ./break_code.py coded-file corpus output-file")
    
#     encrypted_file = (r'C:\Users\umamg\OneDrive\Desktop\AI3\encrypted-text-3.txt')
#     corpus_file = (r'C:\Users\umamg\OneDrive\Desktop\AI3\corpus.txt')                  
#     encoded = encode.read_clean_file(encrypted_file)
#     corpus = encode.read_clean_file(corpus_file)
#     decoded = break_code(encoded, corpus)   
#     print("File",decoded)

    encoded = encode.read_clean_file(sys.argv[1])
    corpus = encode.read_clean_file(sys.argv[2])
    decoded = break_code(encoded, corpus)
    #print(decoded)

#     with open(r'C:\Users\umamg\OneDrive\Desktop\AI3\output.txt', "w") as file:
#             print(decoded, file=file)

    with open(sys.argv[3], "w") as file:
        print(decoded, file=file)

#probabilities(corpus)
#table_guess()
#break_code(encoded,corpus)
