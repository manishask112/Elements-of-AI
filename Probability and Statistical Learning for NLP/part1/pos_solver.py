###################################
# CS B551 Fall 2019, Assignment #3
#
# Your names and user ids: Manisha Suresh Kumar(msureshk), Deepthi Raghu(draghu), Uma Maheswari Gollapudi(ugollap)
#
# (Based on skeleton code by D. Crandall)
#


import random
import math

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def __init__(self):
        self.initial_probability={}
        self.emission_probabilities={}
        self.transition_probabilities={}
        self.pos={}

    def posterior(self, model, sentence, label):
        sum=0.0
        if model in ("Simple","Complex","HMM"):
            
            for i in range(0,len(sentence)):
                if (sentence[i],label[i]) not in self.emission_probabilities:
                    prob=0.0000000001
                else:
                    prob=self.emission_probabilities[(sentence[i],label[i])]
                sum=sum+math.log(prob*self.pos[label[i]],2)    
        else:
            print("Unknown algo!")
        return sum    

    # Do the training!
    # Calculating prior probabilities, initial probabilities, emission probabilities and transition probabilities
    def train(self, data):
        words={}
        
        count=0
        for observed_hidden in data:
            for i in range(0,len(observed_hidden[0])):

                # Word count
                if observed_hidden[0][i] not in words:
                    words.update({observed_hidden[0][i]:1})
                else:
                    words[observed_hidden[0][i]]=words[observed_hidden[0][i]]+1

                # POS count
                try:
                    if  observed_hidden[1][i] not in self.pos:
                        self.pos.update({observed_hidden[1][i]:1})
                        count=count+1
                except IndexError:
                    pass       
                else:
                    self.pos[observed_hidden[1][i]]=self.pos[observed_hidden[1][i]]+1
                    count=count+1

                # Count of POS of initial words
                if i==0:
                    if observed_hidden[1][i] not in self.initial_probability:
                        self.initial_probability.update({observed_hidden[1][i]:1})
                    else:
                        self.initial_probability[observed_hidden[1][i]]=self.initial_probability[observed_hidden[1][i]]+1 
            
                else:
                    # Setting count to 0 for POS of words that are not the first of the sentences
                    try:
                        if observed_hidden[1][i] not in self.initial_probability:
                            self.initial_probability.update({observed_hidden[1][i]:0})
                    except IndexError:
                        pass
                    
                    # Count of (Si,Si+1) for transition probability
                    try:
                        if (observed_hidden[1][i-1],observed_hidden[1][i]) not in self.transition_probabilities:
                            self.transition_probabilities.update({(observed_hidden[1][i-1],observed_hidden[1][i]):1})
                    except IndexError:
                        pass        
                    else:
                        self.transition_probabilities[(observed_hidden[1][i-1],observed_hidden[1][i])]=self.transition_probabilities[(observed_hidden[1][i-1],observed_hidden[1][i])]+1
                # Count of (Wi,Si) for emission probability
                try:
                    if (observed_hidden[0][i],observed_hidden[1][i]) not in self.emission_probabilities:
                        self.emission_probabilities.update({(observed_hidden[0][i],observed_hidden[1][i]):1})
                except IndexError:
                        pass        
                else:
                    self.emission_probabilities[(observed_hidden[0][i],observed_hidden[1][i])]=self.emission_probabilities[(observed_hidden[0][i],observed_hidden[1][i])]+1
        

        # Calculating initial probability
        for key,value in self.initial_probability.items():
            if value==0:
                self.initial_probability[key]=0.0000000001
            else:    
                self.initial_probability[key]=float(self.initial_probability[key])/float(self.pos[key])

        # Calculating transition probabilities
        for key in self.transition_probabilities:
            self.transition_probabilities[key]=float(self.transition_probabilities[key])/float(self.pos[key[0]])
        
        # Setting probability for (Si,Si+1) that does not exist in train to a very small value
        for key1 in self.pos:
            for key2 in self.pos:
                if (key1,key2) not in self.transition_probabilities:
                    self.transition_probabilities.update({(key1,key2):0.0000000001})

        # Calculating emission probabilities
        for key in self.emission_probabilities:
            self.emission_probabilities[key]=float(self.emission_probabilities[key])/float(self.pos[key[1]])   

        for key1 in words:
            for key2 in self.pos:
                if (key1,key2) not in self.emission_probabilities:
                    self.emission_probabilities.update({(key1,key2):0.0000000001})            

        # Calcuating probabilty of occurance  of each POS
        for key in self.pos:
            self.pos[key]=float(self.pos[key])/float(count)

        

    # Functions for each algorithm
    #Naive Bayes 
    def simplified(self, sentence):
        predicted_pos=[]
        for word in sentence:
            max=0
            most_probable_pos=""
            for pos in self.pos:
                # If word doesn't exist in the train set, we set its emission probability to a low value
                if (word,pos) not in self.emission_probabilities:  
                    prob=0.0000000001
                else:    
                    prob=self.emission_probabilities[(word,pos)]*self.pos[pos]
                if prob>max:
                    max=prob
                    most_probable_pos=pos
            predicted_pos.append(most_probable_pos)        
        return predicted_pos
        # return [ "noun" ] * len(sentence)

    # Gibbs Sampling
    def complex_mcmc(self, sentence):
        iterations=2000
        most_probabale_pos=[]
        pos_list=[]
        for key in self.pos:
            pos_list.append(key)
        list_of_particles=[]    
        list_of_particles_prob=[]
        #Becase noun is the most occuring pos, we initial the first particle to all nouns
        particle=['noun']*len(sentence) 
        # Probabilities of tag corresponding to each word
        particle_prob=[0.0]*len(sentence)
        for j in range(0,iterations):
            for i in range(0,len(sentence)):
                # Probability of each tag for one word
                pos_probabilities_list=[]
                for k in range(0,len(pos_list)) :
                    if (sentence[i],pos_list[k]) not in self.emission_probabilities:
                        emission_prob=0.0000000001
                    else:
                        emission_prob=self.emission_probabilities[(sentence[i],pos_list[k])]
                    # For first word, P = P(Si)*P(Wi|Si) -----> When there is one word in the sentence
                    #                 P = P(Si)*P(Wi|Si)*P(Si+1|Si) ----->When theres more than one word in the sentence
                    if i == 0:
                        if len(sentence)==1:
                            prob=self.pos[pos_list[k]]*emission_prob
                        else:
                            prob=self.pos[pos_list[k]]*emission_prob*self.transition_probabilities[(pos_list[k],particle[i+1])]
                    
                    # For last word and when length of sentence is greater than 1:
                    #  P = P(Si)*P(Wi|Si)*P(Si|Si-i) -----> When there is two words in the sentence
                    #  P = P(Si)*P(Wi|Si)*P(Si|Si-i)*P(Si|S0) ----->When there is more than two words in the sentence
                    elif i == len(sentence)-1 and len(sentence)!=1:
                        if len(sentence)==2:
                            prob=self.pos[pos_list[k]]*emission_prob*self.transition_probabilities[(particle[i-1],pos_list[k])]
                        else:    
                            prob=self.pos[pos_list[k]]*emission_prob*self.transition_probabilities[(particle[i-1],pos_list[k])]*self.transition_probabilities[(particle[0],pos_list[k])]
                    # For words that are not first or last, P = P(Si)*P(Wi|Si)*P(Si|Si-i)*P(Si+1|Si)
                    else:    
                        prob=self.pos[pos_list[k]]*emission_prob*self.transition_probabilities[(pos_list[k],particle[i+1])]*self.transition_probabilities[(particle[i-1],pos_list[k])]
                    
                    pos_probabilities_list.append(prob) 

                # Replace existing tag with most robable tag 
                particle[i]=pos_list[pos_probabilities_list.index(max(pos_probabilities_list))] 
                # Replace probabiity with maximum probability
                particle_prob[i]=max(pos_probabilities_list)
            
            #Recording the most probable tag and its probability for all iterations   
            list_of_particles.append(particle)
            list_of_particles_prob.append(particle_prob)

        # Selecting the most probable tag corresponding to the maximum probablity 
        # Obtained from saved max probabilities of all iterations
        for i in range(0,len(sentence)):
            max_prob=[]
            for j in range(0,len(list_of_particles_prob)):
                max_prob.append(list_of_particles[j][i])
            most_probabale_pos.append(list_of_particles[max_prob.index(max(max_prob))][i])        
                

        return most_probabale_pos
        # return [ "noun" ] * len(sentence)

    # Viterbi Algorithm
    def hmm_viterbi(self, sentence):
        most_probable_pos=[]
        most_probable_pos_return=[]

        # Table to implement Viterbi Algorithm
        viterbi_table=[[0]*len(self.pos)for i in range(len(sentence))]
        pos_list=[]

        # List of unique tags
        for key in self.pos:
            pos_list.append(key)

        # Viterbi Algorithm 
        for i in range(0,len(sentence)):
            for j in range(0,len(pos_list)):
                if (sentence[i],pos_list[j]) not in self.emission_probabilities:
                    prob=0.0000000001
                else:
                    prob=self.emission_probabilities[(sentence[i],pos_list[j])]

                # For first word, Vij=P(Si)*P(Wi|Si)
                if i==0:
                    viterbi_table[i][j]=self.initial_probability[pos_list[j]]*prob

                # For words that aren't the first, Vij=P(Wi|Si)*max(VIJ*P(SI|SI-1))
                else:
                    k=0
                    max_vt=0.0
                    for v in viterbi_table[i-1]:
                        v_transition=v*self.transition_probabilities[(pos_list[k],pos_list[j])]
                        if max_vt<v_transition:
                            max_vt=v_transition
                        k=k+1
                    viterbi_table[i][j]=prob*max_vt

        # Extracting the most probable tag fromeh viterbi table 
        for i in range(len(viterbi_table)-1,-1,-1):
            most_probable_pos.append(pos_list[viterbi_table[i].index(max(viterbi_table[i]))])
        for i in range(len(most_probable_pos)-1,-1,-1):    
            most_probable_pos_return.append(most_probable_pos[i])

        return most_probable_pos_return
        # return [ "noun" ] * len(sentence)


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")
