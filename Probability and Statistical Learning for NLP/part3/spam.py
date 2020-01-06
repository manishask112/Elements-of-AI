# Reference websites:
# Naive Bayes Classification Overall prodcedure- https://towardsdatascience.com/unfolding-na%C3%AFve-bayes-from-scratch-2e86dcae4b01, https://medium.com/syncedreview/applying-multinomial-naive-bayes-to-nlp-problems-a-practical-explanation-4f5271768ebf
# email package reference - https://docs.python.org/2/library/email.iterators.html

import os
import math
import email
import re
from os import listdir
from os.path import isfile, join
import sys
from sklearn.metrics import accuracy_score

# laplace smoothening to avoid zero probabilities 
# words present in testing data but not present in training data have been represented and stored as "UnKnown"
# Concept and formula reference - https://towardsdatascience.com/unfolding-na%C3%AFve-bayes-from-scratch-2e86dcae4b01, https://medium.com/syncedreview/applying-multinomial-naive-bayes-to-nlp-problems-a-practical-explanation-4f5271768ebf, https://en.wikipedia.org/wiki/Additive_smoothing
def laplace_smoothening(words):
    word_count = 0
    for token in words:
        word_count += words[token]
    smoothened_words = {}
    for token in words:
        smoothened_words[token] = math.log((words[token] + 1)/(word_count + len(words) + 1))
    smoothened_words["UnKnown"] = math.log(1/(word_count + len(words) + 1))
    return smoothened_words


class SpamClassifier():

    def spamClassifier_train(self, spam_directory, notspam_directory):
        # store filenames of spam directory
        spam_files = [os.path.join(spam_directory, f)
                      for f in os.listdir(spam_directory)]
        # store filenames of notspam directory
        notspam_files = [os.path.join(notspam_directory, f)
                         for f in os.listdir(notspam_directory)]

        # tokenizing all the spam emails
        # email package reference - https://docs.python.org/2/library/email.iterators.html
        spam_tokens = []
        for filename in spam_files:
            with open(filename, encoding='utf-8', errors='replace') as file:
                email_msg = email.message_from_file(file)
                for line in email.iterators.body_line_iterator(email_msg):
                    token = line.split()
                    spam_tokens += token

        # counting spam words in email
        words = {}
        for token in spam_tokens:
            if token in words:
                words[token] += 1
            else:
                words[token] = 1

        self.spam = laplace_smoothening(words)

        # tokenizing all the notspam email
        # email package reference - https://docs.python.org/2/library/email.iterators.html
        notspam_tokens = []
        for filename in notspam_files:
            with open(filename, encoding='utf-8', errors='replace') as file:
                email_msg = email.message_from_file(file)
                for line in email.iterators.body_line_iterator(email_msg):
                    token = line.split()
                    notspam_tokens += token

        # counting notspam words in email
        words = {}
        for token in notspam_tokens:
            if token in words:
                words[token] += 1
            else:
                words[token] = 1
        self.notspam = laplace_smoothening(words)
        
        # calculating probability of spam and notspam emails
        numberOfFiles = len(spam_files) + len(notspam_files)
        self.spam_probability = (len(spam_files)) / numberOfFiles
        self.notspam_probability = (len(notspam_files)) / numberOfFiles

    def spamClassifier_test(self, testFile):

        spam = math.log(self.spam_probability)
        notspam = math.log(self.notspam_probability)

        # tokenizing all the test email 
        tokens = []
        with open(testFile, encoding='utf-8', errors='replace') as file:
            email_msg = email.message_from_file(file)
            for line in email.iterators.body_line_iterator(email_msg):
                token = line.split()
                tokens += token

        # find sum of spam and non spam probabilities based on words in email
        for token in tokens:
            if token in self.spam:
                spam += self.spam[token]
            else:
                spam += self.spam["UnKnown"]
            if token in self.notspam:
                notspam += self.notspam[token]
            else:
                notspam += self.notspam["UnKnown"]

        # if sum of spam more than notspam, tag it as spam, else, notspam
        if spam > notspam:
            return "spam"
        else:
            return "notspam"


# main
train_directory = str(sys.argv[1])
test_directory = str(sys.argv[2])
output_file_name = str(sys.argv[3])

spamClassifierObj = SpamClassifier()
# calculate training files probabilities
spamClassifierObj.spamClassifier_train(train_directory+"/spam", train_directory+"/notspam")
# classify test files and write to output file
test_directory_files = [files for files in listdir(test_directory) if isfile(join(test_directory, files))]
output_file = open(output_file_name, "w")
output_str = ""
for filename in test_directory_files:
    output_str += filename + " " + spamClassifierObj.spamClassifier_test(test_directory+"/"+filename)+"\n"
output_file.write(output_str)

# calculate accuracy using the true values in test-groundtruth.txt and the predicted values in output.txt
f1 = open(output_file_name, "r")
predicted = f1.readlines()
pred_arr = []
pred_dict = {}
# store filenames and class (spam/notspam) of predicted output in pred_dict dictionary
for i in range(len(predicted)):
    split = predicted[i].split(" ")
    pred_dict[split[0]] = split[1]

f2 = open("test-groundtruth.txt", "r")
ground_truth = f2.readlines()
true_arr = []
true_dict = {}
# store filenames and class (spam/notspam) of actual output in true_dict dictionary
for i in range(len(ground_truth)):
    split = ground_truth[i].split(" ")
    true_dict[split[0]] = split[1]

# compare keys in both dictionaries (to order the filenames)
for key in true_dict.keys():
    if key in pred_dict.keys():
        true_arr.append(true_dict[key])
        pred_arr.append(pred_dict[key])

# assign 1 to spam and 0 to notspam
for i in range(len(true_arr)):
    if true_arr[i] == "spam":
        true_arr[i] = 1
    elif true_arr[i] == "notspam":
        true_arr[i] = 0

for i in range(len(pred_arr)):
    if pred_arr[i] == "spam":
        pred_arr[i] = 1
    elif pred_arr[i] == "notspam":
        pred_arr[i] = 0

print("Accuracy:", accuracy_score(true_arr, pred_arr)*100)
