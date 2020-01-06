# Part 1: Part-of-speech Tagging 

Part 1 requires to predict based on probabilities what the tag is for each word in a test set from a set of possible tags (12 tags). We implement 3 probabilistic methods namely, Simplified Bayes, Viterbi Algorithm and MCMC (Gibbs Sampling) and find out the best method.
## Training The Data
The train set is used to find the different probabilities needed for the three implementations, namely:
```
Prior Probability, P(Si) = count(Si)/count(all S)
Initial Probability, P(S0) = count(S0 as the initial tag)/count(S0)
Emission Probability, P(Wi|Si) = count(Wi and its tag Si)/count(Si)
Transition Probability, P(Si|Si-1) = count(transitions from Si-1 to Si)/count(Si-1)
```
These probabilities are stored in python dictionaries.

**Assumption:**
For probabilities that aren't obtained from the train set, we assign a very small probability of 1/(10)**10.

## Simplified Bayes Net
This is a simple approach where the tag of each word is the most probable one for that word using 
```
si = argmax si P(Si = si|W)
```
Emission probability and prior probabilities are used. 

## Viterbi Algorithm for MAP
This is a better model as it takes into consideration the dependencies between one word and the previous word as is seen in the HMM in the Assignment. A Viterbi Table is constructed to calculate every tag for each word:
```
For the first word,
Vij=P(Si)*P(Wi|Si)
For every word after,
Vij=P(Wi|Si)*max(VIJ*P(SI|SI-1))

where,
P(Si) -----> Initial Probability
P(Wi|Si) -----> Emission Probability
P(SI|SI-1) ----> Transition Probability
```
After constructing the Viterbi Table, we take the tag corresponding to the maximum probability for each word and that gives us the POS tag for the sentence.
## MCMC Gibbs Sampling
An even better model, Gibbs sampling has a far reaching dependencies. The last word in every sentence has two parents: the previous word and the very first word as shown in the Bayes Net in the Assignment.
1. Gibbs Sampling Algorithm requires to create a random particle consisting of randomly assigned pos tags for each word. 
**Assumption:**
Since the most occurring pos tag is 'noun', we just generated a particle with all nouns. Since we have many iterations to change this, the initial tag generation doesn't  impact the final list of tags.

3. Now for few thousand iterations (2000), we will find the most probable tag and replace our particle with it. In each iteration, we will also store separately the maximum probability and the tag associated with it. This is done, so that at the end of the iterations, we can select for each word, the tag with the maximum out of the maximum probabilities. 
```
For first word, 
P = P(Si)*P(Wi|Si) ---> When there is one word in the sentence
P = P(Si)*P(Wi|Si)*P(Si+1|Si) --->When theres more than one word in the sentence

For last word and when length of sentence is greater than 1,
P = P(Si)*P(Wi|Si)*P(Si|Si-i) ---> When there is two words in the sentence
P = P(Si)*P(Wi|Si)*P(Si|Si-i)*P(Si|S0) --->When there is more than two words in the sentence

For words that are not first or last,
P = P(Si)*P(Wi|Si)*P(Si|Si-i)*P(Si+1|Si)

where,
P(Si) -----> Initial Probability
P(Wi|Si) -----> Emission Probability
P(Si|Si-1) ----> Transition Probability
```
**Assumption:**
For Viterbi Algorithm and Gibbs Sampling, we have assumed the emission probabilities for words that doesn't already exist in our emission probability dictionary to be 1/(10)**10.
## Results and Comparison

The results obtained after reading all 2000 sentences of the test set were:
```
==> So far scored 2000 sentences with 29442 words.
                   Words correct:     Sentences correct:
   0. Ground truth:      100.00%              100.00%
         1. Simple:       91.75%               37.75%
            2. HMM:       93.35%               44.35%
        3. Complex:       94.36%               52.60%
----
```
As anticipated, the performance betters with each model as the dependencies among words increases.

# Part 2: Code Breaking 
For this part, we are required to perform code-breaking on an encrypted document, given a corpus of words and instructions for performing certain transformations.

## Step I:
The first step was to define initial and transition probabilities for the alphabets, based on their occurrences in the corpus. 
For initial probabilities, we created a dictionary called 'ip'. We found the first letter of each word in the corpus and using a ratio of their total occurrences in the first place to overall occurrences, we computed and stored the probabilities for each letter (which is the key) in 'ip'. For transition probabilities, we iterated from the second letter of each word, checking the previous letter and current letter and counting each time they occur together and finally dividing this by the total number of times current(previous) letter occurs in the corpus This, we stored in another dictionary 'tp', which had a tuple of a combination of two alphabets for a key and transition probabilities as values. 

## Step II:
We created a function called table_guess(), which contained the computations of our guesses of the encryption table. For the original encryption table: T, we created two entities, one to perform replacement and the other for rearrangement. The replace_table was a dictionary which contained the mapping between alphabets and other random alphabets. This was done by randomly shuffling the order of their ASCII codes and then converting this back to alphabets. The rearrange_table was created by defining a list with an ordering of four values between 0-3. Next, the guesses for the modified encryption table: T' were created. The replace_table' was created by randomly generating 'n' alphabets and changing their mapping in the dictionary. The rearrange_table' was created by randomly shuffling the order of the original rearrange table.

## Step III:
In this step, we worked on generating decryption documents D and D' using the encode function from encode.py. D was generated by passing the encoded document, replace_table and rearrange_table, while D' was generated by passing D, replace_table' and rearrange_table'.
Next, we worked on generating the probabilities of the document, by calculating the probabilities of the word, via the initial and transition probabilities. This was done separately in the function word_prob().  Basically, we calculated p(word) and p(document) as,
```
p(word) = initial_probability(first letter)*(a product of all the transition probabilities of all the letters in the word)
p(document) = product of all p(word) in the document.
```
Finally, we converted this probability of the document - P(D) into log values, as otherwise, they would become nan or 0.0 due to their very small values. Also, to avoid divide by zero error, we initialised all the probabilities to a very minute value, before they were all updated.

## Step IV:
In this step, we created a check. Basically, if the ratio of P(D)'/P(D) is greater than some threshold (here, this is subtraction because the values are in log), then we can change replace_table and rearrange_table to now hold the values of replace_table' and rearrange_table' and then update the values of what was supposed to be replace_table' and rearrange_table' to contain new modifications. If this condition is unsatisfied, then the values retain their original states.

## Step V:
We created a function called changes() to take the above values of replace_table and rearrange_table and modify them to created another set of new replace_table' and rearrange_table'. This function outputs the old values and then the new values.

## Step VI:
This step basically involves creating a loop of about a 1000-10000 iterations, in which we run Step IV over and over again until we eventually have enough positive modifications to get the decrypted document.
Since the code takes a very long time to run for a large number of iterations, we have put a timer, that will ensure that the code stops running within 10 minutes, and uses the best value of replace_table and rearrange_table to find out the decrypted document Ddash, which then becomes the output. (This timer check ensures that an output is provided within 570 seconds precisely, so in case it has to be run for more iterations to check for a better decryption, the timer section has to be commented.)

**Discussion and Citations**
In the initial phase, there was a high level discussion about the concept behind the initial table generation with Shreya Bhujbal.
During the coding process, we referred to the following links in order to get a better idea of Markov chains and the probailities and to create random strings of a given length:
https://towardsdatascience.com/markov-chains-and-hmms-ceaf2c854788
https://www.datacamp.com/community/tutorials/markov-chains-python-tutorial
https://pynative.com/python-generate-random-string/

# Part 3: Spam classification
The problem is primarily split into two parts:
* Training the emails in the training directory using bag of words model
* Classifying the emails in the test directory using a Naïve Bayes Classifier

In the first part, to train the emails using bag of words model, first, the emails in training directory have been tokenized into independent words and put into two bags (spam and notspam). A count of the spam and notspam words is calculated and stored as a dictionary.
Probability of spam emails is calculated as 
<b>P(spam|words) = number of spam files/total number of files</b>
Probability of notspam emails is calculated as 
<b>P(notspam|words) = number of notspam files/total number of files</b>

In the second part, to classify the emails in the test directory, ideally, we need to calculate the probability of each email belonging to spam and notspam. For this, we need to multiply the probability of each word in the email occurring in spam and not spam. In some cases, a word present in test directory email might not be present in the train directory. In this case, the probability value becomes zero and the model will not be able to classify it as spam or notspam, due to which an accuracy of 82.14565387627252. To improve this, a laplace smoothening has been done after tokenizing the words (reference - [https://towardsdatascience.com/unfolding-na%C3%AFve-bayes-from-scratch-2e86dcae4b01](https://towardsdatascience.com/unfolding-na%C3%AFve-bayes-from-scratch-2e86dcae4b01) and [https://en.wikipedia.org/wiki/Additive_smoothing](https://en.wikipedia.org/wiki/Additive_smoothing)). The missing words have been represented as "UnKnown" in the words dictionary 

Since all the probability values are small, multiplying them will give a very small value, leading to underflow error. To overcome this, we take log of these values and add each of the log probabilities (since log(x*y) = log(x) + log(y)) -> reference: [https://towardsdatascience.com/unfolding-na%C3%AFve-bayes-from-scratch-2e86dcae4b01](https://towardsdatascience.com/unfolding-na%C3%AFve-bayes-from-scratch-2e86dcae4b01)
Hence, in the testing step, we tokenize all the test emails and sum up the log probabilities of each of the tokenized words. If the sum of spam is higher than that of notspam, we tag it as spam, else we tag it as notspam and the result is written to output-file.

Finally, to calculate accuracy of our model, we read the actual classified values from test-groundtruth.txt and the predicted values from output-file and use scikit-learn accuracy-score method to find accuracy. The accuracy observed is 96.51527016444791

## Running the code:
The code is run using the command  <b>python3 spam.py training-directory testing-directory output-file</b>.

Please ensure that the train directory files are inside a folder "training-directory" which has two folders within it - spam and notspam and test directory files are inside a folder "testing-directory"
