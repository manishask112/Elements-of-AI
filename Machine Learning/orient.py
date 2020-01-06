import sys
import csv 
import math
import numpy as np
from matplotlib import pyplot as plt
import pickle
import pandas as pd

class NeuralNetwork(object):
    def __init__(self, epochs, learning_rate):
        self.learning_rate = learning_rate
        self.epochs = epochs

    # references to Relu and Softmax activation functions -  https://medium.com/@himanshuxd/activation-functions-sigmoid-relu-leaky-relu-and-softmax-basics-for-neural-networks-and-deep-8d9c70eed91e
    def relu_activation(self, A, W, b):
        Z = np.dot(W, A) + b
        A = np.maximum(0, Z)
        return (A)

    # reference to stabilize softmax activation - https://stackoverflow.com/questions/54880369/implementation-of-softmax-function-returns-nan-for-high-inputs
    def softmax_activation(self, A, W, b):
        Z = np.dot(W, A) + b
        expZ = np.exp(Z-np.max(Z))
        return (expZ / np.sum(expZ, axis=0))

    # reference to softmax cross entropy loss function - https://www.quora.com/Is-the-softmax-loss-the-same-as-the-cross-entropy-loss , https://www.coursera.org/learn/deep-neural-network/lecture/LCsCH/training-a-softmax-classifier
    def costFunction(self, A, y_train):
        m = y_train.shape[1]
        cost = (-1/m) * np.sum(np.multiply(y_train, np.log(A)))
        return cost

    # reference to Relu derivative - https://stackoverflow.com/questions/42042561/relu-derivative-in-backpropagation
    def relu_derivative(self, A):
        # if A > 0:
        #   return 1
        # else:
        #   return 0

        # the above code gave me an error.
        # reference code to vectorize - https://stackoverflow.com/questions/46411180/implement-relu-derivative-in-python-numpy
        return 1. * (A > 0)

    # Forward propagation - https://www.coursera.org/learn/neural-networks-deep-learning/lecture/MijzH/forward-propagation-in-a-deep-network
    # Dropout - concept and reference - https://www.coursera.org/learn/deep-neural-network/lecture/eM33A/dropout-regularization , https://www.coursera.org/learn/deep-neural-network/lecture/YaGbR/understanding-dropout
    def forwardprop(self, X, W1, b1, W2, b2, W3, b3):
        A1 = self.relu_activation(np.transpose(X), W1, b1)
        DropOut1 = np.random.rand(A1.shape[0], A1.shape[1])
        DropOut1 = DropOut1 < 0.8
        A1 = A1 * DropOut1
        A1 = A1 / 0.8
        A2 = self.relu_activation(A1, W2, b2)
        DropOut2 = np.random.rand(A2.shape[0], A2.shape[1])
        DropOut2 = DropOut2 < 0.8
        A2 = A2 * DropOut2
        A2 = A2 / 0.8
        A3 = self.softmax_activation(A2, W3, b3)
        return (A1, DropOut1, A2, DropOut2, A3)

    # Backpropagation - concept and formula reference - https://www.coursera.org/learn/neural-networks-deep-learning/lecture/znwiG/forward-and-backward-propagation
    # Dropout - concept and reference - https://www.coursera.org/learn/deep-neural-network/lecture/eM33A/dropout-regularization , https://www.coursera.org/learn/deep-neural-network/lecture/YaGbR/understanding-dropout
    def backprop(self, X, y, DropOut1, W1, A1, DropOut2, W2, A2, W3, A3):
        m = X.shape[1]
        # softmax derivative is (y_predicted - y) -> reference - https://www.coursera.org/learn/deep-neural-network/lecture/LCsCH/training-a-softmax-classifier
        # implemented using the below formulas (reference - https://www.coursera.org/learn/neural-networks-deep-learning/lecture/znwiG/forward-and-backward-propagation)
        # dW = (1/m)*dZ*transpose(A)
        # db = (1/m)*sum(dZ)
        # dA = transpose(W)*dZ

        delta3 = A3 - y
        dW3 = 1. / m * np.dot(delta3, np.transpose(A2))
        db3 = 1. / m * np.sum(delta3, axis=1, keepdims=True)
        dA2 = np.dot(np.transpose(W3), delta3)
        dA2 = dA2 * DropOut2
        dA2 = dA2 / 0.8
        delta2 = np.multiply(dA2, self.relu_derivative(A2))
        dW2 = 1. / m * np.dot(delta2, np.transpose(A1))
        db2 = 1. / m * np.sum(delta2, axis=1, keepdims=True)
        dA1 = np.dot(np.transpose(W2), delta2)
        dA1 = dA1 * DropOut1
        dA1 = dA1 / 0.8
        delta1 = np.multiply(dA1, self.relu_derivative(A1))
        dW1 = 1. / m * np.dot(delta1, np.transpose(X))
        db1 = 1. / m * np.sum(delta1, axis=1, keepdims=True)
        return dW1, dW2, dW3, db1, db2, db3

    def GradientDescent(self, W1, W2, W3, b1, b2, b3, dW1, dW2, dW3, db1, db2, db3):

        W1 = W1 - (self.learning_rate)*dW1
        b1 = b1 - (self.learning_rate)*db1

        W2 = W2 - (self.learning_rate)*dW2
        b2 = b2 - (self.learning_rate)*db2

        W3 = W3 - (self.learning_rate)*dW3
        b3 = b3 - (self.learning_rate)*db3

        return (W1, W2, W3, b1, b2, b3)

    def fit(self, inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons, x_train, y_train):
        # initializing random weights dimensions as (S_j+1 x S_j) - reference - https://www.coursera.org/learn/machine-learning/supplement/Bln5m/model-representation-i, https://www.coursera.org/learn/neural-networks-deep-learning/lecture/Rz47X/getting-your-matrix-dimensions-right
        # First layer has weights of dimension 30 x 192
        # multipying by 0.01 to normalize and start with very small values of weights close to zero - reference- https://towardsdatascience.com/random-initialization-for-neural-networks-a-thing-of-the-past-bfcdd806bf9e
        W1 = np.random.randn(hiddenLayerNeurons, inputLayerNeurons)*0.01
        # First layer has bias of dimension 30 x 192
        b1 = np.zeros(shape=(hiddenLayerNeurons, 1))
        # Second layer has weights of dimension 30 x 30
        W2 = np.random.randn(hiddenLayerNeurons, hiddenLayerNeurons)*0.01
        # Second layer has bias of dimension 30 x 1
        b2 = np.zeros(shape=(hiddenLayerNeurons, 1))
        # Third layer has weights of dimension 4 x 30
        W3 = np.random.randn(outputLayerNeurons, hiddenLayerNeurons)*0.01
        # Third layer has bias of dimension 4 x 1
        b3 = np.zeros(shape=(outputLayerNeurons, 1))

        # minibatch gradient descent - reference - https://wiseodd.github.io/techblog/2016/06/21/nn-sgd/ , https://towardsdatascience.com/gradient-descent-in-python-a0d07285742f
        numberOfSamples = x_train.shape[0]
        epochs = []
        costs = []
        for i in range(self.epochs):
        # np.random.permutation (vs shuffle) - reference - https://stackoverflow.com/questions/15474159/shuffle-vs-permute-numpy
            batch_index = np.random.permutation(numberOfSamples)
            # store shuffled train data
            x_train = x_train[batch_index]
            y_train = y_train[batch_index]

            # forward propagation
            (A1, DropOut1, A2, DropOut2, A3) = self.forwardprop(x_train, W1, b1, W2, b2, W3, b3)
            # back propagation
            dW1, dW2, dW3, db1, db2, db3 = self.backprop(np.transpose(x_train), np.transpose(y_train), DropOut1, W1, A1, DropOut2, W2, A2, W3, A3)
            # Gradient descent
            (W1, W2, W3, b1, b2, b3) = self.GradientDescent(W1, W2, W3, b1, b2, b3, dW1, dW2, dW3, db1, db2, db3)

            costs.append(self.costFunction(A3, np.transpose(y_train)))
            epochs.append(i+1)

        # plot to see if cost is decreasing at every epoch (epochs vs cost)
        plt.plot(epochs, costs)
        plt.xlabel('Epochs')
        plt.ylabel('Costs')
        plt.xticks(range(1, 3000, 250))
        plt.title('Epochs vs Costs')
        plt.legend(['curve: Epochs vs Costs'])
        plt.savefig('epoch_vs_cost.png')

        plt.show()
        return (W1, W2, W3, b1, b2, b3)

    def evaluate(self, X, Y, W1, b1, W2, b2, W3, b3):
        # run forward propagation once with the final weights and biases
        (A1, DropOut1, A2, DropOut2, A3) = self.forwardprop(X, W1, b1, W2, b2, W3, b3)
        # calculate cost
        cost = (self.costFunction(A3, np.transpose(Y)))
        # accuracy calculation reference - https://stackoverflow.com/questions/46800774/how-to-find-accuracy-of-a-model-in-cnn
        A3 = np.argmax(A3, axis=0)
        Y = np.argmax(Y, axis=1)
        accuracy = (A3 == Y).mean()
        return (cost, accuracy * 100,A3)
    
# =======================================================================================    
# ========================CODE FOR DECISION TREE STARTS================================== 

# Each node in the decision tree is of type DecisionTree
# It holds pointers to its children, as well as the relevant question to be asked at that node
class DecisionTree:
    def __init__(self,threshold,feature,right_tree,left_tree):
        self.threshold=threshold
        self.feature=feature
        self.right_tree=right_tree
        self.left_tree=left_tree

# Creates a numpy array of the train set and passes it to the function that builds the tree along with maximum depth
def tree_train(max_depth,filename):
    x_train = []
    file = open(filename, 'r')
    for line in file:
        sample=line.split()
        x_train.append([int(k) for k in sample[1:]])  
    x_train=np.array(x_train)
    root=create_decision_tree(x_train,-1,max_depth)
    return root

# Recursively builds the tree and returns root of the decision tree
def create_decision_tree(samples,curr_depth,max_depth):
    curr_depth+=1
#   Create subtrees based on best split
    impurity,threshold,feature,f=best_split(samples)
    
#   Condition for leaf node, recursion base condition.
#   When maximum depth has reached or when no further splits are possible
    if curr_depth==max_depth or impurity==0 or impurity==-100:
        y_train=samples[:,0]
#       Find label with maximum frequency and that becomes the prediction at that leaf node
        orientation,frequency = np.unique(y_train, return_counts=True)
        return orientation[frequency.tolist().index(max(frequency))]
    
#   If not base condition, split based on best split found
    right_samples,left_samples=split(samples,threshold,feature)
    
#   Recursion calls
    right_tree=create_decision_tree(right_samples,curr_depth,max_depth)
    left_tree=create_decision_tree(left_samples,curr_depth,max_depth)
    
    return DecisionTree(threshold,feature,right_tree,left_tree)

# To find the value, feature that gives the split with the least impurity; the best split
def best_split(samples):
    impurity=calculate_gini_impurity(samples)
    best_info_gain=-100
    features=[]
    for i in range(1,len(samples[0])):
        
#       Sorting samples based on feature under consideration for ease of splitting and efficiency
#       Inspired by Joachim Valente's blog on decision tree:
#       https://towardsdatascience.com/decision-tree-from-scratch-in-python-46e99dfea775
        samples = samples[samples[:,i].argsort()]
        
        for j in range(0,len(samples)):
            if samples[j][i]==samples[j-1][i]:
                continue

            right=samples[j:]
            wrong=samples[0:j]
            
#           When there is no split, that value can be discarded
            if right.size==0 or wrong.size==0:
                continue
            
#           Calculating information gain and recording, the best one
            info_gain=impurity-(((len(right)*calculate_gini_impurity(right))+(len(wrong)*calculate_gini_impurity(wrong)))/len(samples))
            if info_gain>=best_info_gain:
                if i not in features:
                    features.append(i)
                best_info_gain=info_gain
                best_feature=i
                best_threshold=samples[j][i]
                
#   When no split is possible, for the sake of returning            
    if best_info_gain==-100:
        best_threshold=None
        best_feature=None
        
    return (best_info_gain,best_threshold,best_feature,features)  

# Function to calculate Gini impurity of samples at a node
def calculate_gini_impurity(samples):
    y_train=samples[:,0]
    orientation,frequency = np.unique(y_train, return_counts=True)
    impurity=0
    for i in frequency:
        impurity=impurity+(i/len(samples))**2
    impurity=1-impurity
    
    return impurity

# To split data into right and left subtrees based on best feature and value
def split(samples,threshold,feature_id):
    right=[]
    wrong=[]
    s=samples.tolist()
    for i in range(0,len(s)):
        if s[i][feature_id]>=threshold:
            right.append(s[i])
        else:
            wrong.append(s[i])   
    return np.array(right),np.array(wrong) 

# Predict function reads test set and passes each sample to tree and returns accuracy
# Writes predictions to output file
def tree_predict(root_DT,filename):
    file = open(filename, 'r')
    photo_id=[]
    y_test=[]
    x_test=[]
    for line in file:
        sample=line.split()
        photo_id.append(sample[0]) 
        y_test.append(int(sample[1]))
        x_test.append([int(k) for k in sample[2:]])
    file.close()    
    y_pred=[]
    output=[]
    correct_count=0
    file2 = open('output.txt','w+')
    for i in range(0,len(x_test)):
        y_pred.append(tree_traverse(x_test[i],root_DT))  
        file2.write(str(photo_id[i])+" "+str(y_pred[i])+"\n")
        if y_pred[i] == y_test[i]:
            correct_count+=1

    accuracy=float(correct_count)/float(len(y_pred))*100
    file2.close()  
    return accuracy 

# Each sample passed to this function traverses through the decision tree to its prediction
# The leaf node is returned which contains a single label: 0 or 90 or 180 or 270
def tree_traverse(x_test_sample,root_DT):
    node=root_DT
    while isinstance(node,DecisionTree):
        if x_test_sample[node.feature-1]>=node.threshold:
            node=node.right_tree
        else:
            node=node.left_tree
    return node
  
# ========================CODE FOR DECISION TREE ENDS==================================    
# =====================================================================================
# ========================CODE FOR KNN BEGINS==========================================
class KNearestNeighbor:
    def __init__(self):
        self.train_pixels = []
        self.train_orient = []
        self.train_images = []
        self.test_pixels = []
        self.test_orient = []
        self.test_images = []
        self.match = 0
    
    def train(self, file_name):
        train_data = csv.reader(open(file_name), delimiter = " ")
        for element in train_data:
            self.train_images.append(element[0])
            self.train_orient.append(int(element[1]))
            self.train_pixels.append(np.asarray(list(map(int, element[2:]))))
#         print("Length1", len(self.train_orient))
        
    def test(self, file_name):
        test_data = csv.reader(open(file_name), delimiter = " ")
        for element in test_data:
            self.test_images.append(element[0])
            self.test_orient.append(int(element[1]))
            self.test_pixels.append(np.asarray(list(map(int, element[2:]))))

    def model_function(self, file_name, model_file):
        self.train(file_name)
        model = {'images':self.train_images,'pixels' : np.asarray(list(self.train_pixels)), 'orientations': self.train_orient }
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)

    def distance_voting(self, test_file, model_file, k = 200):
        text = []
        with open(model_file, 'rb') as f:
           training = pickle.load(f)
        train_pix = np.array(list(training['pixels']))
        train_orientations = training['orientations']
        self.test(test_file)
        test_image = self.test_images
        self.test_pixels = np.array(list(self.test_pixels))
        shape_test = float(len(self.test_pixels))
        
        
#       https://www.afternerd.com/blog/python-enumerate/
        for index,values in enumerate(self.test_pixels):
            sub = np.square(train_pix-values)
            euclid = (np.sum((sub), axis = 1))
            idx = np.argpartition(euclid,k)
#             print("ArgP",len(idx[:k]))
            idx1  = idx[:k]
#             print("MAX",max(idx1))
            tr_dist = euclid[idx1]
#             print("Lendist1",len(dist1))
            list1=[]
#             print("LENGTH2",len(self.train_orient))
#             print("LENGTH3",len(train_class))
            for i in idx1:
                list1.append(train_orientations[i])
#             print("lol",list1)
#             print("lollength",len(list1))
            train_orient = max(list1,key = list1.count)
#             print("LENGTHA",len(train_image))
            text.append(str(test_image[index])+" "+str(train_orient))
            file2 = open('output.txt','w+')
            file2.write("\n".join(text))
            if train_orient == self.test_orient[index]:
                self.match =  self.match + 1
        print('Accuracy:', (self.match/shape_test)*100,'%')
# class KNearestNeighbor:
#     def __init__(self):
#         self.train_pixels = []
#         self.train_orient = []
#         self.train_images = []
#         self.test_pixels = []
#         self.test_orient = []
#         self.test_images = []
#         self.match = 0
    
#     def train(self, file_name):
#         print("1111111111111")
#         train_data = csv.reader(open(file_name), delimiter = " ")
#         for element in train_data:
#             self.train_images.append(element[0])
#             self.train_orient.append(int(element[1]))
#             self.train_pixels.append(np.asarray(list(map(int, element[2:]))))
#         print("22222222222")

#     def test(self, file_name):
#         print("333333333333")
#         test_data = csv.reader(open(file_name), delimiter = " ")
#         for element in test_data:
#             self.test_images.append(element[0])
#             self.test_orient.append(int(element[1]))
#             self.test_pixels.append(np.asarray(list(map(int, element[2:]))))
#         print("4444444444444")

#     def model_function(self, file_name, model_file):
#         print("55555555555555")
#         self.train(file_name)
#         model = {'pixels' : np.asarray(list(self.train_pixels)), 'orientations': self.train_orient }
#         with open(model_file, 'wb') as f:
#             pickle.dump(model, f)
#         print("666666666666666")

#     def distance_voting(self, test_file, model_file, k = 200):
#         print("77777")
#         text = []
#         with open(model_file, 'rb') as f:
#            training = pickle.load(f)
#         print("training",training)
#         train_pix = np.array(list(training['pixels']))
#         train_class = training['orientations']
#         self.test(test_file)
#         self.test_pixels = np.array(list(self.test_pixels))
#         shape_test = float(len(self.test_pixels))
#         print("888888888888")
# #       https://www.afternerd.com/blog/python-enumerate/
#         for index,values in enumerate(self.test_pixels):
#             sub = np.square(train_pix-values)
#             euclid = (np.sum((sub), axis = 1))
#             idx = np.argpartition(euclid,k)
# #             print("ArgP",len(idx[:k]))
#             idx1  = idx[:k]
# #             print("MAX",max(idx1))
#             tr_dist = euclid[idx1]
# #             print("Lendist1",len(dist1))
#             list1=[]
#             print("Length of train_orient",len(self.train_orient))
#             for i in idx1:
#                 print("I",i)
#                 list1.append(self.train_orient[i])
#                 print("List1",list1)
# #             print("lol",list1)
# #             print("lollength",len(list1))
#             train_orient = max(list1,key = list1.count)
#             text.append(str(self.train_images[index])+" "+str(train_orient))
#             with open('knn_predictions.txt', 'wt') as output:
#                 output.write("\n".join(text))
#             if train_orient == self.test_orient[index]:
#                 self.match =  self.match + 1
#         print("999999999")
#         print('Accuracy:', (self.match/shape_test)*100,'%')
# ========================CODE FOR KNN ENDS============================================     
# =====================================================================================        
# main
if sys.argv[4] == "nnet" or sys.argv[4] == "best":
    if sys.argv[1] == "train":
        print ("Running Neural Network Classifier - Train")
        # read train data
        filepath = sys.argv[2]
        input_arr = []
        target_arr = []
        with open(filepath) as infile:
            lines = infile.readlines()
            for line in lines:
                temp = []
                target_arr.append(int(line.split()[1]))
                for n in line.split()[2:]:
                    temp.append(int(n))
                input_arr.append(temp)

        x_train = np.array(input_arr)
        y_train = np.array(target_arr)

        # one hot encoding of y_train
        y_train = (pd.get_dummies(y_train))
        y_train = (y_train.values)

        # input layer -> 192 neurons since the image is 8x8x3=192
        inputLayerNeurons = 192
        # two hidden layers each consisting of 30 neurons
        hiddenLayerNeurons = 30
        # output -> 4 classes
        outputLayerNeurons = 4
        epochs = 2000
        learning_rate = 0.01

        print("Running for", epochs, "epochs..")
        # initializing neural network class object
        nn = NeuralNetwork(epochs, learning_rate)
        # fitting the train set
        (W1, W2, W3, b1, b2, b3) = nn.fit(inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons, x_train, y_train)
        # write weights and bias to model file
        param_arr = [W1, b1, W2, b2, W3, b3]
        model = nn,param_arr
        with open(sys.argv[3], 'wb') as f:
            pickle.dump(model, f)

        # evaluating the train set for each cross validation iteration
        train_cost, train_accuracy, A3 = nn.evaluate(x_train, y_train, W1, b1, W2, b2, W3, b3)
        print("Train cost is:", train_cost, "and Train accuracy is:", train_accuracy)

    elif sys.argv[1] == "test":
        print ("Running Neural Network Classifier - Test")
        # read test data
        filepath = sys.argv[2]
        input_arr = []
        target_arr = []
        filenames = []
        with open(filepath) as infile:
            lines = infile.readlines()
            for line in lines:
                temp = []
                filenames.append(line.split()[0])
                target_arr.append(int(line.split()[1]))
                for n in line.split()[2:]:
                    temp.append(int(n))
                input_arr.append(temp)

        x_test = np.array(input_arr)
        y_test = np.array(target_arr)

        # one hot encoding of y_test
        y_test = (pd.get_dummies(y_test))
        y_test = (y_test.values)

        # read weights and bias for model file
        with open(sys.argv[3], 'rb') as f:
           nn,param_arr = pickle.load(f)
        W1, b1, W2, b2, W3, b3 = list(param_arr[0]), list(param_arr[1]), list(param_arr[2]), list(param_arr[3]), list(param_arr[4]), list(param_arr[5])

        # evaluating the test set
        test_cost, test_accuracy, A3 = nn.evaluate(x_test, y_test, W1, b1, W2, b2, W3, b3)
        print("\nTest cost is:", test_cost, "and Test accuracy is:", test_accuracy)
        with open('output.txt', 'w') as f:
            write_str = ""
            for i in range(len(filenames)):
                if A3[i] == 1:
                    A3[i] = 90
                elif A3[i] == 2:
                    A3[i] = 180
                elif A3[i] == 3:
                    A3[i] = 270
                output = filenames[i]+" "+str(A3[i])
                write_str += output+"\n"
            f.write(write_str)
            
elif sys.argv[4] == "tree":
    max_depth=7
    if sys.argv[1] == "train":
        filename = sys.argv[2]
        root=tree_train(max_depth,filename) 
        file3 = open(sys.argv[3], 'wb') 
        pickle.dump(root, file3)
        file3.close()
    elif sys.argv[1] == "test": 
        file3 = open(sys.argv[3], 'rb') 
        root_DT = pickle.load(file3)
        filename = sys.argv[2]
        accuracy=tree_predict(root_DT,filename) 
        print("accuracy for a maximum depth of "+str(max_depth)+" : "+str(accuracy)+"%") 
    

elif sys.argv[4] == "nearest":
    # knn main function (starter code)
    knn = KNearestNeighbor()
#You may use anyfile format you’d like for modelfile.txt; the important thing is that your test code knows how to interpret it.
    if sys.argv[1] == "train":
#         model_file = sys.argv[3]    
        knn.model_function(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "test":
#         model_file = sys.argv[3]
        knn.distance_voting(sys.argv[2], sys.argv[3])
    pass
