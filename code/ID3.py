import math
from operator import itemgetter

#Creates tree
def ID3_train(examples, default, IG, depth):
    
    #Checks if all the examples are of the same class
    flagPos=False
    flagNeg=False
    flag=False
    for i in range (0,len(examples)):
        if examples[i][0]>=5:
            flagPos=True
        elif examples[i][0]<5:
            flagNeg=True
        if flagPos==True and flagNeg==True:
            flag=True
    
    #Checks if there are any examples
    if not examples:
        return default
    #Checks if all the examples are of the same class
    elif not flag:
        if examples[0][0] >= 5:
            return "positive"
        else:
            return "negative"
    #Checks if there are any attributes
    elif not IG:
        pos = 0
        neg = 0
        for i in examples:
            if i[0] >= 5:
                pos += 1
            else:
                neg += 1

        #Finds default
        if pos > neg:
            return "positive"
        elif pos < neg:
            return "negative"
        else:
            return default
    else:

        #Finds default
        pos = 0
        neg = 0
        for i in examples:
            if i[0] >= 5:
                pos += 1
            else:
                neg += 1
        if pos >= neg:
            default = "positive"
        else:
            default = "negative"
        #Finds best attribute
        best = IG[0][0]
        IG.pop(0)
        current = Node(pos, neg, best)
        #Checks if maximum depth has been reached
        if depth == 1000:
            return current
        if current.pos_ex / current.total_ex >= 0.85 or current.neg_ex / current.total_ex >= 0.85:
            return current
        #Fills two lists with examples for the children nodes
        l_ex = []
        r_ex = []
        for i in range(0, len(examples)):
            flag = False
            for j in range(1, len(examples[i])):
                
                if examples[i][j][0] == best:
                    flag = True
                    l_ex.append(examples[i])
            if flag == False:
                r_ex.append(examples[i])
        current.left = ID3_train(l_ex, default, IG, depth + 1)
        current.right = ID3_train(r_ex, default, IG, depth + 1)
        return current

#Return positive outcome
def ID3_predict(example, tree):
    
    while not isinstance(tree, str):
        if tree.left == None and tree.right == None:
            if tree.total_ex == 0:
                return 0
            return tree.pos_ex/tree.total_ex
        flag = False
        for i in range(1, len(example)):
            if tree.attr_ind == example[i][0]:
                flag = True
                break
        if flag == True:
            tree = tree.left
        else:
            tree = tree.right
    if tree == "positive":
        return 1
    else:
        return 0
        
        
#Calculates information gain
def IG_calc(examples, Nattr):

    N = len(examples)
    pos = 0
    neg = 0
    for i in examples:
        if i[0] >= 5:
            pos += 1
        else:
            neg += 1
    
    #Calculates the a priori probalities
    Ppos = (pos + 1)/(N + 2)
    Pneg = (neg + 1)/(N + 2)
    #Calculates total entropy
    H = -Ppos*math.log(Ppos, 2) -Pneg*math.log(Pneg, 2)
    IG = []
    for a in range(0, Nattr):
        counter1 = 0 #how many examples contain attribute[a]
        counter2 = 0 #how many examples that contain attribute[a] are positive
        counter3 = 0 #how many examples that do not contain attribute[a] are positive
        for i in range(0, len(examples)):
            flag = False
            for j in range(1, len(examples[i])):
                if a == examples[i][j][0]:
                    flag = True
                    counter1 += 1
                    if examples[i][0] >= 5:
                        counter2 += 1
            if flag == False:
                if examples[i][0] >= 5:
                    counter3 += 1
        #Calculates the probability of existence of attribute
        Pattr = (counter1 + 1)/(N + 2)
        #H(C|X = 0) = - Σ P(C = c| X = 0) * logP(C = c| X = 0)
        H0 = - ((counter3 + 1)/(N - counter1 + 2))*math.log((counter3 + 1) / (N - counter1 + 2), 2) - ((N - counter1 - counter3 + 1) / (N - counter1 + 2))*math.log((N - counter1 - counter3 + 1)/ (N - counter1 + 2), 2)
        #H(C|X = 1) = - Σ P(C = c| X = 1) * logP(C = c| X = 1)
        H1 = - ((counter2 + 1) / (counter1 + 2))*math.log((counter2 + 1) / (counter1 + 2), 2) - ((counter1 - counter2 + 1)/ (counter1 + 2))*math.log((counter1 - counter2 + 1) / (counter1 + 2), 2)
        #Calculates information gain for attribute
        if (H - Pattr * H1 - (1 - Pattr) * H0) != 0:
            IG.append([a, H - Pattr * H1 - (1 - Pattr) * H0])
    IG = sorted(IG, key = itemgetter(1), reverse = True)
    return IG



class Node:
    
    def __init__(self, pos_ex, neg_ex, attr_ind):
        self.right = None
        self.left = None
        #The number of positive examples in this node
        self.pos_ex = pos_ex
        #The number of negative examples in the node
        self.neg_ex = neg_ex
        self.total_ex = self.pos_ex + self.neg_ex
        #The index of the attribute in vocab
        self.attr_ind = attr_ind