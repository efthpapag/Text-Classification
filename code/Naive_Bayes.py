import math
from typing import Counter


class Naive_Bayes :

    def __init__ (self) :

        self.Ppos = 0
        self.Pneg = 0
        self.AttrProb = None

    def Train (self, examples, attributes) :

        N = len(examples)
        pos = 0
        neg = 0

        for i in examples :
            if i[0] >= 5 :
                pos += 1
            else :
                neg += 1

        #Calculates a priori probabilities
        self.Ppos = (pos + 1) / (N + 2) 
        self.Pneg = (neg + 1) / (N + 2)
        self.AttrProb = []

        for a in range(0,len(attributes)) :
            counter0 = 0
            counter1 = 0
            for i in examples :
                flag = False
                for j in range(1, len(i)):
                    if i[j][0] == a :
                        flag = True
                if flag :
                    if i[0] >= 5 :
                        counter1 += 1
                    else :
                        counter0 += 1
            #Calculates the probabilities of each class for attribute
            self.AttrProb.append([(counter0 + 1) / (neg + 2), (counter1 + 1) / (pos + 2), (counter0 + counter1 + 1) / (pos + neg + 2)])

    def Predict (self, example, Nattr) :#returns positive probability

        PX = 1
        P1 = math.log2(self.Ppos)
        P0 = math.log2(self.Pneg)

        for i in range(0, Nattr):
            flag = False
            for j in range(1, len(example)):
                if i == example[j][0] :
                    flag = True
            if flag == True:
                P1 = P1 + math.log2(self.AttrProb[i][1])
                P0 = P0 + math.log2(self.AttrProb[i][0])

            else:
                P1 = P1 + math.log2(1 - self.AttrProb[i][1])
                P0 = P0 + math.log2(1 - self.AttrProb[i][0])
            PX = PX + math.log2(self.AttrProb[i][2])
        P1 = 2**(P1 / PX)
        P0 = 2**(P0 / PX)
        return P1