from File_Readers import *
from ID3 import *
from Naive_Bayes import *
from Random_Forest import *
import numpy as np 
import matplotlib.pyplot as plt
from texttable import Texttable
import matplotlib.ticker as ticker
import sys
sys.setrecursionlimit(1000000)


#calculates accuracy
def acc_calc(predictions, examples):
    accuracy = []
    accuracy.append(0)
    corr_count = 0
    for i in range(0, len(examples)):
        if examples[i][0] >= 5:
            if predictions[i] == "positive":
                corr_count += 1
        else:
            if predictions[i] == "negative":
                corr_count += 1
        accuracy.append(corr_count / (i + 1))
    return accuracy

#Draws accuracy-examples graph 
def acc_graph(acc_train,acc_test):
    x1 = np.arange(start = 1, stop = len(acc_train), step = 1)
    x2 = np.arange(start = 1, stop = len(acc_test), step = 1)
    fig, ax = plt.subplots(figsize = (10,8))
    plt.xlabel("percentage of examples")
    plt.ylabel("accuracy")
    ax.plot(x1, acc_train[1::], 'r', label = 'train')
    ax.plot(x2, acc_test[1::], 'g', label = 'test')
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(len(acc_test) - 1))
    ax.grid()
    ax.legend()
    plt.show()

#creates accuracy-examples table 
def acc_table(acc_test, acc_train) :
    t = Texttable()
    list = []
    list.append(['accuracy test', 'accuracy train', 'percentage of examples'])
    l = max(len(acc_test), len(acc_test))
    for i in range(1, l) :
        if(len(acc_test) - 1 < i) :
            c1 = 0
        else:
            c1 = acc_test[i]
        if(len(acc_train) - 1 < i) :
            c2 = 0
        else:
            c2 = acc_train[i]
        list.append([c1, c2, str(((i + 1) * 100) / l) + "%"])
    t.add_rows(list)
    file = open("accuracy.txt", "w")
    file.write(t.draw())

#calculates precision
def pre_calc(predictions,examples):
    TposPos=0       #True postive for positive of Positive class
    FposPos=0       #False postive for positive of Positive class
    TposNeg=0       #True postive for positive of Negative class
    FposNeg=0       #False postive for positive of Negative class
    for i in range (0, len(examples)):
        if predictions[i] == "positive":
            if examples[i][0] >= 5:
                TposPos += 1
            else:
                FposPos += 1
        else:
            if examples[i][0] >= 5:
                FposNeg += 1
            else:
                TposNeg += 1
    precPos = 0
    precNeg = 0
    if (TposPos + FposPos != 0):
        precPos = TposPos / (TposPos + FposPos)
    if (TposNeg + FposNeg != 0):
        precNeg = TposNeg / (TposNeg + FposNeg)
    return (precPos + precNeg) / 2

#calculates recall
def rec_calc(predictions,examples):
    TposPos = 0         #True postive for positive of Positive class
    FnegPos = 0         #False postive for positive of Positive class
    TposNeg = 0         #True postive for positive of Negative class
    FnegNeg = 0         #False postive for positive of Negative class
    for i in range(0, len(examples)):
        if examples[i][0] >= 5:
            if predictions[i] == "positive":
                TposPos += 1
            else:
                FnegPos += 1
        else:
            if predictions[i] == "positive":
                FnegNeg += 1
            else:
                TposNeg += 1
    precPos = 0
    precNeg = 0
    if (TposPos + FnegPos != 0):
        precPos = TposPos / (TposPos + FnegPos)
    if (TposNeg + FnegNeg != 0):
        precNeg = TposNeg / (TposNeg + FnegNeg)
    return (precPos + precNeg) / 2


#Draws precision-recall graph 
def pre_rec_graph(precision, recall):
    fig, ax = plt.subplots(figsize = (10, 8))
    plt.xlabel("recall")
    plt.ylabel("precision")
    ax.plot(recall, precision , 'b', label = 'precision-recall curve')
    ax.grid()
    ax.legend()
    plt.show()

#creates precision-recall table
def pre_rec_table(precision, recall) :
    t = Texttable()
    list = []
    list.append(['recall', 'precision'])
    for i in range(0, len(precision)) :
        list.append([precision[i], recall[i]])
    t.add_rows(list)
    file = open("precision-recall.txt", "w")
    file.write(t.draw())

#calculates F1
def F1_calc(predictions, examples):
    precision = []
    TposPos = 0       #True postive for positive of Positive class
    FposPos = 0       #False postive for positive of Positive class
    TposNeg = 0       #True postive for positive of Negative class
    FposNeg = 0       #False postive for positive of Negative class

    for i in range(0, len(examples)):
        if predictions[i] == "positive":
            if examples[i][0] >= 5:
                TposPos += 1
            else:
                FposPos += 1
        else :
            if examples[i][0] >= 5:
                FposNeg += 1
            else:
                TposNeg += 1
        if TposPos + FposPos == 0:
            precision.append((TposNeg / (TposNeg + FposNeg)) / 2)
        elif TposNeg + FposNeg ==  0:
            precision.append((TposPos / (TposPos + FposPos)) / 2)
        else:
            precision.append(((TposPos / (TposPos + FposPos)) + (TposNeg / (TposNeg + FposNeg))) / 2)

    recall = []
    TposPos = 0         #True postive for positive of Positive class
    FnegPos = 0         #False postive for positive of Positive class
    TposNeg = 0         #True postive for positive of Negative class
    FnegNeg = 0         #False postive for positive of Negative class

    for i in range(0, len(examples)):
        if examples[i][0] >= 5:
            if predictions[i] == "positive":
                TposPos += 1
            else:
                FnegPos += 1
        else :
            if predictions[i] == "positive":
                FnegNeg += 1
            else:
                TposNeg += 1
        if TposPos + FnegPos == 0 and TposNeg + FnegNeg ==  0:
            recall.append(0)
        elif TposPos + FnegPos == 0:
            recall.append((TposNeg / (TposNeg + FnegNeg)) / 2)
        elif TposNeg + FnegNeg ==  0:
            recall.append((TposPos / (TposPos + FnegPos)) / 2)
        else:
            recall.append(((TposPos / (TposPos + FnegPos)) + (TposNeg / (TposNeg + FnegNeg))) / 2)

    F1 = []
    for i in range(0, len(precision)):
        if precision[i] + recall[i] == 0:
            F1.append(0)
        else: 
            F1.append((2 * precision[i] * recall[i]) / (precision[i] + recall[i]))
    return F1

#Draws F1-examples graph
def F1_graph(F1):
    x = np.arange(start = 1, stop = len(F1), step = 1)
    fig, ax = plt.subplots(figsize = (10,8))
    plt.xlabel("percentage of examples")
    plt.ylabel("F1")
    ax.plot(x, F1[1::], 'b', label = 'F1')
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(len(F1) - 1))
    ax.grid()
    ax.legend()
    plt.show()

#creates F1-examples table
def F1_table(F1) :
    t = Texttable()
    list = []
    list.append(['F1', 'percentage of examples'])
    for i in range(1, len(F1)) :
        list.append([F1[i], str(((i+1) * 100) / len(F1)) + "%"])
    t.add_rows(list)
    file = open("F1.txt", "w")
    file.write(t.draw())

#creates predictions vs real results table
def write_results(predictions_test,examples):
    t = Texttable()
    list = []
    list.append(['Prediction', 'Real'])
    for i in range(0, len(examples)) :
        if examples[i][0] >= 5 :
            list.append([predictions_test[i], "positive"])
        else:
            list.append([predictions_test[i], "negative"])
    t.add_rows(list)
    file = open("Results.txt", "w")
    file.write(t.draw())

#Classifies an example
def final_result(prob, threshold, default):
    if (prob > threshold):
        return "positive"
    elif (prob < threshold):
        return "negative"
    else:
        return default

vocab_path = input("Give path to vocabulary file : ")
vocab = read_vocab(vocab_path)

labeledBow_path = input("Give path to labeledBow file for training : ")
labeledBow_train = read_labeledBow(labeledBow_path)

labeledBow_path = input("Give path to labeledBow file for testing: ")
labeledBow_test = read_labeledBow(labeledBow_path)

alg = input("Select : ID3, Naive Bayes, Random Forest : ")


while (alg != "ID3") and (alg != "Naive Bayes") and (alg != "Random Forest") :
    alg = input("Invalid Input\nSelect : ID3, Naive Bayes, Random Forest")

random.shuffle(labeledBow_train)
#finds default
pos = 0
neg = 0
for i in labeledBow_train:
    if i[0] >= 5:
        pos += 1
    else:
        neg += 1
if pos >= neg:
    default = "positive"
else:
    default = "negative"

flag = False
while flag == False:
    flag = True

    if(alg == "ID3") :
        tree=ID3_train(labeledBow_train, default, IG_calc(labeledBow_train,len(vocab)),0)
        predictions_test = []
        predictions_train = []
        prob_test = []
        for i in labeledBow_test :
            prob_test.append(ID3_predict(i, tree))
            predictions_test.append(final_result(prob_test[-1], 0.5, default))
        for i in labeledBow_train :
            predictions_train.append(final_result(ID3_predict(i, tree), 0.5, default))
        #write results
        write_results(predictions_test,labeledBow_test)
        #calculate accuracy
        acc_train = acc_calc(predictions_train,labeledBow_train)
        acc_test = acc_calc(predictions_test,labeledBow_test)
        #graph for accuracy
        acc_graph(acc_train,acc_test)
        #table for accuracy
        acc_table(acc_test, acc_train)
        #calculate precision and recall for different thresholds
        threshold = np.arange(start = 0.1, stop = 1.0, step = 0.1)
        precisions = []
        recalls = []
        for i in threshold:
            predictions_test = []
            for j in prob_test:
                predictions_test.append(final_result(j, i, default))
            precisions.append(pre_calc(predictions_test, labeledBow_test))
            recalls.append(rec_calc(predictions_test, labeledBow_test))
        #create graph and table for precision-recall
        pre_rec_graph(precisions, recalls)
        pre_rec_table(precisions, recalls)
        #calculate F1 and create graph and table for F1
        F1=F1_calc(predictions_test, labeledBow_test)
        F1_graph(F1)
        F1_table(F1)
    elif(alg == "Naive Bayes") :
        classifier = Naive_Bayes()
        classifier.Train(labeledBow_train,vocab)
        predictions_test = []
        predictions_train = []
        prob_test = []
        prob_train = []
        for i in labeledBow_test :
            prob_test.append(classifier.Predict(i, len(vocab)))
        B = max(prob_test)
        A = min(prob_test)
        for i in range(0, len(prob_test)):
            predictions_test.append(final_result(prob_test[i], B - (0.5 * (B - A)), default))
        for i in labeledBow_train :
            prob_train.append(classifier.Predict(i, len(vocab)))
        B2 = max(prob_train)
        A2 = min(prob_train)
        for i in range(0,len(prob_train)):
            predictions_train.append(final_result(prob_train[i],  B2 - (0.5 * (B2 - A2)), default))
        #write results
        write_results(predictions_test, labeledBow_test)
        #calculate accuracy
        acc_train = acc_calc(predictions_train, labeledBow_train)
        acc_test = acc_calc(predictions_test, labeledBow_test)
        #graph for accuracy
        acc_graph(acc_train, acc_test)
        #table for accuracy
        acc_table(acc_test, acc_train)
        #calculate precision and recall for different thresholds
        threshold = np.arange(start = A, stop = B, step = (B-A)/10)
        precisions = []
        recalls = []
        for i in threshold:
            predictions_test = []
            for j in prob_test:
                predictions_test.append(final_result(j, i, default))
            precisions.append(pre_calc(predictions_test, labeledBow_test))
            recalls.append(rec_calc(predictions_test, labeledBow_test))
        #create graph and table for precision-recall
        pre_rec_graph(precisions, recalls)
        pre_rec_table(precisions, recalls)
        #calculate F1 and create graph and table for F1
        F1 = F1_calc(predictions_test, labeledBow_test)
        F1_graph(F1)
        F1_table(F1)
    elif(alg == "Random Forest"):
        trees = Random_Forest_train(labeledBow_train, vocab, default)
        predictions_test = []
        predictions_train = []
        threshold = np.arange(start = 0.1, stop = 1.0, step = 0.1)
        prob_test_all = []
        for k in threshold: 
            prob_test = []
            for i in labeledBow_test :
                prob_test.append(Random_Forest_predict(i, trees, k))
                predictions_test.append(final_result(prob_test[-1], k, default))
            prob_test_all.append(prob_test)
        for i in labeledBow_train :
            predictions_train.append(final_result(Random_Forest_predict(i, trees, 0.5), 0.5, default))
        #write results
        write_results(predictions_test, labeledBow_test)
        #calculate accuracy
        acc_train = acc_calc(predictions_train, labeledBow_train)
        acc_test = acc_calc(predictions_test, labeledBow_test)
        #graph for accuracy
        acc_graph(acc_train, acc_test)
        #table for accuracy
        acc_table(acc_test, acc_train)
        #calculate precision and recall for different thresholds
        precisions = []
        recalls = []
        for i in range(0, len(threshold)):
            predictions_test = []
            for j in prob_test_all[i]:
                predictions_test.append(final_result(j, threshold[i], default))
            precisions.append(pre_calc(predictions_test, labeledBow_test))
            recalls.append(rec_calc(predictions_test, labeledBow_test))
        #create graph and table for precision-recall
        pre_rec_graph(precisions, recalls)
        pre_rec_table(precisions, recalls)
        #calculate F1 and create graph and table for F1
        F1=F1_calc(predictions_test, labeledBow_test)
        F1_graph(F1)
        F1_table(F1)
    else:
        flag = False
        print("invalid input")
        