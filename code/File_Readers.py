import os


def read_vocab(filename):#turns vocabulary file into list

    file1 = open(filename, encoding = "utf8")
    vocab = []

    while True : 
        line = file1.readline() 
        if not line: 
            break
        vocab.append(line)
    file1.close()

    return vocab


def read_labeledBow(filename):#turns labeledBow file into list

    file1 = open(filename, encoding = "utf8")
    labeledBow = []

    while True : 
        line = file1.readline() 
        if not line: 
            break
        line = line.split(" ")
        for i in range(1, len(line)) :
            line[i] = line[i].split(":")
            for  j in range(0, len(line[i]))  :
                line[i][j] = int(line[i][j])
        line[0] = int(line[0])
        labeledBow.append(line)
    file1.close()

    return labeledBow