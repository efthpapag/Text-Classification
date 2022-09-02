import random
from ID3 import *


def Random_Forest_train(examples, attributes, default) :#creates forest

    trees = []
    ex_per_tree = 1000
    Ntrees = len(examples) // ex_per_tree

    if (len(examples) % ex_per_tree) != 0 :
        Ntrees += 1

    IG = IG_calc(examples, len(attributes))

    for i in range(0, Ntrees):
        l = IG

        for k in range(0, len(IG) // 2) :
            j = random.randrange(0, len(l))
            l.pop(int(j))
        trees.append(ID3_train(random.choices(examples, None, k = ex_per_tree), default, l, 0))

    return trees


def Random_Forest_predict (example, trees, threshold) :#returns positive probability

    pos = 0
    
    for t in trees :
        if ID3_predict(example, t) >= threshold :
            pos += 1

    return pos / len(trees)