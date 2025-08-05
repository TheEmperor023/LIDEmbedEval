import numpy as np


def load_from_file(s):
    file = open(s)
    X=[]
    y=[]
    for line in file:
        line = line[:-1]
        tokens = line.split(':')
        # print(tokens)
        emb1 = tokens[2]
        emb2 = tokens[3]
        con = tokens[4]
        # print(emb1)
        values1 = [float(x) for x in emb1.split(',')]
        values2 = [float(x) for x in emb2.split(',')]
        X.append(values1 + values2)
        y.append(con)

    X = np.array(X)
    y = np.array(y)

    return X, y


def load_from_file_diff(s):
    file = open(s)
    X = []
    y = []
    for line in file:
        line = line[:-1]
        tokens = line.split(':')
        # print(tokens)
        emb1 = tokens[2]
        emb2 = tokens[3]
        con = tokens[4]
        # print(emb1)
        values1 = [float(x) for x in emb1.split(',')]
        values2 = [float(x) for x in emb2.split(',')]
        em1 = np.array(values1)
        em2 = np.array(values2)
        dif = np.abs(em1 - em2)
        X.append(dif)
        y.append(con)

    y = np.array(y)

    return X, y