#!/usr/env/bin python3.6

import os
import sys
import pickle
from collections import defaultdict
import argparse

import numpy as np
import pandas as pd


def readData(fpath):
    with open(fpath, 'rb') as fr:
        inf = pickle.load(fr, encoding='iso-8859-1')
    return inf


def loadData(dataRoot, trace_length):
    if isinstance(dataRoot, bytes):
        dataRoot = dataRoot.decode()

    trainDatapath = os.path.join(dataRoot, 'X_train_NoDef.pkl')
    trainLabelpath = os.path.join(dataRoot, 'y_train_NoDef.pkl')
    #valDatapath = os.path.join(dataRoot, 'X_valid_NoDef.pkl')
    #valLabelpath = os.path.join(dataRoot, 'y_valid_NoDef.pkl')
    testDatapath = os.path.join(dataRoot, 'X_test_NoDef.pkl')
    testLabelpath = os.path.join(dataRoot, 'y_test_NoDef.pkl')


    X_train = readData(trainDatapath)
    y_train = readData(trainLabelpath)
    X_train = np.array(X_train)
    y_train = np.array(y_train, dtype=np.uint8)
    X_train = X_train[: , 0:trace_length]


    X_test = readData(testDatapath)
    y_test = readData(testLabelpath)
    X_test = np.array(X_test)
    y_test = np.array(y_test, dtype=np.uint8)
    X_test = X_test[: , 0:trace_length]

    """
    X_val = readData(valDatapath)
    y_val = readData(valLabelpath)
    X_val = np.array(X_val)
    y_val = np.array(y_val, dtype=np.uint8)
    """

    return X_train, y_train, X_test, y_test


def mergeDataLabel(data, label):
    assert(data.shape[0] == len(label))
    rtn = []
    for i in range(len(label)):
        sample = list(data[i,:])
        y = int(label[i])
        sample.append(y)
        rtn.append(sample)

    return np.array(rtn)


def main(opts):
    trace_length = 2000
    X_train, y_train, X_test, y_test = loadData(opts.droot, trace_length)

    trainData = mergeDataLabel(X_train, y_train)
    testData = mergeDataLabel(X_test, y_test)

    trainData = pd.DataFrame(trainData)
    testData = pd.DataFrame(testData)

    trainDataFile = os.path.join(opts.output, 'traffic_trace_train.csv')
    testDataFile = os.path.join(opts.output, 'traffic_trace_test.csv')

    trainData.to_csv(trainDataFile, sep='\t')
    testData.to_csv(testDataFile, sep='\t')


def parseOpts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--droot', help="")
    parser.add_argument('-o', '--output', help="")
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseOpts(sys.argv)
    main(opts)
