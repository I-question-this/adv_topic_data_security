#!/usr/bin/env python3.6
import os
import sys
import argparse

import numpy as np
import pandas as pd

from keras.utils import np_utils
sys.path.append('src')
import cnn


def sepDataLabel(dataLabel):
    data, label = [], []
    for i in range(dataLabel.shape[0]):
        sample = list(dataLabel[i, :])
        one_data = sample[1:-1]
        one_label = sample[-1]
        data.append(one_data)
        label.append(one_label)

    return data, label

def readCSVFile(fpath):
    allData = pd.read_csv(fpath, sep='\t')
    allData = allData.to_numpy()
    data, label = sepDataLabel(allData)
    data = np.array(data)
    return data, label

def loadData(opts, PARAMS):
    X_train_raw, y_train = readCSVFile(opts.train)
    X_test_raw, y_test = readCSVFile(opts.test)

    NUM_CLASS = len(set(y_test))
    X_train = X_train_raw.reshape(X_train_raw.shape[0], X_train_raw.shape[1], 1)
    y_train = np_utils.to_categorical(y_train, NUM_CLASS)
    X_test = X_test_raw.reshape(X_test_raw.shape[0], X_test_raw.shape[1], 1)
    y_test = np_utils.to_categorical(y_test, NUM_CLASS)

    return X_train, y_train, X_test, y_test, NUM_CLASS

def loadTestData(opts, params):
    X_test_raw, y_test = readCSVFile(opts.test)

    NUM_CLASS = len(set(y_test))
    X_test = X_test_raw.reshape(X_test.shape[0], X_test.shape[1], 1)
    y_test = np_utils.to_categorical(y_test, NUM_CLASS)

    return X_test, y_test, NUM_CLASS

def main(opts):
    PARAMS = cnn.generate_default_params()
    model = cnn.CNN(opts, PARAMS)

    if not opts.testOnly:
        X_train, y_train, X_test, y_test, NUM_CLASS = loadData(opts, PARAMS)
        modelPath = model.train(X_train, y_train, NUM_CLASS)
    else:
        modelPath = opts.testOnly
        X_test, y_test, NUM_CLASS = loadTestData(opts, PARAMS)
        acc = model.test(X_test, y_test, NUM_CLASS, modelPath)


def parseOpts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-t1', '--train',
                        help ='file path of train data file')
    parser.add_argument('-t2', '--test',
                        help ='file path of test data file')
    parser.add_argument('-o', '--output', default='',
                        help ='output file name')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help ='verbose or not')
    parser.add_argument('-t', '--testOnly', default='',
                        help ='only run test with given model')
    parser.add_argument('-p', '--plotModel', action='store_true',
                        help ='verbose or not')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseOpts(sys.argv)
    main(opts)
