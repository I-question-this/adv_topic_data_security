import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.append('src')
import cnn


def readCSVFile(fpath):
    allData = pd.read_csv(fpath, sep='\t')
    data = allData['data'].values
    label = allData['label'].values
    return data, label

def loadData(opts, PARAMS):
    X_train, y_train = readCSVFile(opts.train)
    X_test, y_test = readCSVFile(opts.test)

    NUM_CLASS = len(set(y_test))
    X_train = X_train_raw.reshape(X_train_raw.shape[0], X_train_raw.shape[1], 1)
    y_train = np_utils.to_categorical(y_train, NUM_CLASS)
    X_test = X_test_raw.reshape(X_test_raw.shape[0], X_test_raw.shape[1], 1)
    y_test = np_utils.to_categorical(y_test, NUM_CLASS)

    return X_train, y_train, X_test, y_test, NUM_CLASS

def loadTestData(opts, params):
    X_test, y_test = readCSVFile(opts.test)

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
    parser.add_argument('-i', '--input',
                        help ='file path of config file')
    parser.add_argument('-d', '--dataType',
                        help ='dataType to use, onlyOrder/both')
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
