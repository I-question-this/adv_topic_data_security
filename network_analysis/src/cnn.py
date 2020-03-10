#!/usr/env/bin python3.6
#
# Copyright@Chenggang Wang
# 1277223029@qq.com
# All right reserved
#
# May 27, 2019

import os
import sys
import argparse
import logging

if '1' == os.getenv('useGpu'):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
from keras.layers.convolutional import Conv1D
from keras.layers import Dense, Dropout, Flatten, MaxPooling1D, BatchNormalization, GlobalAveragePooling1D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard
import keras.backend as K
from keras.utils import np_utils

import numpy as np
import pandas as pd


modelDir = 'modelDir'
if not os.path.isdir(modelDir):
    os.makedirs(modelDir)
LOG = logging.getLogger('modelDir/cnn_results')


def generate_default_params():
    return {
            'optimizer': 'Adam',
            'learning_rate': 0.01,
            'activation1': 'softsign',
            'activation2': 'softsign',
            'activation3': 'selu',
            'activation4': 'selu',
            'drop_rate1': 0.3,
            'drop_rate2': 0.1,
            'drop_rate3': 0.3,
            'drop_rate4': 0.5,
            'decay': 0.1,
            'batch_size': 70,
            'data_dim': 2000,
            'epochs': 500,
            'conv1': 64,
            'conv2': 128,
            'conv3': 256,
            'conv4': 128,
            'pool1': 5,
            'pool2': 3,
            'pool3': 1,
            'pool4': 3,
            'kernel_size1': 15,
            'kernel_size2': 21,
            'kernel_size3': 15,
            'kernel_size4': 11,
            'dense1': 150,
            'dense2': 130,
            'dense1_act': 'selu',
            'dense2_act': 'softsign'
            }


class CNN():
    def __init__(self, opts, params, name='cnn'):
        self.params = params
        self.verbose = opts.verbose
        self.plotModel = opts.plotModel
        self.name = name


    def create_model(self, NUM_CLASS):
        print ('Creating model...')

        layers = [Conv1D(self.params['conv1'], kernel_size=self.params['kernel_size1'], activation=self.params['activation1'], input_shape=(self.params['data_dim'], 1), use_bias=False, kernel_initializer='glorot_normal'),
                  BatchNormalization(),
                  MaxPooling1D(self.params['pool1']),
                  Dropout(rate=self.params['drop_rate1']),

                  Conv1D(self.params['conv2'], kernel_size=self.params['kernel_size2'], activation=self.params['activation2'], kernel_initializer='glorot_normal'),
                  BatchNormalization(),
                  MaxPooling1D(self.params['pool2']),
                  Dropout(rate=self.params['drop_rate2']),

                  Conv1D(self.params['conv3'], kernel_size=self.params['kernel_size3'], activation=self.params['activation3'], kernel_initializer='glorot_normal'),
                  BatchNormalization(),
                  MaxPooling1D(self.params['pool3']),
                  Dropout(rate=self.params['drop_rate3']),

                  Conv1D(self.params['conv4'], kernel_size=self.params['kernel_size4'], activation=self.params['activation4'], kernel_initializer='glorot_normal'),
                  BatchNormalization(),
                  MaxPooling1D(self.params['pool4']),
                  GlobalAveragePooling1D(),


                  Dense(self.params['dense1'], activation=self.params['dense1_act'], kernel_initializer='glorot_normal'),
                  BatchNormalization(),
                  Dense(self.params['dense2'], activation=self.params['dense2_act'], kernel_initializer='glorot_normal'),
                  BatchNormalization(),
                  Dense(NUM_CLASS, activation='softmax')]

        model = Sequential(layers)

        print ('Compiling...')
        model.compile(loss='categorical_crossentropy',
                      optimizer=self.params['optimizer'],
                      metrics=['accuracy'])
        return model


    def train(self, X_train, y_train, NUM_CLASS):
        '''train the cnn model'''
        model = self.create_model(NUM_CLASS)
        if self.plotModel:
            picDir = os.path.join(modelDir, 'pic')
            if not os.path.isdir(picDir):
                os.makedirs(picDir)
            picPath = os.path.join(picDir, 'cnn_model.png')
            from keras.utils import plot_model
            plot_model(model, to_file=picPath, show_shapes='True')

        print ('Fitting model...')

        def lr_scheduler(epoch):
            if epoch % 20 == 0 and epoch != 0:
                lr = K.get_value(model.optimizer.lr)
                K.set_value(model.optimizer.lr, lr*self.params['decay'])
                print("lr changed to {}".format(lr*self.params['decay']))
            return K.get_value(model.optimizer.lr)

        modelPath = os.path.join(modelDir, 'cnn_weights_best.hdf5')
        checkpointer = ModelCheckpoint(filepath=modelPath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        CallBacks = [checkpointer]
        if self.params['optimizer'] == 'SGD':
            scheduler = LearningRateScheduler(lr_scheduler)
            CallBacks.append(scheduler)
        CallBacks.append(EarlyStopping(monitor='val_acc', mode='max', patience=6))
        #CallBacks.append(TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True))

        hist = model.fit(X_train, y_train,
                         batch_size=self.params['batch_size'],
                         epochs=self.params['epochs'],
                         validation_split = 0.2,
                         verbose=self.verbose,
                         callbacks=CallBacks)

        if self.plotModel:
            from keras.utils import plot_model
            plot_model(model, to_file='model.png', show_shapes='True')

        return modelPath

    def prediction(self, X_test, NUM_CLASS, modelPath):
        print ('Predicting results with best model...')
        model = self.create_model(NUM_CLASS)
        model.load_weights(modelPath)
        y_pred = model.predict(X_test)
        return y_pred

    def test(self, X_test, y_test, NUM_CLASS, modelPath):
        print ('Testing with best model...')
        model = self.create_model(NUM_CLASS)
        model.load_weights(modelPath)
        score, acc = model.evaluate(X_test, y_test, batch_size=100)

        print('Test score:', score)
        print('Test accuracy:', acc)
        return acc
