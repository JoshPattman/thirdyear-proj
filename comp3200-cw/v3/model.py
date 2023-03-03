import tensorflow as tf
from keras.layers import Dense, Input, Flatten, Conv2D, Reshape
from keras import Model
from keras.models import clone_model
from keras.datasets import fashion_mnist as mnist
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
import keras

import random
import numpy as np

def make_model():
    inp = Input((28,28))
    out = Reshape((28,28,1))(inp)
    out = Conv2D(16, (3,3), activation="relu")(out)
    out = Conv2D(16, (3,3), activation="relu")(out)
    out = Flatten()(out)
    out = Dense(128, activation="relu")(out)
    out = Dense(10, activation="sigmoid")(out)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss=SparseCategoricalCrossentropy(), metrics=[SparseCategoricalAccuracy()])
    return model

global_start_model = make_model()

def make_clone_model():
    m = clone_model(global_start_model)
    m.compile(optimizer="adam", loss=SparseCategoricalCrossentropy(), metrics=[SparseCategoricalAccuracy()])
    return m

(full_train_X, full_train_Y), (full_test_X, full_test_Y) = mnist.load_data()

def get_xy(num_train_samples=60000):
    train_subset = random.sample(range(len(full_train_X)), num_train_samples)
    train_X = np.array([full_train_X[s] for s in train_subset])/255
    train_Y = np.array([full_train_Y[s] for s in train_subset])
    test_X = np.copy(full_test_X)/255
    return (train_X, train_Y), (test_X, np.copy(full_test_Y))

def evaluate_performance(model):
    preds = model.predict(full_test_X, verbose=False)
    num_correct = 0
    for i in range(len(full_test_Y)):
        if np.argmax(preds[i]) == full_test_Y[i]:
            num_correct += 1
    accuracy = 100*num_correct/full_test_Y.shape[0]
    return accuracy