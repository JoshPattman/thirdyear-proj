from swarm_distributor import SwarmDistributor
from flask_backend import FlaskBackend
import time
import numpy as np
import logging
from colored_log_formatter import ColoredFormatter
from threading import Thread
import random
from datetime import datetime

import tensorflow as tf
from keras.layers import Dense, Input, Flatten, Conv2D, Reshape
from keras import Model
from keras.datasets import mnist
from keras.models import clone_model
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy

def tuple_product(t):
    x = 1
    for ta in t:
        x *= ta
    return x

class ModelFlattener:
    def __init__(self, example):
        params = example.get_weights()
        self.params_shapes = []
        for p in params:
            self.params_shapes.append(p.shape)
    def flatten(self, model):
        return np.concatenate([np.reshape(p, (-1,)) for p in model.get_weights()])
    def unflatten(self, model, flat_params):
        params = []
        n = 0
        for ps in self.params_shapes:
            num_params = tuple_product(ps)
            params.append(np.reshape(flat_params[n:n+num_params], ps))
            n += num_params
        model.set_weights(params)

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

flattener = ModelFlattener(make_model())

class Node:
    def __init__(self, port, neighbors, num_train_samples=60000):
        logger = logging.getLogger("node-%s"%port)
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(ColoredFormatter())
        logger.addHandler(ch)
        self.logger = logger

        (self.train_X, self.train_Y), (self.test_X, self.test_Y) = mnist.load_data()
        train_subset = random.sample(range(len(self.train_X)), num_train_samples)
        self.train_X = np.array([self.train_X[s] for s in train_subset])
        self.train_Y = np.array([self.train_Y[s] for s in train_subset])

        self.model = make_model()

        backend = FlaskBackend(port, neighbors, logger = logger)
        self.dist = SwarmDistributor(flattener.flatten(self.model), backend, diff_weight=0)
        flattener.unflatten(self.model, self.dist.get_training_params())

        Thread(target=self.update_loop).start()

    def update_loop(self):
        no_train = self.evaluate_performance("no training")
        self.logger.info("ACCURACY (no training): %s"%no_train)
        for loop in range(5):
            tstart = datetime.now()
            self.model.fit(self.train_X, self.train_Y, epochs=1, verbose=False)
            self.logger.debug("time for train: %s"%(datetime.now() - tstart).total_seconds())
            
            pre = self.evaluate_performance("pre avg step (%s)"%loop, color="\033[035m")

            tstart_sync = datetime.now()
            self.dist.update_params(flattener.flatten(self.model))
            self.dist.sync()
            self.logger.debug("time for sync: %s"%(datetime.now() - tstart_sync).total_seconds())

            flattener.unflatten(self.model, self.dist.get_training_params())

            post = self.evaluate_performance("post avg step (%s)"%loop, color="\033[034m")
            self.logger.info("ACCURACY (pre avg): %s, (post avg): %s"%(pre, post))
            self.logger.debug("time for total loop: %s"%(datetime.now() - tstart).total_seconds())

    def evaluate_performance(self, tag, color="\033[0m"):
        tstart = datetime.now()
        preds = self.model.predict(self.test_X, verbose=False)
        tdiff = round((datetime.now()-tstart).total_seconds(),2)
        num_correct = 0
        for i in range(len(self.test_Y)):
            if np.argmax(preds[i]) == self.test_Y[i]:
                num_correct += 1
        accuracy = 100*num_correct/len(self.test_Y)
        return accuracy
        print(f"Accuracy {color}(%s)\033[0m <%ss>: {color}%s\033[0m"%(tag,tdiff,accuracy))



ports = [9100, 9101, 9102, 9103, 9104]
for p in ports:
    Node(p, ["localhost:%s"%x for x in ports if x != p])
print("Started all nets")