import time
import numpy as np
import logging
from colored_log_formatter import ColoredFormatter
from threading import Thread
import random
from datetime import datetime
from queue import Queue
import sys, json

from flatten_model import flatten_model, unflatten_model
from swarm_dist import SwarmDist, get_random_string
from local_backend import LocalBackend

import tensorflow as tf
from keras.layers import Dense, Input, Flatten, Conv2D, Reshape
from keras import Model
from keras.models import clone_model
from keras.datasets import fashion_mnist as mnist
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
import keras

def make_model():
    inp = Input((28,28))
    out = Reshape((28,28,1))(inp)
    out = Conv2D(16, (3,3), activation="relu")(out)
    out = Conv2D(16, (3,3), activation="relu")(out)
    out = Flatten()(out)
    out = Dense(128, activation="relu")(out)
    out = Dense(10, activation="sigmoid")(out)
    model = Model(inputs=inp, outputs=out)
    return model

global_start_model = make_model()

def make_clone_model():
    m = clone_model(global_start_model)
    m.compile(optimizer="adam", loss=SparseCategoricalCrossentropy(), metrics=[SparseCategoricalAccuracy()])
    return m

data_q = Queue()

class Node:
    def __init__(self,num_train_samples=60000, sync_rate=0.6):
        self.model = make_clone_model()
        backend = LocalBackend()
        self.dist = SwarmDist(backend, -1, initial_params=flatten_model(self.model))

        logger = logging.getLogger("node-%s"%self.dist.node_id)
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(ColoredFormatter())
        logger.addHandler(ch)
        self.logger = logger

        self.sync_rate = sync_rate

        (self.train_X, self.train_Y), (self.test_X, self.test_Y) = mnist.load_data()
        train_subset = random.sample(range(len(self.train_X)), num_train_samples)
        self.train_X = np.array([self.train_X[s] for s in train_subset])/255
        self.train_Y = np.array([self.train_Y[s] for s in train_subset])
        self.test_X = self.test_X/255

        Thread(target=self.update_loop, daemon=True).start()
    
    def update_loop(self):
        unflatten_model(self.model, self.dist.get_state()[0])
        accuracies = [self.evaluate_performance()]
        epochs = [0]
        for i in range(20):
            self.model.fit(self.train_X, self.train_Y, epochs=1, verbose=False)
            self.dist.update_local_params(flatten_model(self.model))
            self.dist.sync(sync_rate=self.sync_rate)
            state = self.dist.get_state()
            unflatten_model(self.model, state[0])
            perf = self.evaluate_performance()
            accuracies.append(perf)
            epochs.append(i+1)
            self.logger.info("accuracy (loop %s, tc %s) - %s"%(i+1, state[1], perf))
        data_q.put((epochs, accuracies))

    def evaluate_performance(self):
        preds = self.model.predict(self.test_X, verbose=False)
        num_correct = 0
        for i in range(len(self.test_Y)):
            if np.argmax(preds[i]) == self.test_Y[i]:
                num_correct += 1
        accuracy = 100*num_correct/self.test_Y.shape[0]
        return accuracy

node_count = 10
sync_rate = 0.6

for i in range(node_count):
    Node(num_train_samples=int(60000/node_count),sync_rate=sync_rate)

results = []
for i in range(node_count):
    results.append(data_q.get())

fn = "./data/"+get_random_string(10)+".json"
with open(fn, "w") as f:
    f.write(json.dumps({
        "nodes_data":results,
        "exid":"swarm_ic",
        "node_count":node_count,
        "sync_rate":sync_rate,
    }))