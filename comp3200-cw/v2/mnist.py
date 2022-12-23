from swarm_distributor import SwarmDistributor
from flask_backend import FlaskBackend
import time
import numpy as np
import logging
from colored_log_formatter import ColoredFormatter
from threading import Thread
import random
from datetime import datetime
from queue import Queue

from flatten_model import flatten_model, unflatten_model

import tensorflow as tf
from keras.layers import Dense, Input, Flatten, Conv2D, Reshape
from keras import Model
from keras.datasets import mnist
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy

def batches(list, chunk_size):
  for i in range(0, len(list), chunk_size):
    yield list[i:i + chunk_size]

resultsQ = Queue()

class Node:
    def __init__(self, port, neighbors, num_train_samples=60000, global_start_time=datetime.now(), checkpoint_every=-1, batch_size=32):
        logger = logging.getLogger("node-%s"%port)
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(ColoredFormatter())
        logger.addHandler(ch)
        self.logger = logger
        self.global_start_time = global_start_time
        self.checkpoint_every = checkpoint_every
        self.batch_size = batch_size

        (self.train_X, self.train_Y), (self.test_X, self.test_Y) = mnist.load_data()
        train_subset = random.sample(range(len(self.train_X)), num_train_samples)
        self.train_X = np.array([self.train_X[s] for s in train_subset])
        self.train_Y = np.array([self.train_Y[s] for s in train_subset])

        self.model = self.make_model()

        backend = FlaskBackend(port, neighbors, logger = logger)
        self.dist = SwarmDistributor(flatten_model(self.model), backend)
        unflatten_model(self.model, self.dist.get_training_params())

        Thread(target=self.update_loop).start()

    def make_model(self):
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

    def update_loop(self):
        times = []
        accuracies = []
        no_train = self.evaluate_performance()
        self.logger.info("ACCURACY (no training): %s"%no_train)
        time_offset = (datetime.now()-self.global_start_time).total_seconds()
        times.append(time_offset)
        accuracies.append(no_train)
        training_start = datetime.now()
        for loop in range(5):
            if self.checkpoint_every == -1:
                self.model.fit(self.train_X, self.train_Y, epochs=1, verbose=False, batch_size=self.batch_size)
                self.dist.update_params(flatten_model(self.model))
            else:
                samples = list(range(len(self.train_X)))
                random.shuffle(samples)
                for batchIndexes in batches(samples, self.checkpoint_every*self.batch_size):
                    xs = np.array([self.train_X[i] for i in batchIndexes])
                    ys = np.array([self.train_Y[i] for i in batchIndexes])
                    self.model.fit(xs, ys, epochs=1, verbose=False, batch_size=self.batch_size)
                    self.dist.update_params(flatten_model(self.model))

            self.dist.sync()
            unflatten_model(self.model, self.dist.get_training_params())

            post = self.evaluate_performance()
            times.append((datetime.now()-training_start).total_seconds()+time_offset)
            accuracies.append(post)
            self.logger.info("ACCURACY (post avg): %.4s"%(post))

        resultsQ.put((times, accuracies))

    def evaluate_performance(self):
        preds = self.model.predict(self.test_X, verbose=False)
        num_correct = 0
        for i in range(len(self.test_Y)):
            if np.argmax(preds[i]) == self.test_Y[i]:
                num_correct += 1
        accuracy = 100*num_correct/len(self.test_Y)
        return accuracy



ports = [9100, 9101, 9102, 9103, 9104]
start_time = datetime.now()
for p in ports:
    Node(p, ["localhost:%s"%x for x in ports if x != p], num_train_samples=500, checkpoint_every=1, batch_size=32, global_start_time=start_time)
print("Started all nets, waiting for results")

nodes_results = []
for p in ports:
    nodes_results.append(resultsQ.get())

print("Finished training:")
print(nodes_results)