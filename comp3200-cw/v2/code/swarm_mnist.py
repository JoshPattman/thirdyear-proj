from swarmlearn.swarm_distributor import SwarmDistributor
from swarmlearn.flask_backend import FlaskBackend
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

import tensorflow as tf
from keras.layers import Dense, Input, Flatten, Conv2D, Reshape
from keras import Model
from keras.datasets import fashion_mnist as mnist
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy


resultsQ = Queue()

class Node:
    def __init__(self, port, neighbors, gpu, num_train_samples=60000, global_start_time=datetime.now(), epochs_per_sync=1, sync_rate=0.5):
        self.gpu = gpu
        logger = logging.getLogger("node-%s"%port)
        logger.setLevel(logging.DEBUG)
        logger.debug("Node starting")
        ch = logging.StreamHandler()
        ch.setFormatter(ColoredFormatter())
        logger.addHandler(ch)
        self.logger = logger
        self.time_training = 0
        self.time_syncing = 0
        self.time_converting = 0
        self.global_start_time = global_start_time
        self.epochs_per_sync = epochs_per_sync

        (self.train_X, self.train_Y), (self.test_X, self.test_Y) = mnist.load_data()
        train_subset = random.sample(range(len(self.train_X)), num_train_samples)
        self.train_X = np.array([self.train_X[s] for s in train_subset])
        self.train_Y = np.array([self.train_Y[s] for s in train_subset])

        self.model = self.make_model()

        backend = FlaskBackend(port, neighbors, logger = logger)
        self.dist = SwarmDistributor(flatten_model(self.model), backend, neighbor_full_sync_weight=sync_rate)

        logger.debug("Node started")

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

    def start_updating(self):
        self.logger.debug("Node enabling")
        start_thread = Thread(target=self.update_loop)
        start_thread.setDaemon(True)
        start_thread.start()
        self.logger.debug("Node enabled")

    def update_loop(self):
        # Init
        self.dist.start()
        unflatten_model(self.model, self.dist.get_training_params())
        # Run updates
        times = []
        accuracies = []
        with tf.device(self.gpu):
            no_train = self.evaluate_performance()
        self.logger.info("ACCURACY (no training): %s"%no_train)
        time_offset = (datetime.now()-self.global_start_time).total_seconds()
        times.append(0)#time_offset)
        accuracies.append(no_train)
        time.sleep(5)
        training_start = datetime.now()
        for loop in range(20):
        #while (datetime.now()-training_start).total_seconds() < 250:
            temp_timer = datetime.now()
            with tf.device(self.gpu):
                self.model.fit(self.train_X, self.train_Y, epochs=self.epochs_per_sync, verbose=False)
            self.time_training += (datetime.now()-temp_timer).total_seconds()

            temp_timer = datetime.now()
            self.dist.update_params(flatten_model(self.model))
            self.time_converting += (datetime.now()-temp_timer).total_seconds()

            temp_timer = datetime.now()
            self.dist.sync()
            self.time_syncing += (datetime.now()-temp_timer).total_seconds()

            temp_timer = datetime.now()
            unflatten_model(self.model, self.dist.get_training_params())
            self.time_converting += (datetime.now()-temp_timer).total_seconds()

            with tf.device(self.gpu):
                post = self.evaluate_performance()
            times.append((loop+1)*self.epochs_per_sync)#(datetime.now()-training_start).total_seconds()+time_offset)
            accuracies.append(post)
            self.logger.info("ACCURACY (post avg): %.4s | T C S %.4ss %.4ss %.4ss"%(post, self.time_training, self.time_converting, self.time_syncing))

        resultsQ.put((times, accuracies))

    def evaluate_performance(self):
        preds = self.model.predict(self.test_X, verbose=False)
        num_correct = 0
        for i in range(len(self.test_Y)):
            if np.argmax(preds[i]) == self.test_Y[i]:
                num_correct += 1
        accuracy = 100*num_correct/len(self.test_Y)
        return accuracy

gpus = [x.name for x in tf.config.list_logical_devices('GPU')]
g = 0
print(gpus)
def next_gpu():
    global g
    gpu = gpus[g]
    g = (1+g)%len(gpus)
    return gpu

# python mnist.py <port:9000> <nodes:5> <num_train_samples:60000> <uid:10> <epochs:1> <sync_rate:0.5>
arg_port, arg_nodes, arg_training_samples, arg_uid, arg_epochs, arg_sync_rate = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), float(sys.argv[6])

ports = list(range(arg_port, arg_port+arg_nodes))
print("Running on ports %s"%ports)
print("Running with %s nodes, each with %s training samples, and %s epochs per step with sync rate of %s"%(arg_nodes, arg_training_samples, arg_epochs, arg_sync_rate))

nodes = []
start_time = datetime.now()
for p in ports:
    nodes.append(Node(p, ["localhost:%s"%x for x in ports if x != p], next_gpu(), num_train_samples=arg_training_samples, global_start_time=start_time, epochs_per_sync=arg_epochs, sync_rate=arg_sync_rate))
print("Starting mother node")
nodes[0].start_updating()
time.sleep(5)
print("Starting child nodes")
for n in range(1, len(nodes)):
    nodes[n].start_updating()
print("Started all nodes, waiting for results")

nodes_results = []
for p in ports:
    nodes_results.append(resultsQ.get())

print("Finished training")
filename = "../data/swarm_accuracy_data/nodes:%s_samples:%s_uid:%s_epochs:%s_sync:%s.json"%(arg_nodes,arg_training_samples,arg_uid, arg_epochs, arg_sync_rate)
print("Saving data log to %s"%filename)
with open(filename, "w") as f:
    f.write(json.dumps(nodes_results))
sys.exit(0)