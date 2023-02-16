import numpy as np
import logging
from colored_log_formatter import ColoredFormatter
from threading import Thread
import random
from queue import Queue
import sys, json, time

from flatten_model import flatten_model, unflatten_model
from swarm.swarm_dist import SwarmDist, get_random_string
from swarm.local_backend import LocalBackend
from model import make_clone_model, get_xy, make_model

from keras.datasets import fashion_mnist as mnist

data_q = Queue()

class Node:
    def __init__(self,num_train_samples=60000, sync_rate=0.6):
        self.model = make_model()#make_clone_model()
        backend = LocalBackend()
        self.dist = SwarmDist(backend, -1, initial_params=flatten_model(self.model))

        logger = logging.getLogger("node-%s"%self.dist.node_id)
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(ColoredFormatter())
        logger.addHandler(ch)
        self.logger = logger

        self.sync_rate = sync_rate

        (self.train_X, self.train_Y), (self.test_X, self.test_Y) = get_xy(num_train_samples=6000)
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
    time.sleep(2)

results = []
for i in range(node_count):
    results.append(data_q.get())

fn = "./data/"+get_random_string(10)+".json"
with open(fn, "w") as f:
    f.write(json.dumps({
        "nodes_data":results,
        "exid":"swarm_ic_stag",
        "node_count":node_count,
        "sync_rate":sync_rate,
    }))