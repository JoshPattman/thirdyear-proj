import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import logging
from colored_log_formatter import ColoredFormatter
from threading import Thread
import random
from queue import Queue
import sys, json, time

from flatten_model import flatten_model, unflatten_model
from swarm.swarm_dist import SwarmDist
from swarm.local_backend import LocalBackend, get_random_string
from model import make_clone_model, get_xy, evaluate_performance, make_model

from keras.datasets import fashion_mnist as mnist
import tensorflow as tf

import graphs

data_q = Queue()

class Node:
    def __init__(self, num_train_samples=60000, alpha=0.6, beta=99999999, gamma=8, node_id=None, neighbors=[], epochs_per_step=1):
        self.model = make_clone_model()#make_model()#
        backend = LocalBackend(node_id=node_id, neighbors=neighbors)
        self.dist = SwarmDist(backend, -1, initial_params=flatten_model(self.model))

        self.epochs_per_step = epochs_per_step

        logger = logging.getLogger("ND[%s]"%self.dist.backend.node_id)
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(ColoredFormatter())
        logger.addHandler(ch)
        self.logger = logger

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        (self.train_X, self.train_Y), (self.test_X, self.test_Y) = get_xy(num_train_samples=num_train_samples)

    def begin(self):
        Thread(target=self.update_loop, daemon=True).start()
    
    def update_loop(self):
        unflatten_model(self.model, self.dist.get_state()[0])
        accuracies = [self.evaluate_performance()]
        self.logger.info("accuracy (pre training) - %s"%accuracies[0])
        msds = [0]
        epochs = [0]
        for i in range(20):
            self.model.fit(self.train_X, self.train_Y, epochs=self.epochs_per_step, verbose=False)
            self.dist.update_local_params(flatten_model(self.model))
            msd = self.dist.sync(alpha=self.alpha, beta=self.beta, gamma=self.gamma, use_ASR=(not alpha==0))
            state = self.dist.get_state()
            unflatten_model(self.model, state[0])
            perf = self.evaluate_performance()
            accuracies.append(perf)
            epochs.append(i+1)
            msds.append(msd)
            self.logger.info("accuracy (loop %s, tc %s) - %s, msd %s"%(i+1, state[1], perf, msd))
            #unflatten_model(self.model, state[0])
        data_q.put((epochs, accuracies, msds))

    def evaluate_performance(self):
        try:
            with tf.device('/cpu:0'):
                preds = self.model.predict(self.test_X, verbose=False)
        except Exception as e:
            print("ERRRRRRRRRRRRRRRRRRRROOOOOOOOOOOOOOOOOOOOOOOOOORRRRRRRRRRRRRRRRRRR Error in prediction\n")
            raise e
        num_correct = 0
        for i in range(len(self.test_Y)):
            if np.argmax(preds[i]) == self.test_Y[i]:
                num_correct += 1
        accuracy = 100.0*float(num_correct)/float(self.test_Y.shape[0])
        return accuracy
print(sys.argv)

node_count = int(sys.argv[1])
startup_delay = float(sys.argv[2])
num_samples = int(sys.argv[3])

alpha = float(sys.argv[4])
beta = float(sys.argv[5])
gamma = float(sys.argv[6])

density = float(sys.argv[7])

epochs_per_step = int(sys.argv[8])

filename = sys.argv[9]

print(f"Running experiment with {node_count} nodes, {num_samples} samples, alpha={alpha}, beta={beta}, gamma={gamma}, startup_delay={startup_delay}, density={density}, epochs_per_step={epochs_per_step}")

nodes = [f"node-%s"%x for x in range(node_count)]
print("All nodes are: ", nodes)
connections = graphs.fully_connected_graph(nodes, density=density)

print("Initialising nodes...")
node_objects = []
for n in nodes:
    # Find neighbors
    neighbors = []
    for c in connections:
        if c[0] == n:
            neighbors.append(c[1])
        if c[1] == n:
            neighbors.append(c[0])
    print("Node %s has neighbors: %s"%(n, neighbors))
    node_objects.append(Node(num_train_samples=num_samples, alpha=alpha, beta=beta, gamma=gamma, node_id=n, neighbors=neighbors, epochs_per_step=epochs_per_step))

print("Starting nodes...")
for n in node_objects:
    n.begin()
    print("Started node %s"%n.dist.backend.node_id)
    time.sleep(startup_delay)

print("Waiting for nodes to finish...")
results_epochs = []
results_accuracies = []
results_msds = []
for i in range(node_count):
    e, a, m = data_q.get()
    results_epochs.append(e)
    results_accuracies.append(a)
    results_msds.append(m)

fn = "./data/"+filename+".json"
with open(fn, "w") as f:
    f.write(json.dumps({
        "alpha":alpha,
        "beta":beta,
        "gamma":gamma,
        "num_samples":num_samples,
        "node_count":node_count,
        "density":density,
        "startup_delay":startup_delay,
        "forward_prob":0,
        "epochs":results_epochs,
        "accuracies":results_accuracies,
        "msds": results_msds,
    }))
print("Saved results to %s"%fn)