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
from model import make_clone_model, get_xy, evaluate_performance

from keras.datasets import fashion_mnist as mnist

data_q = Queue()

class Node:
    def __init__(self, num_train_samples=60000, alpha=0.6, beta=99999999, gamma=8):
        self.model = make_clone_model()
        backend = LocalBackend()
        self.dist = SwarmDist(backend, -1, initial_params=flatten_model(self.model))

        logger = logging.getLogger("node-%s"%self.dist.node_id)
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(ColoredFormatter())
        logger.addHandler(ch)
        self.logger = logger

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        (self.train_X, self.train_Y), (self.test_X, self.test_Y) = get_xy(num_train_samples=num_train_samples)
        Thread(target=self.update_loop, daemon=True).start()
    
    def update_loop(self):
        unflatten_model(self.model, self.dist.get_state()[0])
        accuracies = [self.evaluate_performance()]
        epochs = [0]
        for i in range(20):
            self.model.fit(self.train_X, self.train_Y, epochs=1, verbose=False)
            self.dist.update_local_params(flatten_model(self.model))
            self.dist.sync(alpha=self.alpha, beta=self.beta, gamma=self.gamma, use_ASR=(not alpha==0))
            state = self.dist.get_state()
            unflatten_model(self.model, state[0])
            perf = self.evaluate_performance()#self.model)
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

node_count = int(sys.argv[1])
startup_delay = float(sys.argv[2])
num_samples = int(sys.argv[3])

alpha = float(sys.argv[4])
beta = float(sys.argv[5])
gamma = float(sys.argv[6])

exid = sys.argv[7]

print(f"Running experiment {exid} with {node_count} nodes, {num_samples} samples, alpha={alpha}, beta={beta}, gamma={gamma}, startup_delay={startup_delay}")

for i in range(node_count):
    Node(num_train_samples=num_samples, alpha=alpha, beta=beta, gamma=gamma)
    time.sleep(startup_delay)

results = []
for i in range(node_count):
    results.append(data_q.get())

fn = "./data/"+get_random_string(10)+".json"
with open(fn, "w") as f:
    f.write(json.dumps({
        "nodes_data":results,
        "exid": exid,
    }))