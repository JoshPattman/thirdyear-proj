import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import logging
from colored_log_formatter import ColoredFormatter
from threading import Thread
import random
from queue import Queue
import sys, json, string

from flatten_model import flatten_model, unflatten_model
from fed import backend
from model import make_clone_model, get_xy

class FedClient:
    def __init__(self, epochs_per_step=1, num_samples=60000, dropout_after_first=False, classes=[0,1,2,3,4,5,6,7,8,9]):
        self.backend = backend.Client()
        self.backend.set_train_callback(self.train)

        self.epochs_per_step = epochs_per_step
        self.dropout_after_first = dropout_after_first
        self.training_count = 0

        logger = logging.getLogger("client%s"%random.randint(0, 1000))
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(ColoredFormatter())
        logger.addHandler(ch)
        self.logger = logger

        if dropout_after_first:
            self.logger.debug("dropout after first training step")

        (self.train_X, self.train_Y), _ = get_xy(num_train_samples=num_samples, classes=classes)

        self.logger.debug(np.unique(self.train_Y, return_counts=True))

    def train(self, model_flat):
        #self.logger.debug("starting training")
        if self.dropout_after_first and self.training_count > 0:
            self.logger.debug("dropping out so returing None")
            self.backend.training_complete(None)
            return
        model = make_clone_model()
        unflatten_model(model, model_flat)
        model.fit(self.train_X, self.train_Y, epochs=self.epochs_per_step, verbose=False)
        model_flat = flatten_model(model)
        self.backend.training_complete(model_flat)
        self.training_count += 1
        #self.logger.debug("done training")

class FedServer:
    def __init__(self):
        self.backend = backend.Server()
        self.global_model = make_clone_model()

        logger = logging.getLogger("server")
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(ColoredFormatter())
        logger.addHandler(ch)
        self.logger = logger

        _, (self.test_X, self.test_Y) = get_xy()
        self.update_loop()

    def update_loop(self):
        accuracies = [self.evaluate_performance()]
        epochs = [0]
        self.logger.info("accuracy init - %s"%(accuracies[0]))
        for i in range(20):
            trained_models = self.backend.perform_training(flatten_model(self.global_model))
            avg_model = np.mean(np.array(trained_models), axis=0)
            unflatten_model(self.global_model, avg_model)
            perf = self.evaluate_performance()
            self.logger.info("accuracy (loop %s) - %s"%(i+1, perf))
            epochs.append(i+1)
            accuracies.append(perf)
        self.data_log = (epochs, accuracies)

    def evaluate_performance(self):
        preds = self.global_model.predict(self.test_X, verbose=False)
        num_correct = 0
        for i in range(len(self.test_Y)):
            if np.argmax(preds[i]) == self.test_Y[i]:
                num_correct += 1
        accuracy = 100*num_correct/self.test_Y.shape[0]
        return accuracy
    
def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str
    
node_count = int(sys.argv[1])
num_samples = int(sys.argv[2])
epochs_per_step = int(sys.argv[3])
classes_per_node = int(sys.argv[4])
filename = sys.argv[5]

nodes_classes = []
c = 0
for i in range(10):
    nodes_classes.append([])
    for j in range(classes_per_node):
        nodes_classes[i].append(c)
        c = (c+1)%10

for i in range(node_count):
    FedClient(epochs_per_step=epochs_per_step, num_samples=num_samples, dropout_after_first=False, classes=nodes_classes[i])
    print("Classes for node %s: %s"%(i, nodes_classes[i]))

serv = FedServer()
results_epochs, results_accuracies = serv.data_log

fn = "./data/"+filename+".json"
with open(fn, "w") as f:
    f.write(json.dumps({
        "alpha":-1,
        "beta":-1,
        "gamma":-1,
        "num_samples":num_samples,
        "node_count":node_count,
        "density":-1,
        "startup_delay":-1,
        "forward_prob":-1,
        "epochs":[results_epochs],
        "accuracies":[results_accuracies],
        "msds": [[-1 for e in results_epochs]],
    }))
print("Saved results to %s"%fn)