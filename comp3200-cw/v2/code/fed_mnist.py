from fedlearn.client import Client
from fedlearn.server import Server
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
from keras.datasets import mnist
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

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

def make_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(ColoredFormatter())
    logger.addHandler(ch)
    return logger

class FedClient:
    def __init__(self, num_train_samples, port, epochs_per_sync=1):
        self.logger = make_logger("node-%s"%port)

        (self.train_X, self.train_Y), (self.test_X, self.test_Y) = mnist.load_data()
        train_subset = random.sample(range(len(self.train_X)), num_train_samples)
        self.train_X = np.array([self.train_X[s] for s in train_subset])
        self.train_Y = np.array([self.train_Y[s] for s in train_subset])

        self.epochs_per_sync = epochs_per_sync

        def train_fn(params):
            # Create a new model and load params
            model = make_model()
            unflatten_model(model, params)
            # fit the model a bit
            model.fit(self.train_X, self.train_Y, epochs=self.epochs_per_sync, verbose=False)
            # return the flattened model
            return flatten_model(model)

        self.client = Client(train_fn, port, self.logger)
        self.client.start()

class FedServer:
    def __init__(self, ip, port, nodes_addrs):
        self.logger = make_logger("node-%s"%port)

        self.global_model = make_model()
        (self.train_X, self.train_Y), (self.test_X, self.test_Y) = mnist.load_data()
        self.server = Server(ip, port, nodes_addrs, self.logger)
        self.server.start()
    
    def update_loop(self):
        times = []
        accuracies = []
        times.append(0)
        accuracies.append(self.evaluate_performance())
        self.logger.info("Update loop started")
        training_start = datetime.now()
        while (datetime.now()-training_start).total_seconds() < 250:
            self.logger.debug("Init update loop iteration")
            params = self.server.train(flatten_model(self.global_model))
            self.logger.debug("Finish update loop iteration")
            unflatten_model(self.global_model, params)
            accuracy = self.evaluate_performance()
            self.logger.info("ACCURACY: %.4s"%accuracy)
            times.append((datetime.now()-training_start).total_seconds())
            accuracies.append(accuracy)
        return times, accuracies

    def evaluate_performance(self):
        preds = self.global_model.predict(self.test_X, verbose=False)
        num_correct = 0
        for i in range(len(self.test_Y)):
            if np.argmax(preds[i]) == self.test_Y[i]:
                num_correct += 1
        accuracy = 100*num_correct/len(self.test_Y)
        return accuracy

# python mnist.py <port:9000> <nodes:5> <num_train_samples:60000> <uid:10> <epochs:1>
arg_port, arg_nodes, arg_training_samples, arg_uid, arg_epochs = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5])

ports = list(range(arg_port+1, arg_port+arg_nodes+1))

print("Starting clients on ports %s"%ports)
for p in ports:
    FedClient(arg_training_samples, p, epochs_per_sync=arg_epochs)
print("Clients started")
time.sleep(3)

print("Starting server")
srv = FedServer("localhost", arg_port, ["http://localhost:%s"%p for p in ports])
print("Server started")
times, accuracies = srv.update_loop()

print("Finished training")
filename = "../data/fed_accuracy_data/nodes:%s_samples:%s_uid:%s_epochs:%s_sync:%s.json"%(arg_nodes,arg_training_samples,arg_uid, arg_epochs, 0)
print("Saving data log to %s"%filename)
with open(filename, "w") as f:
    f.write(json.dumps([(times, accuracies)]))
sys.exit(0)
