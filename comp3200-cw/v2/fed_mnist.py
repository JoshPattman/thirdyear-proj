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

class FedClient:
    def __init__(self, num_train_samples, port, epochs_per_sync=1):
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

        self.client = Client(train_fn, port)
        self.client.start()

class FedServer:
    def __init__(self, ip, port, nodes_addrs):
        self.global_model = make_model()
        (self.train_X, self.train_Y), (self.test_X, self.test_Y) = mnist.load_data()
        self.server = Server(ip, port, nodes_addrs)
        self.server.start()
        #Thread(target=self.update_loop, daemon=True).start()
        self.update_loop()
    
    def update_loop(self):
        print("Update loop started")
        training_start = datetime.now()
        while (datetime.now()-training_start).total_seconds() < 250:
            print("Init update loop iteration")
            params = self.server.train(flatten_model(self.global_model))
            print("Finish update loop iteration")
            unflatten_model(self.global_model, params)
            print(self.evaluate_performance())

    def evaluate_performance(self):
        preds = self.global_model.predict(self.test_X, verbose=False)
        num_correct = 0
        for i in range(len(self.test_Y)):
            if np.argmax(preds[i]) == self.test_Y[i]:
                num_correct += 1
        accuracy = 100*num_correct/len(self.test_Y)
        return accuracy

ports = [9011, 9012, 9013, 9014, 9015]

print("Starting clients")
for p in ports:
    FedClient(60000, p)
print("Clients started")
time.sleep(3)

print("Starting server")
FedServer("localhost", 9010, ["http://localhost:%s"%p for p in ports])
print("Server started")
