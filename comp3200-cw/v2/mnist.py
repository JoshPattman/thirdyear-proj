from swarm_distributor import SwarmDistributor
from flask_backend import FlaskBackend
import time
import numpy as np
import logging
from colored_log_formatter import ColoredFormatter

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
    return model

flattener = ModelFlattener(make_model())

class Node:
    def __init__(self, port, neighbors):
        logger = logging.getLogger("node-%s"%port)
        ch = logging.StreamHandler()
        ch.setFormatter(ColoredFormatter())
        logger.addHandler(ch)

        backend = FlaskBackend(port, neighbors, logger = logger)
        self.dist = SwarmDistributor(np.array([1,2,3]), backend)
