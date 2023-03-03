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
    def __init__(self):
        self.backend = backend.Client()
        self.backend.set_train_callback(self.train)

        logger = logging.getLogger("client")
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(ColoredFormatter())
        logger.addHandler(ch)
        self.logger = logger

        (self.train_X, self.train_Y), _ = get_xy(num_train_samples=6000)

    def train(self, model_flat):
        #self.logger.debug("starting training")
        model = make_clone_model()
        unflatten_model(model, model_flat)
        model.fit(self.train_X, self.train_Y, epochs=1, verbose=False)
        model_flat = flatten_model(model)
        self.backend.training_complete(model_flat)
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
    
node_count = int(sys.argv[1])#10
exid = sys.argv[2]#"fed"

for i in range(10):
    FedClient()

serv = FedServer()
results = [serv.data_log]
fn = "./data/"+get_random_string(10)+".json"
with open(fn, "w") as f:
    f.write(json.dumps({
        "nodes_data":results,
        "exid":exid,
        "node_count":node_count,
        "sync_rate":0,
        "stagger":0,
    }))