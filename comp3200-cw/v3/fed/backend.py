from threading import Lock, Thread
from queue import Queue
import logging, random, string, time


clients = []

server = None

class Client:
    def __init__(self):
        clients.append(self)
        self.training_callback = None

    def set_train_callback(self, callback):
        self.training_callback = callback

    def training_complete(self, trained_model):
        server.trained_q.put(trained_model)

class Server:
    def __init__(self):
        global server
        server = self
        self.trained_q = Queue()

    def perform_training(self, model):
        for c in clients:
            Thread(target=c.training_callback, args=(model,), daemon=True).start()
        trained_models = []
        for c in clients:
            trained = self.trained_q.get()
            if trained is not None:
                trained_models.append(trained)
        return trained_models