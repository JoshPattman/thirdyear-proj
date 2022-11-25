import protocol_v2 as protocol
import numpy as np
import threading, time, socket
from queue import Queue

import tensorflow as tf
from keras.layers import Dense, Input, Flatten, Conv2D, Reshape
from keras import Model
from keras.datasets import mnist
from keras.models import clone_model
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
from keras.initializers import glorot_uniform

print_lock=threading.Lock()
def thread_print(x):
    with print_lock:
        print(x)
        
def new_model():
    inp = Input((28,28))
    out = Reshape((28,28,1))(inp)
    out = Conv2D(16, (3,3), activation="relu")(out)
    out = Conv2D(16, (3,3), activation="relu")(out)
    out = Flatten()(out)
    out = Dense(128, activation="relu")(out)
    out = Dense(10, activation="sigmoid")(out)
    model = Model(inputs=inp, outputs=out)
    return model

class Node:
    # This will start the node listening
    def __init__(self, port, neighbors_addresses, model_ref=None):
        self.port=port
        self.neighbors_addresses = neighbors_addresses
        self.updates_queue = Queue()
        (self.train_X, self.train_Y), (self.test_X, self.test_Y) = mnist.load_data()
        if model_ref == None:
            self.model = new_model()
        else:
            self.model=clone_model(model_ref)
        self.model.compile(optimizer="adam", loss=SparseCategoricalCrossentropy(), metrics=[SparseCategoricalAccuracy()])
        self.log("Starting update thread")
        threading.Thread(target=self.listener_loop).start()
        threading.Thread(target=self.update_loop).start()
    
    def update_loop(self):
        def get_accuracy():
            preds = self.model.predict(self.test_X, verbose=False)
            num_correct = 0
            for i in range(len(self.test_Y)):
                if np.argmax(preds[i]) == self.test_Y[i]:
                    num_correct += 1
            return (100*num_correct/len(self.test_Y))
        self.log("Initial accuracy: %s"%get_accuracy())
        for i in range(1):
            self.send_to_neighbors()
            self.model.fit(self.train_X, self.train_Y, epochs=1, verbose=False)
            self.log("Accuracy: %s"%get_accuracy())
            
    def listener_loop(self):
        self.log("Listening for connections")
        s = protocol.new_soc()
        s.bind(("", self.port))
        s.listen()
        def handle(soc, addr):
            self.log("%s has connected"%(addr,))
            try:
                with soc:
                    weights = protocol.read_model_weights(soc)
                self.updates_queue.put(weights)
                self.log("Added wights from %s to queue"%(addr,))
            except Exception as e:
                self.log("Failure when communicating with %s: %s"%(addr, e), is_err=True)
        while True:
            soc, addr = s.accept()
            threading.Thread(target=handle, args=(soc,addr)).start()
        
    def send_to_neighbors(self):
        weights = []
        for layer in self.model.layers:
            weights.append(layer.get_weights())
        def send_to_neighbor(addr):
            try:
                s = protocol.new_soc()
                s.connect(addr)
                with s:
                    protocol.send_model_weights(s, weights)
                #self.log("Sent weights to %s"%addr)
            except Exception as e:
                self.log("Could not send weights to %s: %s"%(addr,e), is_warn=True)
        threads = []
        for n in self.neighbors_addresses:
            t = threading.Thread(target=send_to_neighbor, args=(n,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        self.log("Finished sending weights to all neighbors")
        
    def log(self, msg, is_err=False, is_warn=False):
        reset="\033[0m"
        error="\033[031m"
        warn="\033[033m"
        log="\033[036m"
        if is_err:
            thread_print(f"{error}[%s] ERROR >{reset} %s"%(self.port, msg))
        elif is_warn:
            thread_print(f"{warn}[%s] WARN  >{reset} %s"%(self.port, msg))
        else:
            thread_print(f"{log}[%s] LOG   >{reset} %s"%(self.port, msg))