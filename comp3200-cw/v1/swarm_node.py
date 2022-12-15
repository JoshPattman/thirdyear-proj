import protocol_v2 as protocol
import numpy as np
import threading, time, socket, random
from queue import Queue
from datetime import datetime

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
        
def log(sender, msg, is_err=False, is_warn=False):
        reset="\033[0m"
        error="\033[031m"
        warn="\033[033m"
        log="\033[036m"
        if is_err:
            thread_print(f"{error}[%s] ERROR >{reset} %s"%(sender, msg))
        elif is_warn:
            thread_print(f"{warn}[%s] WARN  >{reset} %s"%(sender, msg))
        else:
            thread_print(f"{log}[%s] LOG   >{reset} %s"%(sender, msg))
        
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
    def __init__(self, port, neighbors_addresses, model_ref=None, num_train_samples=10000):
        self.port=port
        self.neighbors_addresses = neighbors_addresses
        self.weights_lock = threading.Lock()
        (self.train_X, self.train_Y), (self.test_X, self.test_Y) = mnist.load_data()
        train_subset = random.sample(range(len(self.train_X)), num_train_samples)#[random.randint(0,len(self.train_X)-1) for x in range(num_train_samples)]
        self.train_X = np.array([self.train_X[s] for s in train_subset])
        self.train_Y = np.array([self.train_Y[s] for s in train_subset])
        if model_ref == None:
            self.model = new_model()
        else:
            self.model=clone_model(model_ref)
        self.model.compile(optimizer="adam", loss=SparseCategoricalCrossentropy(), metrics=[SparseCategoricalAccuracy()])
        self.update_last_model_weights()
        self.log("Starting update thread")
        self.train_times = []
        self.train_accuracies = []
        self.start_time = datetime.now()
        threading.Thread(target=self.listener_loop).start()
        threading.Thread(target=self.update_loop).start()
    
    def update_loop(self):
        self.start_time = datetime.now()
        self.evaluate_performance("no training", include_sample=True)
        # How many times are we going to loop
        i = 0
        while True:
            # Do a step of training
            self.model.fit(self.train_X, self.train_Y, epochs=1, verbose=False)
            # Update the cached model weights that get sent to other nodes
            self.update_last_model_weights()
            # Test accuracy
            self.evaluate_performance("pre avg step (%s)"%i, color="\033[035m")
            # Get all other weights
            weightsQ = Queue()
            def add_weights_to_q(addr):
                try:
                    w = protocol.request_model_weights(addr)
                    weightsQ.put(w)
                except Exception as e:
                    self.log("Failed to get weights from %s: %s"%(addr, e),is_err=True)
                    weightsQ.put(None)
            for n in self.neighbors_addresses:
                threading.Thread(target=add_weights_to_q, args=(n,)).start()
            tstart = datetime.now()
            while weightsQ.qsize() < len(self.neighbors_addresses):
                time.sleep(0.1)
                if (datetime.now() - tstart).total_seconds() >= 5:
                    break
            recv_weights = []
            while weightsQ.qsize() > 0:
                w = weightsQ.get()
                if w is not None:
                    recv_weights.append(w)
            # Merge weights together
            with self.weights_lock:
                new_w = []
                for ai in range(len(self.last_weights)):
                    a_w = []
                    for bi in range(len(self.last_weights[ai])):
                        total = np.copy(self.last_weights[ai][bi])
                        for rw in recv_weights:
                            total += rw[ai][bi]
                        av = total/(len(recv_weights)+1)
                        a_w.append(av)
                    new_w.append(a_w)
            for layer in range(len(self.model.layers)):
                self.model.layers[layer].set_weights(new_w[layer])
            # Test accuracy again
            self.evaluate_performance("post avg nstep (%s)"%i, color="\033[034m", include_sample=True)
            if (datetime.now()-self.start_time).total_seconds() > 240:
                break
            i += 1
        
    def listener_loop(self):
        self.log("Listening for weight requests")
        def get_json_last_weights():
            with self.weights_lock:
                return protocol.json_model_weights(self.last_weights)
        protocol.start_listen_server(get_json_last_weights, self.port)
        
    def update_last_model_weights(self):
        with self.weights_lock:
            self.last_weights = []
            for layer in self.model.layers:
                self.last_weights.append(layer.get_weights())
                
    def evaluate_performance(self, tag, color="\033[0m", include_sample=False):
        tstart = datetime.now()
        preds = self.model.predict(self.test_X, verbose=False)
        tdiff = round((datetime.now()-tstart).total_seconds(),2)
        num_correct = 0
        for i in range(len(self.test_Y)):
            if np.argmax(preds[i]) == self.test_Y[i]:
                num_correct += 1
        accuracy = 100*num_correct/len(self.test_Y)
        if include_sample:
            self.train_times.append((datetime.now()-self.start_time).total_seconds())
            self.train_accuracies.append(accuracy)
        self.log(f"Accuracy {color}(%s)\033[0m <%ss>: {color}%s\033[0m"%(tag,tdiff,accuracy))
        
    def log(self, msg, is_err=False, is_warn=False):
        log(self.port, msg, is_err=is_err, is_warn=is_warn)