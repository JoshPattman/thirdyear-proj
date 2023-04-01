from queue import Queue
from threading import Lock

all_backends = []
all_backends_lock = Lock()

class LocalBackend:
    def __init__(self, node_id=None, neighbors=[]):
        self.node_id = node_id
        self.neighbors = neighbors
        self.update_callback = None
        with all_backends_lock:
            all_backends.append(self)

    def distribute_state(self, node_id, params, training_counter):
        with all_backends_lock:
            for a in all_backends:
                if a.node_id in self.neighbors:
                    if not a.update_callback == None:
                        a.update_callback(node_id, params, training_counter)

    def set_update_callback(self, callback):
        self.update_callback = callback

    def start(self):
        pass
