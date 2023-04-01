from queue import Queue
from threading import Lock
import string, random

all_backends = []
all_backends_lock = Lock()

def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

class LocalBackend:
    def __init__(self, node_id=None, neighbors=[]):
        if node_id is not None:
            self.node_id = node_id
        else:
            self.node_id = get_random_string(5)
        self.neighbors = neighbors
        self.update_callback = None
        with all_backends_lock:
            all_backends.append(self)

    def distribute_state(self, params, training_counter):
        with all_backends_lock:
            for a in all_backends:
                if a.node_id in self.neighbors:
                    if not a.update_callback == None:
                        a.update_callback(self.node_id, params, training_counter)

    def set_update_callback(self, callback):
        self.update_callback = callback

    def start(self):
        pass
