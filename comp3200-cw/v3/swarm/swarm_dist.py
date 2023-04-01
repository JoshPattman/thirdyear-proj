import logging, random, string, time
import numpy as np
from threading import Lock

class DummyBackend:
    def set_update_callback(self, callback):
        pass
    def distribute_state(self, node_id, params, training_counter):
        pass
    def start():
        pass

class SwarmDist:
    def __init__(self, backend, num_parameters, initial_params=None):
        self.backend = backend
        self.backend.set_update_callback(self.on_new_params)

        self.latest_states = {}
        self.latest_states_lock = Lock()
        self.training_counter = 0
        self.local_update_lock = Lock()

        if initial_params is None:
            self.local_params = np.zeros(num_parameters)
        else:
            self.local_params = np.copy(np.array(initial_params))

    def start(self):
        self.backend.start()

    def on_new_params(self, node_id, params, training_counter):
        with self.latest_states_lock:
            if node_id in self.latest_states:
                state = self.latest_states[node_id]
                if state['tc'] < training_counter:
                    # This is new data
                    self.latest_states[node_id] = {"params":params, "tc":training_counter}
                    # In the future maby bounce this update to all neighbors
            else:
                self.latest_states[node_id] = {"params":params, "tc":training_counter}

    def get_state(self):
        with self.local_update_lock:
            return np.copy(self.local_params), self.training_counter

    def update_local_params(self, params, increment=1):
        with self.local_update_lock:
            self.local_params = np.copy(params)
            self.training_counter += increment
            tccopy = self.training_counter
        self.backend.distribute_state(np.copy(params), tccopy)

    def sync(self, alpha=0.6, beta=99999999, gamma=8, use_ASR=True, max_attempts=10):
        # Only repeat a certain number of times to prevent infinite loops
        for i in range(max_attempts):
            neighbor_states = []
            # Find all neighbor states who are synced up with us
            with self.latest_states_lock:
                for nsid in self.latest_states:
                    ns = self.latest_states[nsid]
                    if ns['tc'] + beta >= self.training_counter:
                        neighbor_states.append(ns.copy())
            # Ensure we have enough neighbors to do more training
            if len(neighbor_states) >= gamma:
                # Average their params and also their tcs
                neighbor_params = [x['params'] for x in neighbor_states]
                neighbor_tcs = [x['tc'] for x in neighbor_states]
                if not use_ASR:
                    with self.local_update_lock:
                        neighbor_params.append(np.copy(self.local_params))
                        neighbor_tcs.append(self.training_counter)
                        avg_neighbor_params = np.mean(neighbor_params, axis=0)
                        avg_neighboar_tc = np.mean(neighbor_tcs)
                        self.local_params = avg_neighbor_params
                        self.training_counter = avg_neighboar_tc
                else:
                    avg_neighbor_params = np.mean(neighbor_params, axis=0)
                    avg_neighboar_tc = np.mean(neighbor_tcs)
                    with self.local_update_lock:
                        self.local_params = (1-alpha)*self.local_params + alpha*avg_neighbor_params
                        self.training_counter = (1-alpha)*self.training_counter + alpha*avg_neighboar_tc
                #print("Synced with %d neighbors" % len(neighbor_states))
                return
            # Repeat if we did not have enough neighbors
            time.sleep(1)
        print("Failed to sync with enough neighbors. Our cache is", self.latest_states.keys(), "and gamma is", gamma)
                    