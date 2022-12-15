import numpy as np
from threading import Lock

class SwarmDistributor:
    def __init__(self, initial_model_params, backend, diff_weight=1, neighbor_full_sync_weight=0.5, use_updated_params=True, sync_initial_params=True):
        self.training_params_lock = Lock()
        self.updated_params_lock = Lock()
        self.training_params = np.copy(initial_model_params)
        self.updated_params = np.copy(initial_model_params)

        self.diff_weight = diff_weight
        self.neighbor_full_sync_weight = neighbor_full_sync_weight

        self.backend=backend
        self.backend.set_expected_length(self.training_params.shape[0])

        self.backend.register_diff_callback(self.on_recv_diff)
        if use_updated_params:
            self.backend.register_param_function(self.get_updated_params)
        else:
            self.backend.register_param_function(self.get_training_params)

        if sync_initial_params:
            neighbor_params = self.backend.query_params()
            # Only bother if we actually got a response
            if len(neighbor_params) > 0:
                total_neighbor_params = np.zeros_like(self.training_params)
                for p in neighbor_params:
                    total_neighbor_params += p
                avg_neighbor_params = total_neighbor_params / len(neighbor_params)
                self.training_params = np.copy(avg_neighbor_params)
                self.updated_params = np.copy(avg_neighbor_params)

        # Threading only starts below here
        self.backend.start()

    def sync(self, full_model_sync=True):
        # Update to latest diffs
        with self.updated_params_lock:
            updated_params = np.copy(self.updated_params)
        # If required, do a full model sync
        if full_model_sync:
            responses = self.backend.query_params()
            num_neighbors = len(responses)
            total_neighbor_params = np.zeros_like(updated_params)
            for r in responses:
                total_neighbor_params += r
            if num_neighbors > 0:
                average_neighbor_params = total_neighbor_params/num_neighbors
                updated_params = ((1-self.neighbor_full_sync_weight)*updated_params) + (self.neighbor_full_sync_weight*average_neighbor_params)
        # Copy new parameters into training params
        with self.training_params_lock:
            self.training_params = updated_params

    # This calculates the diff from the parameters at the last sync
    def update_params(self, new_params, send_updates=True):
        # We add the diffs to both the training paramters and also the updated parameters otherwise they will be lost on next sync
        with self.training_params_lock:
            if new_params.shape != self.training_params.shape:
                raise ValueError("incorrect shape")
            diffs = new_params - self.training_params
            self.training_params += diffs
        with self.updated_params_lock:
            self.updated_params += diffs
        if send_updates:
            self.backend.send_diffs(diffs)

    def on_recv_diff(self, diff):
        with self.updated_params_lock:
            self.updated_params += diff * self.diff_weight

    def get_training_params(self):
        with self.training_params_lock:
            return np.copy(self.training_params)

    def get_updated_params(self):
        with self.updated_params_lock:
            return np.copy(self.updated_params)

class DummyBackend:
    def __init__(self):
        pass
    def set_expected_length(self, n):
        pass
    def register_diff_callback(self, f):
        pass
    def register_param_function(self, f):
        pass
    def query_params(self):
        return []
    def send_diffs(self, ds):
        print("Send diffs %s"%ds)
    def start(self):
        pass
