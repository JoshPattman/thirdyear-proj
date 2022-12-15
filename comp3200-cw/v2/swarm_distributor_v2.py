import numpy as np
from threading import Lock

class SwarmParamDistributor:
    def __init__(self, num_params, backend, diff_weight=1, neighbor_full_sync_weight=0.5, use_updated_params=True):
        self.training_params_lock = Lock()
        self.updated_params_lock = Lock()
        self.training_params = np.zeros(num_params)
        self.updated_params = np.zeros(num_params)

        self.diff_weight = diff_weight
        self.neighbor_full_sync_weight = neighbor_full_sync_weight

        self.backend=backend
        backend.register_diff_callback(self.on_recv_diff)
        if use_updated_params:
            backend.register_param_function(self.get_updated_params)
        else:
            backend.register_param_function(self.get_training_params)

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
            if not new_params.shape == self.training_params.shape:
                raise ValueError("Not correct shape of parameters")
            diffs = new_params - self.training_params
            self.training_params += diffs
        with self.updated_params_lock:
            self.update_params += diffs
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
    def register_diff_callback(self, f):
        pass
    def register_param_function(self, f):
        pass
    def query_params(self):
        return []
    def send_diffs(self, ds):
        pass

if __name__ == "__main__":
    SwarmParamDistributor(10, DummyBackend())