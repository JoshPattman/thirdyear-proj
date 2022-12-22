import numpy as np
from threading import Lock

class SwarmDistributor:
    def __init__(self, initial_model_params, backend, neighbor_full_sync_weight=0.5, sync_initial_params=True):
        self.training_params_lock = Lock()
        self.training_params = np.copy(initial_model_params)

        self.neighbor_full_sync_weight = neighbor_full_sync_weight

        self.backend=backend
        self.backend.set_expected_length(self.training_params.shape[0])
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

        # Threading only starts below here
        self.backend.start()

    # Syncs this network with neighboring networks (one way sync, does not change neighbors)
    def sync(self):
        with self.training_params_lock:
            num_params = self.training_params.shape[0]
        responses = self.backend.query_params()
        num_neighbors = len(responses)
        total_neighbor_params = np.zeros(num_params)
        for r in responses:
            total_neighbor_params += r
        if num_neighbors > 0:
            total_neighbor_params = total_neighbor_params/num_neighbors
            with self.training_params_lock:
                self.training_params = ((1-self.neighbor_full_sync_weight)*self.training_params) + (self.neighbor_full_sync_weight*total_neighbor_params)

    # Updates the parameter cache that is used when other nodes sync
    def update_params(self, new_params, copy=True):
        # We add the diffs to both the training paramters and also the updated parameters otherwise they will be lost on next sync
        with self.training_params_lock:
            if new_params.shape != self.training_params.shape:
                raise ValueError("incorrect shape")
            self.training_params = np.copy(new_params) if copy else new_params

    # Gets the most up to date training params to train with
    def get_training_params(self):
        with self.training_params_lock:
            return np.copy(self.training_params)