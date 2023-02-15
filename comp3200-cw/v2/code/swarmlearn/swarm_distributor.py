import numpy as np
from threading import Lock
import logging

class SwarmDistributor:
    def __init__(self, initial_model_params, backend, neighbor_full_sync_weight=0.5, sync_initial_params=True, logger=None):
        if logger is None:
            logger = logging.getLogger('dummy')
        self.logger = logger

        self.training_params_lock = Lock()
        self.training_params = np.copy(initial_model_params)
        self.training_counter=0

        self.sync_initial_params = sync_initial_params

        self.neighbor_full_sync_weight = neighbor_full_sync_weight

        self.backend=backend
        self.backend.set_expected_length(self.training_params.shape[0])
        def pf():
            return self.get_training_params(training_counter=True)
        self.backend.register_param_function(pf)
        self.logger.debug("Distributor ready")

    def start(self):
        self.logger.debug("Distributor starting")
        self.backend.start()
        if self.sync_initial_params:
            (neighbor_params, neighbor_training_counters) = self.backend.query_params(warn=False)
            if len(neighbor_params) > 0:
                self.training_params = np.mean(np.array(neighbor_params), axis=0)
        self.backend.enable_incoming()
        self.logger.debug("Distributor started")

    # Syncs this network with neighboring networks (one way sync, does not change neighbors)
    def sync(self, use_training_counter=False):
        (neighbor_params, neighbor_training_counters) = self.backend.query_params()
        if len(neighbor_params) > 0:
            with self.training_params_lock:
                if not use_training_counter:
                    avg_neighbor_params = np.mean(np.array(neighbor_params), axis=0)
                    self.training_params = ((1-self.neighbor_full_sync_weight)*self.training_params) + (self.neighbor_full_sync_weight*avg_neighbor_params)
                else:
                    raise ValueError("Need to implement this")

    # Updates the parameter cache that is used when other nodes sync
    def update_params(self, new_params, copy=True, increment_training_counter_amount=1):
        with self.training_params_lock:
            if new_params.shape != self.training_params.shape:
                raise ValueError("incorrect shape")
            self.training_params = np.copy(new_params) if copy else new_params
            self.training_counter += increment_training_counter_amount

    # Gets the most up to date training params to train with
    def get_training_params(self, training_counter=False):
        with self.training_params_lock:
            if not training_counter:
                return np.copy(self.training_params)
            else:
                return (np.copy(self.training_params), self.training_counter)