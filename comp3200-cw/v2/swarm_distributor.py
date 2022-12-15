import numpy as np

from queue import Queue
from threading import Lock

class ParamDistributor:
    def __init__(self, num_params, local_address, diff_weight=1, neighbor_full_sync_weight=0.5):
        self.local_params_lock = Lock()
        self.local_params = np.zeros(num_params)
        self.diffs_queue = Queue()
        self.diff_weight = diff_weight
        self.neighbor_full_sync_weight = neighbor_full_sync_weight

    # Reset the local parameters to provided ones without sending any model updates
    def reset_params(self, params):
        with self.local_params_lock:
            if not params.shape == self.local_params.shape:
                raise ValueError("Not correct shape of parameters")
            self.local_params = np.copy(params)

    # Get the most recent parameters
    def get_current_params(self):
        with self.local_params_lock:
            return np.copy(self.local_params)

    # Set the local params to our new updated parameters, then send diffs to all neighbors
    def set_updated_params(self, new_params, send_updates=True):
        with self.local_params_lock:
            if not new_params.shape == self.local_params.shape:
                raise ValueError("Not correct shape of parameters")
            diffs = new_params - self.local_params
            self.local_params = np.copy(new_params)
        if send_updates:
            self.send_all_neighbor_diffs(diffs)
    
    # Sync local parameters with neighbors. This will add up all the neighbor diffs then apply them. If full_model_sync is specified, full models will be aquired of al neighbors too
    def sync_params(self, full_model_sync=True, diff_sync=True, max_diffs=100, use_diff_averaging=False):
        # Create array for storing totals in
        with self.local_params_lock:
            total_diff = np.zeros_like(self.local_params)
        num_diffs = 0
        if diff_sync:
            # For as many times as we have diff that is less than max_diffs
            for _ in range(max_diffs):
                try:
                    # This will fail if diff q empty, which will cause the next bit of code to get run
                    diff = self.diffs_queue.get_nowait()
                    total_diff += diff
                    num_diffs += 1
                except:
                    break
            if num_diffs > 0:
                # we recv some diffs
                # should we use average or total
                if use_diff_averaging:
                    total_diff = total_diff / num_diffs

        #neighbor averaging
        total_neighbor_params = np.zeros_like(total_diff)
        num_neighbors=0
        if full_model_sync:
            responses = self.query_all_neighbor_params()
            num_neighbors = len(responses)
            for r in responses:
                total_neighbor_params += r

        if num_neighbors > 0:
            average_neighbor_params = total_neighbor_params/num_neighbors

        with self.local_params_lock:
                self.local_params += self.diff_weight * total_diff
                # We only average when there are some neighbor responses
                if num_neighbors > 0:
                    self.local_params = ((1-self.neighbor_full_sync_weight)*self.local_params) + (self.neighbor_full_sync_weight*average_neighbor_params)

    def send_all_neighbor_diffs(self, diffs):
        pass

    def query_all_neighbor_params(self):
        pass


if __name__ == "__main__":
    print("Checking")
    p = ParamDistributor(10, "8888")
    p.reset_params(np.random.uniform(size=(10,)))
    p.set_updated_params()
    print(p.get_current_params())