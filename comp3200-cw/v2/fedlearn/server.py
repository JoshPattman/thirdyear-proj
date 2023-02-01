import numpy as np
from flask import Flask, request, make_response
import requests
import json
import numpy as np
from queue import Queue
from threading import Thread
import time
from datetime import datetime
from .shared import *


class Server:
    def __init__(self, ip, port, nodes_addrs, logger, array_json_encode=fast_json_encode_array):
        self.ip = ip
        self.logger = logger
        self.port = port
        self.app = Flask(__name__)
        self.array_json_encode = array_json_encode
        self.current_id = ""
        self.nodes_addrs = nodes_addrs

        self.recv_params = Queue()

        silence_flask()

        # listening for param requests
        @self.app.route("/callback")
        def handler_get_params():
            # read incoming model params
            js = request.get_json(force=True)
            params, id = js["params"], js["id"]
            # params is double encoded
            params = json.loads(params)
            # add the recv params to the queue
            if id == self.current_id:
                self.recv_params.put(params)
            return "ok"

    def start(self):
        def start_fn():
            self.app.run(port=self.port)
        Thread(target=start_fn, daemon=True).start()
        time.sleep(1)

    def train(self, params):
        # Setup the current id
        self.current_id = get_random_string(10)
        # Setup what we need to send
        callback_addr = "http://%s:%s/callback"%(self.ip, self.port)
        encoded_params = self.array_json_encode(params)
        send_data = {"params":encoded_params, "callback_endpoint":callback_addr, "id":self.current_id}
        # Send to all neighbors
        for n in self.nodes_addrs:
            self.logger.debug("Requesting node %s"%n)
            requests.get(n+"/train", json=send_data)
        
        # Wait for neighbor responses and read them
        while self.recv_params.qsize() < len(self.nodes_addrs):
            time.sleep(0.1)
        all_params = []
        for n in self.nodes_addrs:
            all_params.append(self.recv_params.get())
        self.logger.info("Recv networks from %s nodes"%len(all_params))
        # reset the current id to ignore old nodes
        self.current_id = ""

        # average the params and return
        total_neighbor_params = np.zeros_like(all_params[0])
        for p in all_params:
            total_neighbor_params += p
        avg_neighbor_params = total_neighbor_params / len(all_params)
        return avg_neighbor_params
