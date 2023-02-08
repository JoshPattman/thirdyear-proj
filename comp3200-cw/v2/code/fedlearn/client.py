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

class Client:
    def __init__(self, train_function, port, logger, array_json_encode=fast_json_encode_array):
        self.train_function = train_function
        self.app = Flask(__name__)
        self.array_json_encode = array_json_encode
        self.port=port
        self.logger = logger

        silence_flask()

        # listening for param requests
        @self.app.route("/train")
        def handler_get_params():
            self.logger.debug("Client recv params")
            # read incoming model params and callback point
            js = request.get_json(force=True)
            params, callback_endpoint, train_id = js["params"], js["callback_endpoint"], js["id"]
            params = json.loads(params)
            
            # run the training callback as a thread
            def callback_training(init_params, callback_endpoint, train_id):
                params = self.train_function(init_params)
                # data is actually a double encoded json in the response
                data = self.array_json_encode(params.tolist())
                requests.get(callback_endpoint, json={"params":data, "id":train_id})
                self.logger.debug("client send params")
            Thread(target=callback_training, args=(params, callback_endpoint, train_id), daemon=True).start()
            self.logger.debug("Client done recv params and training start")
            return "ok"

    def start(self):
        def start_fn():
            self.app.run(port=self.port)
        Thread(target=start_fn, daemon=True).start()
        self.logger.info("Client running on port %s"%self.port)
        #time.sleep(1)