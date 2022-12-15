from flask import Flask, request
import requests
import logging
import json
import numpy as np
from queue import Queue
from threading import Thread
import time
import click
import random
import string
from datetime import datetime

random.seed(datetime.now().timestamp())

def silence_flask():
    def secho(text, file=None, nl=None, err=None, color=None, **styles):
        pass

    def echo(text, file=None, nl=None, err=None, color=None, **styles):
        pass
    click.echo = echo
    click.secho = secho
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)

def get_random_string(n):
    return ''.join(random.choice(string.ascii_letters) for i in range(n))

class FlaskBackend:
    def __init__(self, port, neighbors, logger=None):
        if logger is None:
            logger = logging.getLogger('dummy')
        self.logger = logger
        self.expected_length = 0
        self.port = port
        self.diff_callback = None
        self.param_function = None
        self.neighbors = neighbors

        self.app = Flask(__name__)

        silence_flask()

        # listening for param requests
        @self.app.route("/get_params")
        def handler_get_params():
            if self.param_function is None:
                self.logger.error("The param function has not been set")
                return
            params = self.param_function()
            if not (params.shape[0] == self.expected_length):
                self.logger.error("That length of params is not correct")
            return json.dumps(params.tolist())

        # listening for diff updates
        @self.app.route("/add_diff")
        def handler_add_diff():
            if self.diff_callback is None:
                self.logger.error("The diff function has not been set")
                return
            try:
                js = request.get_json(force=True)
                diff = np.array(js)
                if not (diff.shape[0] == self.expected_length):
                    raise ValueError("json diff did not have correct length")
                else:
                    self.diff_callback(diff)
            except Exception as e:
                self.logger.warning("json diff was not in correct format: %s"%e)
            
        
    def start(self):
        def start_fn():
            self.app.run(port=self.port)
        self.logger.info("Starting server")
        Thread(target=start_fn).start()
        time.sleep(1)
        
    def set_expected_length(self, n):
        self.expected_length = n
    def register_diff_callback(self, f):
        self.diff_callback = f
    def register_param_function(self, f):
        self.param_function = f

    def query_params(self):
        responses = Queue()
        def request_function(addr):
            try:
                js = requests.get(f"http://{addr}/get_params", timeout=10).json()
                params = np.array(js)
                if not (params.shape[0] == self.expected_length):
                    raise ValueError("params did not have correct shape")
                responses.put(params)        
            except Exception as e:
                self.logger.warning("Failed to get params: %s"%e)
                responses.put(None)

        for n in self.neighbors:
            Thread(target=request_function, args=(n,)).start()

        while responses.qsize() < len(self.neighbors):
            time.sleep(0.1)

        all_params = []
        for n in self.neighbors:
            params = responses.get()
            if not (params is None): 
                all_params.append(params)
        return all_params

    def send_diffs(self, ds):
        data = ds.tolist()
        if not (len(data) == self.expected_length):
            self.logger.error("diff length not equal to data length")
            return
        def diff_function(addr):
            try:
                requests.post(f"http://{addr}/add_diff", json=data)
            except Exception as e:
                self.logger.warning("Failed to send diff: %s"%e)
        for n in self.neighbors:
            Thread(target=diff_function, args=(n,)).start()
        