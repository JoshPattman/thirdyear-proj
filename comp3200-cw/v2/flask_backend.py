from flask import Flask, request, make_response
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
import gzip

random.seed(datetime.now().timestamp())

def json_encode_array(arr):
    s = "["
    for e in arr:
        s += f"{e:.5f},"
    if len(s) > 1:
        s = s[:len(s)-1]
    s += "]"
    return s

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
    def __init__(self, port, neighbors, logger=None, use_gzip=False, use_fast_arr=True):
        if logger is None:
            logger = logging.getLogger('dummy')
        self.logger = logger
        self.expected_length = 0
        self.port = port
        self.diff_callback = None
        self.param_function = None
        self.neighbors = neighbors
        self.use_gzip = use_gzip

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
            if use_fast_arr:
                data = json_encode_array(params)
            else:
                data = json.dumps(params.tolist())

            if self.use_gzip:
                compressed = gzip.compress(data.encode('utf8'), 9)
                return compressed
            else:
                return data
        
    def start(self):
        def start_fn():
            self.app.run(port=self.port)
        self.logger.info("Starting server")
        start_thread = Thread(target=start_fn)
        start_thread.setDaemon(True)
        start_thread.start()
        time.sleep(1)
        
    def set_expected_length(self, n):
        self.expected_length = n

    def register_param_function(self, f):
        self.param_function = f

    def query_params(self):
        responses = Queue()
        def request_function(addr):
            try:
                response = requests.get(f"http://{addr}/get_params", timeout=10)
                if self.use_gzip:
                    decompressed = gzip.decompress(response.content)
                    decoded = decompressed.decode('utf-8')
                    listed = json.loads(decoded)
                    params = np.array(listed)
                else:
                    params = np.array(response.json())
                if not (params.shape[0] == self.expected_length):
                    raise ValueError("params did not have correct shape")
                responses.put(params)        
            except Exception as e:
                self.logger.warning("Failed to get params: %s"%e)
                responses.put(None)

        for n in self.neighbors:
            start_thread = Thread(target=request_function, args=(n,))
            start_thread.setDaemon(True)
            start_thread.start()

        while responses.qsize() < len(self.neighbors):
            time.sleep(0.1)

        all_params = []
        for n in self.neighbors:
            params = responses.get()
            if not (params is None): 
                all_params.append(params)
        return all_params
        