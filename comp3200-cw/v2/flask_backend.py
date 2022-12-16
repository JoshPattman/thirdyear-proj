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
    def __init__(self, port, neighbors, logger=None, use_gzip=False):
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
            data = json.dumps(params.tolist())

            if self.use_gzip:
                compressed = gzip.compress(data.encode('utf8'), 9)
                return compressed
            else:
                return data

        # listening for diff updates
        @self.app.route("/add_diff", methods = ["POST"])
        def handler_add_diff():
            if self.diff_callback is None:
                self.logger.error("The diff function has not been set")
                return "server_error"
            try:
                js = request.get_json(force=True)
                diff = np.array(js)
                if not (diff.shape[0] == self.expected_length):
                    raise ValueError("json diff did not have correct length")
                else:
                    self.diff_callback(diff)
                    return "ok"
            except Exception as e:
                self.logger.warning("json diff was not in correct format: %s"%e)
                return "bad_message"
        
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
                tstart = datetime.now()
                response = requests.get(f"http://{addr}/get_params", timeout=10)
                self.logger.debug("time for model: %s"%(datetime.now() - tstart).total_seconds())
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
        return
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
        