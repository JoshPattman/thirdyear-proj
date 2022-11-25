import json
import numpy as np
import socket
from flask import Flask
import requests
import logging
import os


logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

def json_model_weights(weights):
    list_weights = [[y.tolist() for y in x] for x in weights]
    return {"weights":list_weights}

def read_model_weights_json(js):
    list_weights = js["weights"]
    return [[np.array(y) for y in x] for x in list_weights]

def request_model_weights(addr):
    resp = requests.get("http://"+addr+"/get_weights")
    return read_model_weights_json(resp.json())

def start_listen_server(fn_get_json_weights, port):
    app = Flask(__name__)
    
    @app.route("/get_weights")
    def handler_get_weights():
        return json.dumps(fn_get_json_weights())
    
    app.run(port=port)