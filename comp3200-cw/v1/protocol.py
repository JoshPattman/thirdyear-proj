import json
import numpy as np
import socket

# 1mb buffer
MAX_BUF_SIZE = int(65536/2)#int(1048576/2)#

def new_soc():
    soc = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    #soc.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, MAX_BUF_SIZE)
    #soc.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, MAX_BUF_SIZE)
    return soc

def packetize(s, max_size=MAX_BUF_SIZE):
    p = []
    while len(s) > 0:
        if len(s) < max_size:
            p.append(s)
            s=""
        else:
            p.append(s[:max_size])
            s = s[max_size:]
    return p

def readpackets(soc, exp_size, max_size=MAX_BUF_SIZE):
    s = ""
    while exp_size > 0:
        if exp_size < max_size:
            s += soc.recv(exp_size).decode("utf-8")
            exp_size = 0
        else:
            s += soc.recv(max_size).decode("utf-8")
            exp_size -= max_size
    return s

# Useful for a lot of things functions
def send_string_message(soc, s):
    payload = s.encode("utf-8")
    soc.sendall(("%s;"%len(payload)).encode("utf-8"))
    for p in packetize(payload):
        soc.sendall(p)
    #soc.sendall(payload)

def send_json_message(soc, data):
    send_string_message(soc, json.dumps(data))

def read_string_message(soc):
    s = ""
    while True:
        char = soc.recv(1).decode("utf-8")
        if char == ";":
            msg_len = int(s)
            payload = readpackets(soc, msg_len)#soc.recv(msg_len).decode("utf-8")
            if not msg_len == len(payload):
                raise ValueError("The recived message length was not the same as it should have been. This is cause of pythons awful sockets")
            return payload
        s += char

def read_json_message(soc):
    return json.loads(read_string_message(soc))


# Actual protocol stuff

# Weights should be a list of numpy arrays
def send_model_weights(soc, weights):
    list_weights = [[y.tolist() for y in x] for x in weights]
    send_json_message(soc, {"weights":list_weights})

def read_model_weights(soc):
    js = read_json_message(soc)
    list_weights = js["weights"]
    return [[np.array(y) for y in x] for x in list_weights]

def send_model_request(soc):
    send_string_message(soc, "REQ_MODEL")

def is_model_request(s):
    return s == "REQ_MODEL"