import json
import numpy as np

# Useful for a lot of things functions
def send_string_message(soc, s):
    payload = s.encode("utf-8")
    soc.write(("%s;"%len(payload)).encode("utf-8"))
    soc.write(payload)

def send_json_message(soc, data):
    send_string_message(soc, json.dumps(data))

def read_string_message(soc):
    s = ""
    while True:
        char = soc.recv(1).decode("utf-8")
        if char == ";":
            msg_len = int(s)
            payload = soc.recv(msg_len).decode("utf-8")
            return payload
        s += char

def read_json_message(soc):
    return json.loads(read_string_message(soc))


# Actual protocol stuff

# Weights should be a list of numpy arrays
def send_model_weights(soc, weights):
    list_weights = [x.tolist() for x in weights]
    send_json_message(soc, {"weights":list_weights})

def read_model_weights(soc):
    js = read_json_message(soc)
    list_weights = js["weights"]
    return [np.array(x) for x in list_weights]

def send_model_request(soc):
    send_string_message(soc, "REQ_MODEL")

def is_model_request(s):
    return s == "REQ_MODEL"