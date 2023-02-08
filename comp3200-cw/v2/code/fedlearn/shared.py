import logging
import click
import random
import string

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

def dummy_train_function(params):
    return params

def fast_json_encode_array(arr):
    s = "["
    for e in arr:
        s += f"{e:.5f},"
    if len(s) > 1:
        s = s[:len(s)-1]
    s += "]"
    return s

def get_random_string(n):
    return ''.join(random.choice(string.ascii_letters) for i in range(n))