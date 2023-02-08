import numpy as np

def flatten_model(model):
    return np.concatenate([np.reshape(p, (-1,)) for p in model.get_weights()])

def unflatten_model(model, flat_params):
    params_shapes = get_model_params_shapes(model)
    params = []
    n = 0
    for ps in params_shapes:
        num_params = tuple_product(ps)
        params.append(np.reshape(flat_params[n:n+num_params], ps))
        n += num_params
    model.set_weights(params)

def get_model_params_shapes(model):
    params = model.get_weights()
    params_shapes = []
    for p in params:
        params_shapes.append(p.shape)
    return params_shapes

def tuple_product(t):
        x = 1
        for ta in t:
            x *= ta
        return x