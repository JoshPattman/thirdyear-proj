import tensorflow as tf

gpus = [x.name for x in tf.config.list_logical_devices('GPU')]

print(gpus)