import tensorflow as tf
from keras.layers import Dense, Input, Flatten, Conv2D, Reshape
from keras import Model
from keras.datasets import mnist
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy

import random
import numpy as np
import sys
from datetime import datetime
import json


# python control.py <num_train_samples:60000> <uid:10> <epochs:1>
arg_training_samples, arg_uid, arg_epochs = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])


(train_X, train_Y), (test_X, test_Y) = mnist.load_data()

train_subset = random.sample(range(len(train_X)), arg_training_samples)
train_X = np.array([train_X[s] for s in train_subset])
train_Y = np.array([train_Y[s] for s in train_subset])

def evaluate_performance(model):
    preds = model.predict(test_X, verbose=False)
    num_correct = 0
    for i in range(len(test_Y)):
        if np.argmax(preds[i]) == test_Y[i]:
            num_correct += 1
    accuracy = 100*num_correct/len(test_Y)
    return accuracy

inp = Input((28,28))
out = Reshape((28,28,1))(inp)
out = Conv2D(16, (3,3), activation="relu")(out)
out = Conv2D(16, (3,3), activation="relu")(out)
out = Flatten()(out)
out = Dense(128, activation="relu")(out)
out = Dense(10, activation="sigmoid")(out)
model = Model(inputs=inp, outputs=out)
model.compile(optimizer="adam", loss=SparseCategoricalCrossentropy(), metrics=[SparseCategoricalAccuracy()])

data_times = []
data_accuracies = []

training_start = datetime.now()

for i in range(arg_epochs*5):
    model.fit(train_X, train_Y, epochs=1, verbose=False)
    data_accuracies.append(evaluate_performance(model))
    data_times.append((datetime.now()-training_start).total_seconds())
    print(data_accuracies[len(data_accuracies)-1])

print("Finished training")
filename = "./data/nodes:%s_samples:%s_uid:%s_epochs:%s.json"%(-1,arg_training_samples,arg_uid, arg_epochs)
print("Saving data log to %s"%filename)
with open(filename, "w") as f:
    f.write(json.dumps([(data_times, data_accuracies)]))