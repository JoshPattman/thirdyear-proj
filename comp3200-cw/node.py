#!/usr/bin/env python
# coding: utf-8

# In[6]:


import protocol
import socket, sys
from keras.datasets import mnist
import numpy as np
import tensorflow as tf


# In[7]:


print('GPU: ', tf.config.experimental.list_physical_devices('GPU'))


# In[8]:


listen_port = sys.argv[1]
try:
    listen_port = int(listen_port)
except:
    listen_port = 9000
print("Listening on port %s"%listen_port)


# In[9]:


(train_X, train_Y), (test_X, test_Y) = mnist.load_data()


# In[27]:


from keras.layers import Dense, Input, Flatten, Conv2D, Reshape
from keras import Model
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy


# In[28]:


inp = Input((28,28))
out = Reshape((28,28,1))(inp)
out = Conv2D(16, (3,3), activation="relu")(out)
out = Flatten()(out)
out = Dense(128, activation="relu")(out)
out = Dense(10, activation="sigmoid")(out)
model = Model(inputs=inp, outputs=out)
model.compile(optimizer="adam", loss=SparseCategoricalCrossentropy(from_logits=True), metrics=[SparseCategoricalAccuracy()])
model.summary()


# In[ ]:


model.fit(train_X, train_Y, epochs=10)


# In[19]:


preds = model.predict(test_X)


# In[20]:


for i in range(100):
    print(test_Y[i], np.argmax(preds[i]))

