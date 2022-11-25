#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 


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


# In[38]:


inp = Input((28,28))
out = Reshape((28,28,1))(inp)
out = Conv2D(16, (3,3), activation="relu")(out)
out = Conv2D(16, (3,3), activation="relu")(out)
out = Flatten()(out)
out = Dense(128, activation="relu")(out)
out = Dense(10, activation="sigmoid")(out)
model = Model(inputs=inp, outputs=out)
model.compile(optimizer="adam", loss=SparseCategoricalCrossentropy(), metrics=[SparseCategoricalAccuracy()])
model.summary()


# In[39]:


model.fit(train_X, train_Y, epochs=3)


# In[40]:


preds = model.predict(test_X)


# In[41]:


num_correct = 0
for i in range(len(test_Y)):
    if np.argmax(preds[i]) == test_Y[i]:
        num_correct += 1
print("Accuracy: %s"%(100*num_correct/len(test_Y)))

