#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 


# In[2]:


import swarm_node
import importlib


# In[3]:


num_nodes = 5

importlib.reload(swarm_node)
node_ports = [x for x in range(9000, 9000+num_nodes)]
init_model = swarm_node.new_model()
for p in node_ports:
    swarm_node.Node(p, [("localhost",x) for x in node_ports if x != p], model_ref=init_model)

