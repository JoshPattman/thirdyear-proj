#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("Loading modules")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
print("Loading modules")
import swarm_node
print("Loading modules")
import importlib
print("Loading modules")
num_nodes = 5
importlib.reload(swarm_node)

node_ports = [x for x in range(9000, 9000+num_nodes)]
print(f"Ports: {node_ports}")
init_model = swarm_node.new_model()
#for p in node_ports:
    #swarm_node.Node(p, ["localhost:%s"%x for x in node_ports if x != p], model_ref=init_model)
#print("Started all nets")

