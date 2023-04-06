#!/bin/bash

numnodes=10
startupdelay=0
numsamples=1000

# Benchmark - See what the performance should be with just one node
#python3 just_model_mnist.py

# test0 - Testing FL
#python fed_mnist.py $numnodes $numsamples test0.1
#python fed_mnist.py $numnodes $numsamples test0.2
#python fed_mnist.py $numnodes $numsamples test0.3
#python fed_mnist.py $numnodes $numsamples test0.4
#python fed_mnist.py $numnodes $numsamples test0.5

# test1 - Testing a stable configuration of the algorithm
#export density=1.0
#export alpha=0.9 beta=0 gamma=4
#python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density test1.1
#python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density test1.2
#python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density test1.3
#python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density test1.4
#python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density test1.5

# test2 - 
# test1 - Testing a stable configuration of the algorithm
export density=1.0
export alpha=0.65 beta=0 gamma=3
python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density test5.1
#python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density test3.2
#python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density test3.3
#python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density test3.4
#python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density test3.5
