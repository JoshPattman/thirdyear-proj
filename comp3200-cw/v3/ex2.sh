#!/bin/bash

alpha=0.9
gamma=0

numnodes=10
startupdelay=0
numsamples=6000

for i in {1..5}; do
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha 9999 $gamma ex2_9999
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha 2    $gamma ex2_2
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha 1    $gamma ex2_1
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha 0.5  $gamma ex2_0.5
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha 0    $gamma ex2_0
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha -0.5 $gamma ex2_-0.5
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha -1   $gamma ex2_-1
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha -2   $gamma ex2_-2
done