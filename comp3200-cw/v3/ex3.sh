#!/bin/bash

alpha=0.9

numnodes=10
startupdelay=0
numsamples=6000

for beta in 9999 0 -0.5; do
    for i in {1..5}; do
        python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta 0 "ex3_($beta)_0"
        python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta 3 "ex3_($beta)_3"
        python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta 5 "ex3_($beta)_5"
        python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta 8 "ex3_($beta)_8"
    done
done