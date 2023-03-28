#!/bin/bash

alpha=0.9

numnodes=10
startupdelay=0
numsamples=6000

for i in {1..5}; do
    # 0.5 0.7 0.9 1 AVG
    for alpha in 0.5 0.7 0.9 1 0; do
        for beta in -9999 -1.0 -0.5 0 0.5 1.0; do
            for gamma in 0 3 5 8; do
                python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma "ex4B_($alpha)_($beta)_($gamma)"
            done
        done
    done
done