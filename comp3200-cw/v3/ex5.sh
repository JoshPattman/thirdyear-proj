#!/bin/bash

numnodes=10
startupdelay=0
numsamples=6000


#for i in {1..5}; do
    # Density of the graph
    for density in 0 0.15 0.75 1; do
        # 0.5 0.7 0.9 1 AVG
        for alpha in 0.5 0.7 0.9 1 0; do
            # How far behind are neighbors allowed to be 
            for beta in 0 0.5 1.0 2.0; do
                # How many neighbors to wait for
                for gamma in 0 2 3 5; do
                    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density
                done
            done
        done
    done
#done