#!/bin/bash

beta=0
gamma=0

numnodes=10
startupdelay=0
numsamples=6000

for i in {1..5}; do
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples 0   $beta $gamma ex1_avg
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples 0.3 $beta $gamma ex1_asr_0.3
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples 0.5 $beta $gamma ex1_asr_0.5
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples 0.7 $beta $gamma ex1_asr_0.7
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples 0.9 $beta $gamma ex1_asr_0.9
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples 1   $beta $gamma ex1_asr_1.0
done