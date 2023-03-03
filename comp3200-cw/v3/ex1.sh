#!/bin/bash

for i in {1..5}; do
    python3 swarm_mnist.py 10 0 6000 0 0 8 ex1_avg
    python3 swarm_mnist.py 10 0 6000 0 0.3 8 ex1_asr_0.3
    python3 swarm_mnist.py 10 0 6000 0 0.5 8 ex1_asr_0.5
    python3 swarm_mnist.py 10 0 6000 0 0.7 8 ex1_asr_0.7
    python3 swarm_mnist.py 10 0 6000 0 0.9 8 ex1_asr_0.9
    python3 swarm_mnist.py 10 0 6000 0 1 8 ex1_asr_1.0
done