#!/bin/bash

numnodes=10
startupdelay=0

# Benchmark - See what the performance should be with just one node
#python3 just_model_mnist.py

# Tests
# 1. FL 6000 samples
# 2. FL 1000 samples
# 3. FL 100 samples
# 4. SL 6000 samples
# 5. SL 1000 samples
# 6. SL 100 samples


# ------------------------ FL Dense ------------------------
export numsamples=6000 eps=2
for i in {1..5}; do
    python3 fed_mnist.py $numnodes $numsamples $eps test0.$i
done

export numsamples=1000 eps=5
for i in {1..5}; do
    python3 fed_mnist.py $numnodes $numsamples $eps test1.$i
done

export numsamples=100 eps=15
for i in {1..5}; do
    python3 fed_mnist.py $numnodes $numsamples $eps test2.$i
done
#
## ------------------------ SL Dense ------------------------
#export density=1.0 alpha=0.75 beta=0.5 gamma=8
#
#export numsamples=6000 eps=2
#for i in {1..5}; do
#    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test3.$i
#done
#
#export numsamples=1000 eps=5
#for i in {1..5}; do
#    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test4.$i
#done
#
#export numsamples=100 eps=15
#for i in {1..5}; do
#    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test5.$i
#done
#
#export density=1.0 alpha=0.9 beta=0.5 gamma=8
#export numsamples=100 eps=15
#for i in {1..5}; do
#    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test6.$i
#done
#
#export density=1.0 alpha=0.5 beta=0.5 gamma=8
#export numsamples=100 eps=15
#for i in {1..5}; do
#    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test7.$i
#done