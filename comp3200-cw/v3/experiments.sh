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
#export numsamples=6000 eps=2
#for i in {1..5}; do
#    python3 fed_mnist.py $numnodes $numsamples $eps test0.$i
#done
#
#export numsamples=1000 eps=5
#for i in {1..5}; do
#    python3 fed_mnist.py $numnodes $numsamples $eps test1.$i
#done
#
#export numsamples=100 eps=15
#for i in {1..5}; do
#    python3 fed_mnist.py $numnodes $numsamples $eps test2.$i
#done
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

# ------------------------ SL sparse 0.75 ------------------------
# Mean min steps, Mean cons per node = 1.2, 7.2
export density=0.75 alpha=0.9 beta=0.5 gamma=7

export numsamples=6000 eps=2
for i in {1..5}; do
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test6.$i
done

export numsamples=1000 eps=5
for i in {1..5}; do
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test7.$i
done

export numsamples=100 eps=15
for i in {1..5}; do
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test8.$i
done

# ------------------------ SL sparse 0.5 ------------------------
# Mean min steps, Mean cons per node = 1.4, 5.4
export density=0.5 alpha=0.9 beta=0.5 gamma=5

export numsamples=6000 eps=2
for i in {1..5}; do
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test9.$i
done

export numsamples=1000 eps=5
for i in {1..5}; do
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test10.$i
done

export numsamples=100 eps=15
for i in {1..5}; do
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test11.$i
done

# ------------------------ SL sparse 0.25 ------------------------
# Mean min steps, Mean cons per node = 1.7, 3.6
export density=0.25 alpha=0.9 beta=0.5 gamma=3

export numsamples=6000 eps=2
for i in {1..5}; do
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test12.$i
done

export numsamples=1000 eps=5
for i in {1..5}; do
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test13.$i
done

export numsamples=100 eps=15
for i in {1..5}; do
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test14.$i
done

# ------------------------ SL sparse 0 ------------------------
# Mean min steps, Mean cons per node = 3.0, 1.8
export density=0 alpha=0.9 beta=0.5 gamma=1

export numsamples=6000 eps=2
for i in {1..5}; do
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test15.$i
done

export numsamples=1000 eps=5
for i in {1..5}; do
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test16.$i
done

export numsamples=100 eps=15
for i in {1..5}; do
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test17.$i
done