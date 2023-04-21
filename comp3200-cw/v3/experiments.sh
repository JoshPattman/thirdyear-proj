#!/bin/bash

numnodes=10
startupdelay=0

# Benchmark - See what the performance should be with just one node
#python3 just_model_mnist.py

# Tests
# 0-2 - FL Dense
# 3-5 - SL Dense
# 6-8 - SL sparse 0.75
# 9-11 - SL sparse 0.5
# 12-14 - SL sparse 0.25
# 15-17 - SL sparse 0
# 18-20 - FL w/ node dropout
# 21-23 - SL w/ node dropout


## ------------------------ FL Dense ------------------------
#export dropout=0
#export numsamples=6000 eps=2
#for i in {1..5}; do
#    python3 fed_mnist.py $numnodes $numsamples $eps $dropout test0.$i
#done
#
#export numsamples=1000 eps=5
#for i in {1..5}; do
#    python3 fed_mnist.py $numnodes $numsamples $eps $dropout test1.$i
#done
#
#export numsamples=100 eps=15
#for i in {1..5}; do
#    python3 fed_mnist.py $numnodes $numsamples $eps $dropout test2.$i
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
#
## ------------------------ SL sparse 0.75 ------------------------
## Mean min steps, Mean cons per node = 1.2, 7.2
#export density=0.75 alpha=0.9 beta=0.5 gamma=7
#
#export numsamples=6000 eps=2
#for i in {1..5}; do
#    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test6.$i
#done
#
#export numsamples=1000 eps=5
#for i in {1..5}; do
#    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test7.$i
#done
#
#export numsamples=100 eps=15
#for i in {1..5}; do
#    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test8.$i
#done
#
## ------------------------ SL sparse 0.5 ------------------------
## Mean min steps, Mean cons per node = 1.4, 5.4
#export density=0.5 alpha=0.9 beta=0.5 gamma=5
#
#export numsamples=6000 eps=2
#for i in {1..5}; do
#    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test9.$i
#done
#
#export numsamples=1000 eps=5
#for i in {1..5}; do
#    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test10.$i
#done
#
#export numsamples=100 eps=15
#for i in {1..5}; do
#    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test11.$i
#done
#
## ------------------------ SL sparse 0.25 ------------------------
## Mean min steps, Mean cons per node = 1.7, 3.6
export density=0.25 alpha=0.9 beta=0.5 gamma=3
#
export numsamples=6000 eps=2
for i in {1..5}; do
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test12.$i
done
#
#export numsamples=1000 eps=5
#for i in {1..5}; do
#    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test13.$i
#done
#
#export numsamples=100 eps=15
#for i in {1..5}; do
#    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test14.$i
#done
#
## ------------------------ SL sparse 0 ------------------------
## Mean min steps, Mean cons per node = 3.0, 1.8
#export density=0 alpha=0.9 beta=0.5 gamma=1
#
#export numsamples=6000 eps=2
#for i in {1..5}; do
#    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test15.$i
#done
#
#export numsamples=1000 eps=5
#for i in {1..5}; do
#    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test16.$i
#done
#
#export numsamples=100 eps=15
#for i in {1..5}; do
#    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test17.$i
#done

# ------------------------ FL w/ node dropout ------------------------
export dropout=3
#export numsamples=6000 eps=2
#for i in {1..5}; do
#    python3 fed_mnist.py $numnodes $numsamples $eps $dropout test18.$i
#done
#
#export numsamples=1000 eps=5
#for i in {1..5}; do
#    python3 fed_mnist.py $numnodes $numsamples $eps $dropout test19.$i
#done

export numsamples=100 eps=15
for i in {1..5}; do
    python3 fed_mnist.py $numnodes $numsamples $eps $dropout test20.$i
done