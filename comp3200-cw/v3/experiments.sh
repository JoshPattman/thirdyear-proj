#!/bin/bash

# Benchmark - See what the performance should be with just one node
#python3 just_model_mnist.py

# Tests
# 0-2 FL Dense
# 3-5 SL Dense
# 6-8 SL Sparse 0.75
# 9-11 SL Sparse 0.5
# 12-14 SL Sparse 0.25
# 15-17 SL Sparse 0.0

export numnodes=10 startupdelay=0

export alpha=0.75 beta=0.5

for i in 3; do
    # FL Dense
    export numsamples=1000 eps=5
    python3 fed_mnist.py $numnodes $numsamples $eps test0.$i
    export numsamples=100 eps=10
    python3 fed_mnist.py $numnodes $numsamples $eps test1.$i
    export numsamples=25 eps=20
    python3 fed_mnist.py $numnodes $numsamples $eps test2.$i

    # SL Dense
    export density=1.0 gamma=8
    export numsamples=1000 eps=5
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test3.$i
    export numsamples=100 eps=10
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test4.$i
    export numsamples=25 eps=20
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test5.$i

    # SL Sparse 0.75
    export density=0.75 gamma=7
    export numsamples=1000 eps=5
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test6.$i
    export numsamples=100 eps=10
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test7.$i
    export numsamples=25 eps=20
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test8.$i

    # SL Sparse 0.5
    export density=0.5 gamma=5
    export numsamples=1000 eps=5
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test9.$i
    export numsamples=100 eps=10
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test10.$i
    export numsamples=25 eps=20
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test11.$i

    # SL Sparse 0.25
    export density=0.5 gamma=3
    export numsamples=1000 eps=5
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test12.$i
    export numsamples=100 eps=10
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test13.$i
    export numsamples=25 eps=20
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test14.$i

    # SL Sparse 0.0
    export density=0.0 gamma=1
    export numsamples=1000 eps=5
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test15.$i
    export numsamples=100 eps=10
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test16.$i
    export numsamples=25 eps=20
    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps test17.$i
done