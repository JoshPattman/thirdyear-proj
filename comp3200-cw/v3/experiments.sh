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

for i in {5..5}; do
    # FL Dense
#    export classes=10
#    export numsamples=1000 eps=5
#    python3 fed_mnist.py $numnodes $numsamples $eps $classes test0.$i
#    export numsamples=100 eps=10
#    python3 fed_mnist.py $numnodes $numsamples $eps $classes test1.$i
#    export numsamples=25 eps=20
#    python3 fed_mnist.py $numnodes $numsamples $eps $classes test2.$i
#
#    # SL Dense
#    export density=1.0 gamma=8 classes=10
#    export numsamples=1000 eps=5
#    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps $classes test3.$i
#    export numsamples=100 eps=10
#    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps $classes test4.$i
#    export numsamples=25 eps=20
#    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps $classes test5.$i
#
#    # SL Sparse 0.75
#    export density=0.75 gamma=7 classes=10
#    export numsamples=1000 eps=5
#    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps $classes test6.$i
#    export numsamples=100 eps=10
#    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps $classes test7.$i
#    export numsamples=25 eps=20
#    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps $classes test8.$i
#
#    # SL Sparse 0.5
#    export density=0.5 gamma=5 classes=10
#    export numsamples=1000 eps=5
#    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps $classes test9.$i
#    export numsamples=100 eps=10
#    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps $classes test10.$i
#    export numsamples=25 eps=20
#    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps $classes test11.$i
#
#    # SL Sparse 0.25
#    export density=0.5 gamma=3 classes=10
#    export numsamples=1000 eps=5
#    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps $classes test12.$i
#    export numsamples=100 eps=10
#    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps $classes test13.$i
#    export numsamples=25 eps=20
#    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps $classes test14.$i
#
#    # SL Sparse 0.0
#    export density=0.0 gamma=1 classes=10
#    export numsamples=1000 eps=5
#    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps $classes test15.$i
#    export numsamples=100 eps=10
#    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps $classes test16.$i
#    export numsamples=25 eps=20
#    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps $classes test17.$i
#
#    # FL Dense w/ 3 classes
#    export classes=3
#    export numsamples=1000 eps=5
#    python3 fed_mnist.py $numnodes $numsamples $eps $classes test18.$i
#    export numsamples=100 eps=10
#    python3 fed_mnist.py $numnodes $numsamples $eps $classes test19.$i
#    export numsamples=25 eps=20
#    python3 fed_mnist.py $numnodes $numsamples $eps $classes test20.$i
#
#    # SL Dense w/ 3 classes
#    export density=1.0 gamma=8 classes=3
#    export numsamples=1000 eps=5
#    python3 swarm_mnist.py $numnodes $startupdelay $numsamples $alpha $beta $gamma $density $eps $classes test210.$i # standard
#    python3 swarm_mnist.py $numnodes $startupdelay $numsamples 0.25 $beta $gamma $density $eps $classes test212.$i # divergent
#    export gamma=5
#    export numsamples=1000 eps=5
#    python3 swarm_mnist.py $numnodes 2 $numsamples $alpha $beta $gamma $density $eps $classes test211.$i # modified
#    export numsamples=100 eps=10
#    python3 swarm_mnist.py $numnodes 2 $numsamples $alpha $beta $gamma $density $eps $classes test22.$i
#    export numsamples=25 eps=20
#    python3 swarm_mnist.py $numnodes 2 $numsamples $alpha $beta $gamma $density $eps $classes test23.$i

    # SL Sparse 0.75 w/ 3 classes
#    export density=0.75 gamma=4 classes=3
#    export numsamples=1000 eps=5
#    python3 swarm_mnist.py $numnodes 2 $numsamples $alpha $beta $gamma $density $eps $classes test24.$i
#    export numsamples=100 eps=10
#    python3 swarm_mnist.py $numnodes 2 $numsamples $alpha $beta $gamma $density $eps $classes test25.$i
#    export numsamples=25 eps=20
#    python3 swarm_mnist.py $numnodes 2 $numsamples $alpha $beta $gamma $density $eps $classes test26.$i
#
#    # SL Sparse 0.5 w/ 3 classes
#    export density=0.5 gamma=3 classes=3
#    export numsamples=1000 eps=5
#    python3 swarm_mnist.py $numnodes 2 $numsamples $alpha $beta $gamma $density $eps $classes test27.$i
#    export numsamples=100 eps=10
#    python3 swarm_mnist.py $numnodes 2 $numsamples $alpha $beta $gamma $density $eps $classes test28.$i
#    export numsamples=25 eps=20
#    python3 swarm_mnist.py $numnodes 2 $numsamples $alpha $beta $gamma $density $eps $classes test29.$i
#
#    # SL Sparse 0.25 w/ 3 classes
#    export density=0.25 gamma=2 classes=3
#    export numsamples=1000 eps=5
#    python3 swarm_mnist.py $numnodes 2 $numsamples $alpha $beta $gamma $density $eps $classes test30.$i
#    export numsamples=100 eps=10
#    python3 swarm_mnist.py $numnodes 2 $numsamples $alpha $beta $gamma $density $eps $classes test31.$i
#    export numsamples=25 eps=20
#    python3 swarm_mnist.py $numnodes 2 $numsamples $alpha $beta $gamma $density $eps $classes test32.$i
#
#    # SL Sparse 0.0 w/ 3 classes
#    export density=0.0 gamma=1 classes=3
#    export numsamples=1000 eps=5
#    python3 swarm_mnist.py $numnodes 2 $numsamples $alpha $beta $gamma $density $eps $classes test33.$i
#    export numsamples=100 eps=10
#    python3 swarm_mnist.py $numnodes 2 $numsamples $alpha $beta $gamma $density $eps $classes test34.$i
#    export numsamples=25 eps=20
#    python3 swarm_mnist.py $numnodes 2 $numsamples $alpha $beta $gamma $density $eps $classes test35.$i

    # Below are the experiments for the conference. They were not used in my thesis.
    # FL Dense
    export classes=10
    export numsamples=1000 eps=5
    python3 fed_mnist.py $numnodes $numsamples $eps $classes test36.$i
    export numsamples=100 eps=10
    python3 fed_mnist.py $numnodes $numsamples $eps $classes test37.$i
    export numsamples=25 eps=20
    python3 fed_mnist.py $numnodes $numsamples $eps $classes test38.$i

#    # FL Sparse ~0.75 (7)
#    dumnodes=7
#    export classes=10
#    export numsamples=1000 eps=5
#    python3 fed_mnist.py $dumnodes $numsamples $eps $classes test39.$i
#    export numsamples=100 eps=10
#    python3 fed_mnist.py $dumnodes $numsamples $eps $classes test40.$i
#    export numsamples=25 eps=20
#    python3 fed_mnist.py $dumnodes $numsamples $eps $classes test41.$i
#
#    # FL Sparse ~0.5 (5)
#    dumnodes=5
#    export classes=10
#    export numsamples=1000 eps=5
#    python3 fed_mnist.py $dumnodes $numsamples $eps $classes test42.$i
#    export numsamples=100 eps=10
#    python3 fed_mnist.py $dumnodes $numsamples $eps $classes test43.$i
#    export numsamples=25 eps=20
#    python3 fed_mnist.py $dumnodes $numsamples $eps $classes test44.$i

    # FL Sparse ~0.25 (4)
    dumnodes=4
    export classes=10
    export numsamples=1000 eps=5
    python3 fed_mnist.py $dumnodes $numsamples $eps $classes test45.$i
    export numsamples=100 eps=10
    python3 fed_mnist.py $dumnodes $numsamples $eps $classes test46.$i
    export numsamples=25 eps=20
    python3 fed_mnist.py $dumnodes $numsamples $eps $classes test47.$i

    # FL Sparse ~0 (2)
    dumnodes=2
    export classes=10
    export numsamples=1000 eps=5
    python3 fed_mnist.py $dumnodes $numsamples $eps $classes test48.$i
    export numsamples=100 eps=10
    python3 fed_mnist.py $dumnodes $numsamples $eps $classes test49.$i
    export numsamples=25 eps=20
    python3 fed_mnist.py $dumnodes $numsamples $eps $classes test50.$i
done