#!/bin/bash

# Swarm with varying sync rates
echo "1/20"
python swarm_mnist.py 9000 10 6000 1 1 0.2
echo "2/20"
python swarm_mnist.py 9000 10 6000 2 1 0.2
echo "3/20"
python swarm_mnist.py 9000 10 6000 3 1 0.2
echo "4/20"
python swarm_mnist.py 9000 10 6000 4 1 0.2

echo "5/20"
python swarm_mnist.py 9000 10 6000 1 1 0.4
echo "6/20"
python swarm_mnist.py 9000 10 6000 2 1 0.4
echo "7/20"
python swarm_mnist.py 9000 10 6000 3 1 0.4
echo "8/20"
python swarm_mnist.py 9000 10 6000 4 1 0.4

echo "9/20"
python swarm_mnist.py 9000 10 6000 1 1 0.6
echo "10/20"
python swarm_mnist.py 9000 10 6000 2 1 0.6
echo "11/20"
python swarm_mnist.py 9000 10 6000 3 1 0.6
echo "12/20"
python swarm_mnist.py 9000 10 6000 4 1 0.6

echo "13/20"
python swarm_mnist.py 9000 10 6000 1 1 0.8
echo "14/20"
python swarm_mnist.py 9000 10 6000 2 1 0.8
echo "15/20"
python swarm_mnist.py 9000 10 6000 3 1 0.8
echo "16/20"
python swarm_mnist.py 9000 10 6000 4 1 0.8

# Fed
echo "17/20"
#python fed_mnist.py 9000 20 60000 1 1
echo "18/20"
#python fed_mnist.py 9000 20 60000 2 1
echo "19/20"
#python fed_mnist.py 9000 20 60000 3 1
echo "20/20"
#python fed_mnist.py 9000 20 60000 4 1
echo "done"