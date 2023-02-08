#!/bin/bash

i=0

for numsamples in 60000 30000 10000 1000
do
    for epochs in 1 2 3
    do
        for id in 1 2 3 4
        do
            percent=$(python -c "print(100*$i/240)")
            echo -e "\n\n\n $percent%\n================================================== RUNNING WITH ID $id NUMSAMPLES $numsamples EPOCHS $epochs =============="
            python fed_mnist.py 9000 5 $numsamples $id $epochs
            i=$((i+1))
        done
    done
done
