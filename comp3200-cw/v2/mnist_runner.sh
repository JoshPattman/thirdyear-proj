#!/bin/bash

TODO ADD WEIGHTS


for numsamples in 60000 30000 10000 5000 1000 100 10
do
    for epochs in 1 2 3 4
    do
        for id in 1 2 3 4 5
        do
            echo -e "\n\n\n================================================== RUNNING WITH ID $id NUMSAMPLES $numsamples EPOCHS $epochs =============="
            python mnist.py 9000 5 $numsamples $id $epochs
        done
    done
done