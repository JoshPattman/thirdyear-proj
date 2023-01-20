#!/bin/bash


for numsamples in 60000 30000 10000 1000 100
do
    for epochs in 1 2 3 4
    do
        for id in 1 2 3 4
        do
            for sync in 0.9 0.7 0.5 0.3 0.1
            do
                echo -e "\n\n\n================================================== RUNNING WITH ID $id NUMSAMPLES $numsamples EPOCHS $epochs SYNC $sync =============="
                python mnist.py 9000 5 $numsamples $id $epochs $sync
            done
        done
    done
done
