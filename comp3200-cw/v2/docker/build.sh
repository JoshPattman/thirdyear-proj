#!/bin/bash

cp -r ../code ./code
docker build -t comp3200image .
rm -r -d ./code

docker save -o ./bin/comp3200.tar custom-jupyter-notebook

scp -r ./bin iridis:~/comp3200/bin