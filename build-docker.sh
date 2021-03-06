#!/bin/bash

docker build -t tracking:latest .

# DATE=$( date +%N )
docker run --name tracking -it --rm \
    --gpus '"device=1"' \
    -p 8888:8888 -p 6006:6006 \
    -e DISPLAY=${DISPLAY} \
    --mount type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix \
    --mount type=bind,src=/home/$(whoami)/repositories,dst=/tf \
    --mount type=bind,src=/home/$(whoami)/logs,dst=/tf/ADFQ/results \
    -w /tf \
    tracking:latest
