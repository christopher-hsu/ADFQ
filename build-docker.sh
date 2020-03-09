#!/bin/bash

docker build -t tracking:latest .

# DATE=$( date +%N )
nvidia-docker run --name tracking --shm-size=5G -it --rm \
	-p 8888:8888 -p 6006:6006 \
	-e DISPLAY=${DISPLAY} \
	--mount type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix \
    --mount type=bind,src=/home/$(whoami)/repositories/tt,dst=/tf \
    -w /tf \
    tracking:latest