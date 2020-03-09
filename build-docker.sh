#!/bin/bash

docker build -t tracking:latest .

# DATE=$( date +%N )
docker run --name tracking --shm-size=5G -it --rm \
	--device /dev/nvidia2:/dev/nvidia2	\
	-p 8888:8888 -p 6006:6006 \
	-e DISPLAY=${DISPLAY} \
	--mount type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix \
    --mount type=bind,src=/home/$(whoami)/repositories,dst=/tf \
    -w /tf \
    tracking:latest
