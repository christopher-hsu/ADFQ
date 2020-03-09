# Tensorflow gpu image as parent image
FROM tensorflow/tensorflow:1.14.0-gpu-py3-jupyter

# Set the working directory to /tf
WORKDIR /tf

# Copy the current directory contents into the container at /tf
COPY . /tf

# During building, skip package interactions
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y eog python3-dev python3-tk python3-yaml && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists

RUN apt-get update && \
    apt-get install -y zlib1g-dev libjpeg-dev cmake swig python-pyglet python3-opengl \
                libboost-all-dev libsdl2-dev libosmesa6-dev patchelf ffmpeg xvfb

# Install latest ray and any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Add other directories to PYTHONPATH
#RUN source setup

# Push GUI outisde container to local host
ENV QT_X11_NO_MITSHM=1

#RUN useradd -ms /bin/bash chrishsu

#USER chrishsu
