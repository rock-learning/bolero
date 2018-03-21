# Using BOLeRo via Docker

This directory contains a Dockerfile from which we create the docker image
for BOLeRo.

## Install Docker

For Ubuntu:

    sudo apt-get install docker.io

Installation instructions for other systems are available at the
[official documentation](https://docs.docker.com/engine/installation/).

## Build Image

Note: The average user does not have to build an image. Usually the image will
be distributed to the users.

    docker build -t bolero:latest .

Sometimes it is necessary to clean the docker cache if you want to rebuild the
image. You just have to add `--no-cache` in this case.

Now you could [push the image to Docker Hub](https://docs.docker.com/docker-cloud/builds/push-images/).

## Create Container

You can create a container (runtime environment) from the image. You can
either use your own image (bolero:latest) or you can use the prebuilt image
from Docker Hub. We made an image available at af01/bolero:latest.

Without GUI:

    docker run -it af01/bolero:latest

With GUI:

    xhost local:root
    docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix --privileged af01/bolero:latest

Additionally, you can mount external directories in the container. This can
be useful because you might want to destroy and recreate containers from time
to time. It can be done with the option

    -v <host-directory>:<mount-point>

You can give containers names that can be used like their IDs:

    --name <name>

## Setup GPU

You usually have to set up your GPU if you use the visualization of MARS. For
example, to make an Nvidia GPU available in the docker container, the following
steps have to be taken:

On the host, check your driver version:

    # you might have to install this package:
    $ sudo apt-get install mesa-utils
    $ glxinfo |grep "OpenGL version"
    OpenGL version string: 4.5.0 NVIDIA 375.39

In this example, the driver version is "375.39". Now, you have install exactly
the same driver in the container:

    export VERSION=375.39
    wget http://us.download.nvidia.com/XFree86/Linux-x86_64/$VERSION/NVIDIA-Linux-x86_64-$VERSION.run
    chmod +x NVIDIA-Linux-x86_64-$VERSION.run
    apt-fast install -y module-init-tools
    ./NVIDIA-Linux-x86_64-$VERSION.run -a -N --ui=none --no-kernel-module

## Working with Docker

Overview of containers:

    docker ps -a

Start container:

    docker start <id>

Connect to container:

    docker attach <id>

Typing the first 2-3 characters of the container ID is usually sufficient.

