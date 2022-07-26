#!/usr/bin/env bash

export $(cat .env | xargs)

XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

docker run \
    --rm \
    --gpus all \
    -it \
	-v $XSOCK:$XSOCK:rw \
	-v $XAUTH:$XAUTH:rw \
    --device=/dev/dri/card0:/dev/dri/card0 \
	-e DISPLAY=$DISPLAY \
	-e XAUTHORITY=$XAUTH \
    -v $PWD:/content/neural-supersampling \
    -w /content/neural-supersampling \
    "nytimes/blender:${BLENDER_VERSION}-gpu-ubuntu18.04" "$@"