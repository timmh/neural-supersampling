#!/usr/bin/env bash

export $(cat .env | xargs)

docker run \
    --rm \
    --gpus all \
    -v $PWD:/opt/neural-supersampling \
    -w /opt/neural-supersampling \
    "nytimes/blender:${BLENDER_VERSION}-gpu-ubuntu18.04" "$@"