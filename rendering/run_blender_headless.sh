#!/usr/bin/env bash

export $(cat .env | xargs)

docker run \
    --rm \
    --gpus all \
    -v $PWD:/content/neural-supersampling \
    -w /content/neural-supersampling \
    "nytimes/blender:${BLENDER_VERSION}-gpu-ubuntu18.04" "$@"