#!/bin/bash

# script to start a notebook on a remote host in a docker container

# set the name of the container
CONTAINER_NAME="notebook"

# enter name of image to run
IMAGE_NAME=$1

REMOTE_REPO="diffwave_docker"

# set the port to use
PORT=8888

# Get user ssh password
read -s -p "Enter remote password: " PASSWORD

# ssh into remote and print ls to terminal
# requires install of sshpass
sshpass -p "$PASSWORD" ssh -T gpubox <<EOL
    cd $REMOTE_REPO

    echo "Starting docker container..."
    CONTAINER_ID=$(docker run -d -t -p $PORT:$PORT --name $CONTAINER_NAME --rm $IMAGE_NAME)
    echo "Container started with ID: $CONTAINER_ID"

    docker exec -it $CONTAINER_ID sh -c "jupyter-notebook --no-browser --ip=0.0.0.0 --allow-root"

EOL

#echo "Building docker container..."
#IMAGE_ID=$(docker build -q -t $IMAGE_NAME .)
#IMAGE_ID=${IMAGE_ID#*:} # remove sha256: from image id
# CONTAINER_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $CONTAINER_ID)