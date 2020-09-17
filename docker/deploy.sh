#!/bin/bash 

if [ -z $1 ]; then
   echo "Usage: ./deploy.sh <path to local directory that mapps into cusnarksdata in docker>"
   exit
fi

sudo docker run  --gpus all --rm  -d -ti -p 3000:3000 -p 3001:3001 --name=cusnarks -v $1:/usr/src/app/cusnarksdata cusnarks:latest
