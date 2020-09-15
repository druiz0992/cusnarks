#!/bin/bash 

sudo docker run  --rm  -d -ti -p 3000:3000 -p 3001:3001 --name=docker_cusnarks4 -v /home/david/iden3/cusnarks/docker/circuits:/usr/src/app/circuits cusnarks:1.0
