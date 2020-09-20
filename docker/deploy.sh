#!/bin/bash 

# Deploy docker

# Params
# $1 : Number of roots ( 20 <= NROOTS <= 28)
# $2 : Bind volume where circuits can be found

if [ "$#" -ne 2 ]; then
   echo "Usage: ./deploy.sh <NROOTS> <path to local directory that mapps into cusnarksdata in docker>"
   exit
fi

service_ip=$(cat server_config.yaml | grep serviceapi  | awk '{split($0,a,":"); print a[2] }')
service_port=$(cat server_config.yaml | grep serviceapi  | awk '{split($0,a,":"); print a[3] }')
admin_ip=$(cat server_config.yaml | grep adminapi  | awk '{split($0,a,":"); print a[2] }')
admin_port=$(cat server_config.yaml | grep adminapi  | awk '{split($0,a,":"); print a[3] }')

docker run  -e CUSNARKS_NROOTS=$1 --gpus all --rm  -d -ti -p ${service_port}:${service_port} -p ${admin_port}:${admin_port} --name=docker_cusnarks -v $2:/usr/src/app/cusnarksdata davidrz/cusnarks_adx:0.1
