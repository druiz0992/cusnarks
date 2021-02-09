#!/bin/bash

# Creates cusnarks docker

# Params
# $1 : Dockerfiler
# $2 : Push to AWS -> 1, push to Docker -> 2, Push to both 3
# $2 : Recompute docker struct -> 1

dockerF=$1
extension="${dockerF##*.}"
pushDocker=$2
echo $extension
if [[ $3 = 1 ]]; then
  ./clean.sh

  mkdir -p cusnarks/circuits
  mkdir -p cusnarks/circuits/_PROVER
  
  cd ../
  make clean 
  cp -R ../cusnarks/src ./docker/cusnarks
  cp -R ../cusnarks/config ./docker/cusnarks
  if [[ "$extension" == *"debug"* ]]; then
     cp -R ../cusnarks/test ./docker/cusnarks
  fi
  cp ../cusnarks/Makefile ./docker/cusnarks/
  rm -rf ./docker/cusnarks/__pycache__

  cp ../cusnarks/third_party_libs/circom_runtime/c/*.{cpp,hpp,asm}  ./docker/auxdata/runtime

  cp ../cusnarks/circuits/_PROVER/taillog.sh  ./docker/cusnarks/circuits/_PROVER
  cp ../cusnarks/circuits/_PROVER/catlog.sh  ./docker/cusnarks/circuits/_PROVER

  cp ./docker/server_config.tpl  ./docker/server_config.yaml

  cd docker

  git clone https://github.com/iden3/go-cusnarks.git
fi

GIT_COMMIT=$( git rev-parse HEAD)
docker build -f $dockerF . -t hermeznet/cusnarks_$extension:latest -t hermeznet/cusnarks_$extension:${GIT_COMMIT} -t 811278125247.dkr.ecr.eu-west-3.amazonaws.com/proofserver-${extension}:integration -t 811278125247.dkr.ecr.eu-west-3.amazonaws.com/proofserver-${extension}:integration-${GIT_COMMIT} 

exitStatus=$?
if [ $exitStatus -ne 0 ]; then
   echo "Error building docker"
   exit 1
fi

if [[ ${pushDocker} = 1 ]] || [[ ${pushDocker} = 3 ]]; then
   export AWS_REGION=eu-west-3
   aws ecr get-login-password --region eu-west-3 | docker login --username AWS --password-stdin 811278125247.dkr.ecr.eu-west-3.amazonaws.com
   docker push 811278125247.dkr.ecr.eu-west-3.amazonaws.com/proofserver-${extension}:integration
   docker push 811278125247.dkr.ecr.eu-west-3.amazonaws.com/proofserver-${extension}:integration-${GIT_COMMIT}
fi

if [[ ${pushDocker} = 2 ]] || [[ ${pushDocker} = 3 ]]; then
   docker login
   docker push hermeznet/cusnarks_$extension:latest
   docker push hermeznet/cusnarks_$extension:${GIT_COMMIT}
fi
