#!/bin/bash

# Creates cusnarks docker

# Params
# $1 : Recompute docker struct -> 1

if [[ $1 = 1 ]]; then
  ./clean.sh

  mkdir -p cusnarks/circuits
  mkdir -p cusnarks/circuits/_PROVER
  
  cd ../
  make clean 
  cp -R ../cusnarks/src ./docker/cusnarks
  cp -R ../cusnarks/config ./docker/cusnarks
  cp ../cusnarks/Makefile ./docker/cusnarks/
  rm -rf ./docker/cusnarks/__pycache__

  cp ../cusnarks/third_party_libs/circom_runtime/c/*.{cpp,hpp,asm}  ./docker/auxdata/runtime

  cp ../cusnarks/circuits/_PROVER/taillog.sh  ./docker/cusnarks/circuits/_PROVER
  cp ../cusnarks/circuits/_PROVER/catlog.sh  ./docker/cusnarks/circuits/_PROVER

  cp ./docker/server_config.tpl  ./docker/server_config.yaml

  cd docker

  git clone https://github.com/iden3/go-cusnarks.git
fi

docker build -t cusnarks .
