#!/bin/bash

# Creates cusnarks docker

# TODO : as input params, i should pass :
# 1 : Recompute docker struct -> 1
# 2 : Number of roots
# 3 : Test/Hermez circuits (.zkey, witness.cpp, witness.dat...)

if [[ $1 = 1 ]]; then
  ./clean.sh

  #cusnarkspath=$(cat server_config.yaml  | grep cusnarkspath | awk -F' ' '{print $(NF)}' | awk -F'"' '{print $2}')
  #DOCKDIRR=$(echo "${cusnarkspath%/*}")

  #circompath=$(cat server_config.yaml  | grep circompath | awk -F' ' '{print $(NF)}' | awk -F'/' '{print $(NF)}' | awk -F'"' '{print $1}')
  #inputsfile=$(cat server_config.yaml  | grep inputsfile | awk -F' ' '{print $(NF)}' | awk -F'"' '{print $2}')

  mkdir cusnarks
  mkdir -p cusnarks/circom_runtime
  mkdir cusnarks/circuits
  mkdir cusnarks/circuits/_PROVER
  mkdir third_party_libs
  mkdir circuits
  
  cd ../
  make clean 
  cp -R ../cusnarks/src ./docker/cusnarks
  cp -R ../cusnarks/config ./docker/cusnarks
  cp ../cusnarks/Makefile ./docker/cusnarks/
  rm -rf ./docker/cusnarks/__pycache__

  cp ../cusnarks/test/python/aux_data/calcwit.* ./docker/cusnarks/circom_runtime
  cp ../cusnarks/test/python/aux_data/circom.hpp ./docker/cusnarks/circom_runtime
  cp ../cusnarks/test/python/aux_data/utils.* ./docker/cusnarks/circom_runtime
  cp ../cusnarks/test/python/aux_data/fr.* ./docker/cusnarks/circom_runtime
  cp ../cusnarks/test/python/aux_data/main.cpp ./docker/cusnarks/circom_runtime
  cp ../cusnarks/test/python/aux_data/fail.cpp ./docker/cusnarks/circom_runtime
  cp ../cusnarks/test/python/aux_data/test_circom_input.json ./docker/cusnarks/circom_runtime
  cp ../cusnarks/src/cuda/fr.casm ./docker/cusnarks/circom_runtime

  cp ../cusnarks/test/python/aux_data/test_circom.cpp ./docker/cusnarks/circom_runtime
  cp ../cusnarks/test/python/aux_data/witness_calc.dat ./docker/cusnarks/circom_runtime
  #cp -R ../cusnarks/third_party_libs/snarkjs ./docker/third_party_libs
  #cp -R ../cusnarks/third_party_libs/ffiasm ./docker/third_party_libs
  #cp -R ../cusnarks/third_party_libs/pcg-cpp ./docker/third_party_libs
  
  cp ../cusnarks/docker/auxdata/test_circom_c.r1cs ./docker/circuits
  cp ../cusnarks/docker/auxdata/test_circom_pk.zkey ./docker/circuits
  cp ../cusnarks/docker/auxdata/test_circom_w.wtns ./docker/circuits
  cp ../cusnarks/docker/auxdata/test_circom_w.wshm ./docker/circuits
  cp ../cusnarks/docker/auxdata/test_circom_pk.bin  ./docker/circuits
  cp ../cusnarks/docker/auxdata/test_circom_vk.json  ./docker/circuits

  cp ../cusnarks/circuits/_PROVER/taillog.sh  ./docker/cusnarks/circuits/_PROVER
  cp ../cusnarks/circuits/_PROVER/catlog.sh  ./docker/cusnarks/circuits/_PROVER

  cd docker

  git clone https://github.com/iden3/go-cusnarks.git
fi

sudo docker build --build-arg NVARS=$2 -t cusnarks:1.0 .
