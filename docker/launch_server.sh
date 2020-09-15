#!/bin/bash 

export CIRCUITS_PATH=/usr/src/app/circuits
R1CS1_FILE=${CIRCUITS_PATH}/$1
PK_FILE=${CIRCUITS_PATH}/$2
VK_FILE=${CIRCUITS_PATH}/$3

cd /usr/src/app/cusnarks/src/python

if [[ ! -f ${PK_FILE} ]]; then
     echo "${PK_FILE} should exist" >> /dev/stderr
     exit 1
fi
  
if [[ ${PK_FILE} == *.bin && ! -f ${VK_FILE} ]]; then
     echo "${VK_FILE} should exist" >> /dev/stderr
     exit 1
fi

if [[ ${R1CS1_FILE} != 0 && ${PK_FILE} == *.bin ]]; then
   # Launch Trusted Setup
   echo "Launching trusted setup"
   python3 pysnarks.py -m s -in_c $R1CS1_FILE -pk $PK_FILE -vk $VK_FILE -v 0
fi

echo "Starting cusnark in server mode..."
if [[ ${PK_FILE} == *.bin ]]; then
   python3 pysnarks.py -m p -pk $PK_FILE -vk $VK_FILE 
else
   python3 pysnarks.py -m p -pk $PK_FILE 
fi
