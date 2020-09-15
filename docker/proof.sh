#!/bin/bash 

export CIRCUITS_PATH=/usr/src/app/circuits
WIT_FILE=${CIRCUITS_PATH}/$1
PROOF_FILE=${CIRCUITS_PATH}/$2
PDATA_FILE=${CIRCUITS_PATH}/$3
VERIFY=$4
NGPUS=$5

VERIFY_STRING=" -v 0 "
if [[ ${VERIFY} -eq 1 ]]; then
  VERIFY_STRING=" -v 1 "
fi

NGPUS_STRING=""
if [[ ! -z ${NGPUS} ]]; then
  $NGPUS_STRING=" -gpu ${NGPUS} "
fi

if [[ ! -f ${WIT_FILE} ]]; then
     echo "Witness File: ${WIT_FILE} should exist" >> /dev/stderr
     exit 1
fi

cd /usr/src/app/cusnarks/src/python
echo "python3 pysnarks.py -m p -w $WIT_FILE -p $PROOF_FILE -pd $PDATA_FILE $VERIFY_STRING $NGPUS_STRING"
python3 pysnarks.py -m p -w $WIT_FILE -p $PROOF_FILE -pd $PDATA_FILE $VERIFY_STRING $NGPUS_STRING
