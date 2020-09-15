#!/bin/bash -x

export CIRCUITS_PATH=/usr/src/app/circuits
export NAME=$1
export R1CS1_FILE=${CIRCUITS_PATH}/${NAME}.r1cs
export PK_BIN_FILE=${CIRCUITS_PATH}/${NAME}_proving_key.bin
export VK_JSON_FILE=${CIRCUITS_PATH}/${NAME}_verification_key.json
export WITNESS_JSON_FILE=${CIRCUITS_PATH}/${NAME}_witness_cpp.json
export OUTPUT_PROOF_FILE=${CIRCUITS_PATH}/${NAME}_proof.json
export OUTPUT_PUBLIC_DATA_FILE=${CIRCUITS_PATH}/${NAME}_public.json

snarkjs verify --vk $VK_JSON_FILE --proof $OUTPUT_PROOF_FILE --public $OUTPUT_PUBLIC_DATA_FILE
