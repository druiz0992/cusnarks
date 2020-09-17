#!/bin/sh

if [ -z $1 ]; then
   echo "Usage: ./build_key.sh <NROOTS>"
   exit
fi

NROOTS=$1

snarkjs powersoftau new bn128 $NROOTS pot${NROOTS}_0000.ptau 
snarkjs powersoftau contribute pot${NROOTS}_0000.ptau pot${NROOTS}_0001.ptau --name="First contribution"  -e="some random text"
snarkjs powersoftau contribute pot${NROOTS}_0001.ptau pot${NROOTS}_0002.ptau --name="Second contribution"  -e="some random text"
snarkjs powersoftau beacon pot${NROOTS}_0002.ptau pot${NROOTS}_beacon.ptau 0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f 10 -n="Final Beacon"
snarkjs powersoftau prepare phase2 pot${NROOTS}_beacon.ptau pot${NROOTS}_final.ptau 
snarkjs powersoftau verify pot${NROOTS}_final.ptau

cp test_circom.cir circuit.circom
circom circuit.circom --r1cs 
snarkjs zkey new circuit.r1cs pot${NROOTS}_final.ptau circuit_0000.zkey

snarkjs zkey contribute circuit_0000.zkey circuit_0001.zkey --name="1st Contributor Name"  -e="adad"
snarkjs zkey contribute circuit_0001.zkey circuit_0002.zkey --name="Second contribution Name"  -e="Another random entropy"
snarkjs zkey beacon circuit_0002.zkey circuit_final.zkey 0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f 10 -n="Final Beacon phase2"



circom circuit.circom -c test_circom.cpp
#delete intermediate files
rm *.ptau circuit_0*.zkey *.circom 
