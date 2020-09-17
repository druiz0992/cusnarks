#!/bin/sh

CIRCOM=../../../third_party_libs/circom_runtime/c

nasm -felf64 ${CIRCOM}/fr.asm
g++ -std=c++11 ${CIRCOM}/calcwit.cpp ${CIRCOM}/main.cpp ${CIRCOM}/utils.cpp ${CIRCOM}/fr.cpp ${CIRCOM}/fr.o test_circom.cpp -I ${CIRCOM} -o ./wit_calc -lpthread -lgmp

shmem=$(ipcs -m  | grep 0x0001e240 )
if [ "$shmem" ]; then
	ipcrm -M 0x0001e240 
fi

mv test_circom.dat wit_calc.dat
./wit_calc test_circom_input.json test_circom_w.wtns
./wit_calc test_circom_input.json test_circom_w.wshm

mv circuit.r1cs ../../../circuits/test_circom_c.r1cs
mv test_circom_w.wtns ../../../circuits/test_circom_w.wtns
mv test_circom_w.wshm ../../../circuits/test_circom_w.wshm
mv circuit_final.zkey ../../../circuits/test_circom_pk.zkey

rm wit_calc 


