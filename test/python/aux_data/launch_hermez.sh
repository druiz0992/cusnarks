#!/bin/sh

CIRCOM=../../../third_party_libs/circom_runtime/c
HERMEZ=../../../docker/auxdata
g++ -std=c++11 ${CIRCOM}/calcwit.cpp ${CIRCOM}/main.cpp ${CIRCOM}/utils.cpp ${CIRCOM}/fr.cpp ${CIRCOM}/fr.o ${HERMEZ}/runtime/hermez.cpp -I ${CIRCOM} -o ./hermez_witbuild -lpthread -lgmp

shmem=$(ipcs -m  | grep 0x0001e240)
if [ "$shmem" ]; then
	ipcrm -M 0x0001e240 
fi

cd ${HERMEZ}/circuits
tar -xvzf hermez_pk.tgz
cd -

cp ${HERMEZ}/runtime/hermez_witbuild.dat .
./hermez_witbuild ${HERMEZ}/runtime/hermez_input.json hermez.wtns
./hermez_witbuild ${HERMEZ}/runtime/hermez_input.json hermez.wshm

cp ${HERMEZ}/circuits/hermez* ../../../circuits


rm hermez_witbuild* 
