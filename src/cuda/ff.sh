#!/bin/bash

set -x

add_correct_labels() {
  FILE=$1
  EXT=${FILE:1:1}
  
  ADX_SUPPORT=$(cat /proc/cpuinfo | grep -m1 -oP 'adx')

  if [ -z "$ADX_SUPPORT" ]; then
     sed -i "1s;^;\n\n\n\n        global F${EXT}_rawAdd\n        global F${EXT}_rawSub\n        global F${EXT}_rawMMul\n        global F${EXT}_rawMSquare\n;" "$FILE";
  
     sed -i "s/\brawAddLL\b/F${EXT}_rawAdd/g" "$FILE";
     sed -i "s/\brawSubLL\b/F${EXT}_rawSub/g" "$FILE";
     sed -i "s/\brawMontgomeryMul\b/F${EXT}_rawMMul/g" "$FILE";
     sed -i "s/\brawMontgomerySquare\b/F${EXT}_rawMSquare/g" "$FILE";
  fi
  

  sed -i "0,/F${EXT}_fail/s//fail_h/" "$FILE";
  sed -i "s/F${EXT}_fail/fail_h wrt ..plt/1" "$FILE";
}


cd ../../

if [ ! -d "third_party_libs" ]; then
  echo "Directory third_party_libs doesnt exist. Exiting...";
  exit 1
fi

cd third_party_libs

if [ ! -d "ffiasm" ]; then
  echo "clone ffiasm repo"
  git clone https://github.com/iden3/ffiasm.git
fi

cd ffiasm;

ADX_SUPPORT=$(cat /proc/cpuinfo | grep -m1 -oP 'adx')
echo ${ADX_SUPPORT}
if [ -z "$ADX_SUPPORT" ]; then
    echo "system does not support adx"
    git checkout cdabe2242a9bd1dc61285fec918f9e7cd610d7ef;
fi

npm i;
mkdir -p tmp;
cd tmp;

if [ "$1" == "BN256" ] || [ "$1" == "" ]; then
   echo "Generating BN256 files";
   ../src/buildzqfield.js -q 21888242871839275222246405745257275088548364400416034343698204186575808495617 -n Fr;
   ../src/buildzqfield.js -q 21888242871839275222246405745257275088696311157297823662689037894645226208583 -n Fp;
   NWORDS_FR=8
   NWORDS_FP=8

elif  [ "$1" == "BLS12381" ]; then
   echo "Generating BLS12-381 files";
   ../src/buildzqfield.js -q 52435875175126190479447740508185965837690552500527637822603658699938581184513 -n Fr;
   ../src/buildzqfield.js -q 4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787
   
   NWORDS_FR=8
   NWORDS_FP=12

else 
   echo "Curve not valid";
   echo "usage : ./ff.sh <BN256 | BLS12381>"
   exit 1;
fi

add_correct_labels "fr.asm"
add_correct_labels "fp.asm"


echo "#ifndef __FF_H_" > _ff.h
echo "#define __FF_H_" >> _ff.h
echo "#define NWORDS_FR  (${NWORDS_FR})" >> _ff.h
echo "#define NWORDS_FP  (${NWORDS_FP})" >> _ff.h
echo "#endif" >> _ff.h

echo "NWORDS_FR = ${NWORDS_FR}" > _ff.py
echo "NWORDS_FP = ${NWORDS_FP}" >> _ff.py

mv fr.asm ../../../src/cuda/fr.casm
mv fp.asm ../../../src/cuda/fp.casm

mv _ff.h ../../../src/cuda
mv _ff.py ../../../src/python

