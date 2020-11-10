#!/bin/bash

set -x

add_correct_labels() {
  FILE=$1
  EXT=${FILE:1:1}
  
  ADX_SUPPORT=$(cat /proc/cpuinfo | grep -m1 -oP 'adx')

  sed -i "1s;^;\n\n\n\n        global F${EXT}_rawAdd\n        global F${EXT}_rawSub\n        global F${EXT}_rawMMul\n        global F${EXT}_rawMSquare\n;" "$FILE";
  
  if [ -z $ADX_SUPPORT ]; then 
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

if [ ! -d "circom_runtime" ]; then
   git clone https://github.com/iden3/circom_runtime.git
fi

cd ffiasm;

ADX_SUPPORT=$(cat /proc/cpuinfo | grep -m1 -oP 'adx')
echo ${ADX_SUPPORT}
if [ -z $ADX_SUPPORT ]; then 
    echo "system does not support adx"
    git checkout cdabe2242a9bd1dc61285fec918f9e7cd610d7ef;
else
    echo "system does support adx"
    #git checkout af581d6f0f78d8ac9256b556512f69501d59c368;
fi

npm i;
mkdir -p tmp;
cd tmp;

if [ "$1" == "BN256" ] || [ "$1" == "" ]; then
   echo "Generating BN256 files";
   #E(Fq) := y^2 = x^3 + 3
   #twisted curve ofver FQ**2, b = FQ2([3,0])/FQ2([9,1])
   #E'(Fq2) := y^2 = x^3 + 3(i + 1)
   FR=21888242871839275222246405745257275088548364400416034343698204186575808495617 
   FP=21888242871839275222246405745257275088696311157297823662689037894645226208583
   A=0
   B=3
   G1X=1
   G1Y=2
   G2X1=10857046999023057135944570762232829481370756359578518086990519993285655852781
   G2X2=11559732032986387107991004021392285783925812861821192530917403151452391805634
   G2Y1=8495653923123431417604973247489272438418190587263600148770280649306958101930
   G2Y2=4082367875863433681332203403145435568316851327593401208105741076214120093531
   FD="{'factor_data': {'factors': [2, 3, 13, 29, 983, 11003, 237073, 405928799, 1670836401704629, 13818364434197438864469338081], 'exponents': [28, 2, 1, 1, 1, 1, 1, 1, 1, 1]}}"

   NWORDS_FR=8
   NWORDS_FP=8
   DEF="_BN256"
   UNDEF="_BLS12381"

elif  [ "$1" == "BLS12381" ]; then
   #E(Fq) := y^2 = x^3 + 4
   #Fq2 := Fq[i]/(x^2 + 1)
   #E'(Fq2) := y^2 = x^3 + 4(i + 1)

   echo "Generating BLS12-381 files";
   FR=52435875175126190479447740508185965837690552500527637822603658699938581184513
   FP=4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787
   A=0
   B=4
   G1X=3685416753713387016781088315183077757961620795782546409894578378688607592378376318836054947676345821548104185464507
   G1Y=1339506544944476473020471379941921221584933875938349620426543736416511423956333506472724655353366534992391756441569
   G2X1=352701069587466618187139116011060144890029952792775240219908644239793785735715026873347600343865175952761926303160
   G2X2=3059144344244213709971259814753781636986470325476647558659373206291635324768958432433509563104347017837885763365758
   G2Y1=1985150602287291935568054521177171638300868978215655730859378665066344726373823718423869104263333984641494340347905
   G2Y2=927553665492332455747201965776037880757740193453592970025027978793976877002675564980949289727957565575433344219582
   FD="{'factor_data': {'factors': [2, 3, 11, 19, 10177, 125527, 859267, 906349, 2508409, 2529403, 52437899, 254760293], 'exponents': [32, 1, 1, 1, 1, 1, 1, 2, 1,1,1,2]}}"

   NWORDS_FR=8
   NWORDS_FP=12
   UNDEF="_BN256"
   DEF="_BLS12381"

else 
   echo "Curve not valid";
   echo "usage : ./ff.sh <BN256 | BLS12381>"
   exit 1;
fi

cd ../../../config

python3 toFElement.py  ${FR} ${FP} ${G1X} ${G1Y} ${G2X1} ${G2X2} ${G2Y1} ${G2Y2} "${FD}"
cp ../src/cuda/constants.template ../src/cuda/constants.cpp

while read f; do
  LABEL=$(echo "$f" | cut -d' ' -f1)
  DATA=$(echo "$f" | cut -d' ' -f2-)
  if [ ${LABEL:1:2} == "FP" ]; then
    PERIOD=$NWORDS_FP
  else
    PERIOD=$NWORDS_FR
  fi
  PROC_DATA=$(echo $DATA | awk -v p="$PERIOD" '{for(i=1; i<=NF; i++) if(count++%p==0)  $i="TTTT"$i}1')
  sed -i'' "s#${LABEL}#${PROC_DATA}#g" "../src/cuda/constants.cpp"
done <test.dat
sed -i'' "s/TTTT/\n/g" "../src/cuda/constants.cpp"

echo "${DEF}" > .curve_tmp
cd -
../src/buildzqfield.js -q ${FR} -n Fr;
../src/buildzqfield.js -q ${FP} -n Fp;

if [ -z $ADX_SUPPORT ]; then 
  mv fr.c fr.cpp
  #mv fr.h fr.hpp
  cd ../../circom_runtime/c
  sed -i '/Fr_toLongNormal/c\Fr_toLongNormal(&v);' main.cpp 
  cd -
fi
cp fr.* ../../circom_runtime/c 

add_correct_labels "fr.asm"
add_correct_labels "fp.asm"

FP_ONE="1 "
FP_ZERO="0 "
for i in $(seq 2 $NWORDS_FP); do
  FP_ONE="${FP_ONE} , 0"
  FP_ZERO="${FP_ZERO} , 0"
done

FR_ONE="1 "
FR_ZERO="0 "
for i in $(seq 2 $NWORDS_FR); do
  FR_ONE="${FR_ONE} , 0"
  FR_ZERO="${FR_ZERO} , 0"
done

echo "#ifndef __FF_H_" > _ff.h
echo "#define __FF_H_" >> _ff.h
echo "#define NWORDS_FR  (${NWORDS_FR})" >> _ff.h
echo "#define NWORDS_FP  (${NWORDS_FP})" >> _ff.h
echo "#define FP_INIT_ARRONE(V) uint32_t V[]={${FP_ONE}}" >> _ff.h
echo "#define FP_INIT_ARRZERO(V) uint32_t V[]={${FP_ZERO}}" >> _ff.h
echo "#define FR_INIT_ARRONE(V) uint32_t V[]={${FR_ONE}}" >> _ff.h
echo "#define FR_INIT_ARRZERO(V) uint32_t V[]={${FR_ZERO}}" >> _ff.h
echo "#define ${DEF}" >> _ff.h
echo "#undef ${UNDEF}" >> _ff.h
     
echo "#endif" >> _ff.h

echo "NWORDS_FR = ${NWORDS_FR}" > _ff.py
echo "NWORDS_FP = ${NWORDS_FP}" >> _ff.py


mv fr.asm ../../../src/cuda/fr.casm
mv fp.asm ../../../src/cuda/fp.casm

mv _ff.h ../../../src/cuda
mv _ff.py ../../../src/python

