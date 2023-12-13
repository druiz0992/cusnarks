#!/bin/sh

# What do we install?
#    - g++ compiler 
#    - nasm 
#    - Python3.6+
#        - Cython 
#        - numpy 
#        - future 
#        - nvgpu 
#    - openmp 
#    - node 
#    - npm 
#    - rustup 
#    - clang 
#    - build-essential 
#    - libssl-dev 
# To force build dependencies, run ./cusnarks_build.sh force

set -ex


SCRIPT=$(readlink -f "$0")
CUSNARKS_HOME=$(dirname "$SCRIPT")

SEP="-------------------------------------------"
LOG=".build.log"
INIT_FILE=".init"
CUSNARKS_CONFIG_DIR="${CUSNARKS_HOME}/config"
CUSNARKS_CONFIG_F=".nroots"

cd "${CUSNARKS_HOME}"
[ -e ${LOG} ] && rm ${LOG}

####
arg1="$1"

if [ "$arg1" = "force" ]; then
  [ -e ${INIT_FILE} ] && rm ${INIT_FILE}
fi

if [ ! -f ${INIT_FILE} ]; then
  CC=g++
  ASM=nasm
  
  PREREQ="git ${CURL} build-essential libssl-dev clang libomp-dev"
  
  # Oter
  CURL=curl
  CURL_OPT="-sSf"
  CURL_INST="curl_inst"
  UPDATE="apt-get update"
  INSTALL="sudo apt-get install"
  
  
  # Javascript
  NODEJS=nodejs
  NPM=npm
  JS="${NODEJS} ${NPM}"
  JS_VER=10
  JS_ADDR="https://deb.nodesource.com/setup_${JS_VER}.x"
  
  #Rust
  RUST_ADDR="https://sh.rustup.rs"
  
  # Python
  PIP=pip3
  PY="python3 ${PIP}"
  PYLIB="Cython numpy future nvgpu"
  PIP_INSTALL="${PIP} install"
  
  ## Start
  
  #${UPDATE} > /dev/null 2>&1
  
  EXEC=" ${PREREQ} ${CC} ${ASM} ${PY}"
  for prog in ${EXEC}
  do
      if command -v ${prog} >/dev/null 2>&1 ; then
           echo "${prog} Installed....." >> ${LOG};
           ${prog} --version >> ${LOG};
      else
          ${INSTALL} ${prog} >> ${LOG};
      fi
      echo ${SEP} >> ${LOG};
  done
  
  #Python
  ${PIP_INSTALL} ${PYLIB} >> ${LOG} 
  
  #Rust
  if [ ! -d ${HOME}/.cargo]; then
    ${CURL}  ${RUST_ADDR} ${CURL_OPT} > ${CURL_INST}
    [ -e ${CURL_INST} ] && chmod 777 ${CURL_INST} && ./${CURL_INST}
    [ -e ${CURL_INST} ] && rm ${CURL_INST}
  fi
  . ${HOME}/.cargo/env
  
  #JS
  if command -v ${NODEJS} >/dev/null 2>&1; then
           echo "${NODEJS} Installed....." >> ${LOG};
           ${NODEJS} --version >> ${LOG};
      else
          ${CURL} ${JS_ADDR} ${CURL_OPT} > ${CURL_INST}
          [ -e ${CURL_INST} ] && chmod 777 ${CURL_INST} && ./${CURL_INST}
          ${INSTALL} ${NODEJS} >> ${LOG};
          [ -e ${CURL_INST} ] && rm ${CURL_INST}
  fi
  echo 1 > ${INIT_FILE}
fi
     
#MAKE CUSNARKS
make third_party_libs
make build

#CONFIGURE CUSNARKS
cd ${CUSNARKS_CONFIG_DIR}
if [ "$arg1" = "force" ]; then
  [ -e ${CUSNARKS_CONFIG_F} ] && rm ${CUSNARKS_CONFIG_F}
fi
export NROOTS=20
if [ ! -f ${CUSNARKS_CONFIG_F} ]; then
  cd ${CUSNARKS_HOME};
  echo "Building Roots"
  export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH; make config
fi
cd ${CUSNARKS_HOME};



  
