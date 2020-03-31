#!/bin/sh

# Launch cusnarks server 
#
# Args : 
#  
#  $1 : Proving key file  
#  $2 : Verification key file

set -ex

SCRIPT=$(readlink -f "$0")
CUSNARKS_HOME=$(dirname "$SCRIPT")
DBPATH="/tmp"
SERVER="cusnarks"
TABLE="cusnarks_table"

cd "${CUSNARKS_HOME}/src/python"
export LD_LIBRARY_PATH="${CUSNARKS_HOME}/lib:$LD_LIBRARY_PATH"

if [ -z $1] || [ "$#" -ne 2 ]; then
  PKF="${CUSNARKS_HOME}/circuits/test_cusnarks_pk.bin"
  VKF="${CUSNARKS_HOME}/circuits/test_cusnarks_vk.json"
else
  PKF="$1"
  VKF="$2"
fi

runpsql() {
    sudo -u postgres psql -c "${1}"
}

clean_postgres() {
    oldpwd="`pwd`"
    cd "${DBPATH}"
    runpsql "DROP TABLE IF EXISTS ${TABLE};"
    runpsql "DROP DATABASE IF EXISTS ${SERVER};"
    runpsql "DROP ROLE IF EXISTS ${SERVER};"
    cd "${oldpwd}"
}

init_postgres() {
    oldpwd="`pwd`"
    cd "${DBPATH}"
    runpsql "CREATE ROLE ${SERVER};"
    runpsql "CREATE DATABASE ${SERVER} OWNER ${SERVER};"
    cd "${oldpwd}"
}

create_table() {
    oldpwd="`pwd`"
    cd "${DBPATH}"
    runpsql "CREATE TABLE ${TABLE} (
          id int PRIMARY KEY,
          result int NOT NULL,
          prooft float NOT NULL,
          evalt  float NOT NULL,
          ht     float NOT NULL,
          mexp1t float NOT NULL,
          mexp2t float NOT NULL );"
    cd "${oldpwd}"
}


clean_postgres
init_postgres
create_table

SEED=$(od -A n -t d -N 1 /dev/urandom)
python3 pysnarks.py -m p -pk ${PKF} -vk ${VKF}  -v 1 -seed ${SEED} -server 1 

