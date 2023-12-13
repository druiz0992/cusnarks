#!/bin/sh

# Launch cusnarks server and client and collects stats
#
# Args : 
#  
#  $1 : Witness File
#  $2 : Proof File
#  $3 : Public Data File

set -ex

SCRIPT=$(readlink -f "$0")
CUSNARKS_HOME=$(dirname "$SCRIPT")
DBPATH="/tmp"
TABLE="cusnarks_table"

runpsql() {
    sudo -u postgres psql -c "${1}"
}

add_dbelement() {
    oldpwd="`pwd`"
    cd "${DBPATH}"
    runpsql "INSERT INTO ${TABLE} 
             (id, result, prooft, evalt, ht, mexp1t, mexp2t) VALUES 
             (${1});"
    cd "${oldpwd}"
}


cd "${CUSNARKS_HOME}/src/python"
export LD_LIBRARY_PATH="${CUSNARKS_HOME}/lib:$LD_LIBRARY_PATH"

if [ -z $1] || [ "$#" -ne 3 ]; then
  WF="${CUSNARKS_HOME}/circuits/test_cusnarks_w.dat"
  PF="${CUSNARKS_HOME}/circuits/test_cusnarks_p.json"
  PDF="${CUSNARKS_HOME}/circuits/test_cusnarks_pd.json"
else
  WF="$1"
  PF="$2"
  PDF="$3"
fi

RESULTS="$(python3 pysnarks.py -m p -w ${WF} -p ${PF} -pd ${PDF} -v 1)"

PROOF_RESULT="$(echo $RESULTS | grep -m1 -oP '"result"\s*:\s*[0-9]' | cut -d ' ' -f 2)"
PROOF_ID="$(echo $RESULTS | grep -m1 -oP '"proof_id"\s*:\s*[0-9]+' | cut -d ' ' -f 2)"
PROOF_TIME="$(echo $RESULTS | grep -m1 -oP '"Proof"\s*:\s*[0-9]+.[0-9]+' | cut -d ' ' -f 2)"
EVAL_TIME="$(echo $RESULTS | grep -m1 -oP '"Eval"\s*:\s*\[\s*[0-9]+\.*[0-9]*' | cut -d ' ' -f 3)"
H_TIME="$(echo $RESULTS | grep -m1 -oP '"H"\s*:\s*\[\s*[0-9]+\.*[0-9]*' | cut -d ' ' -f 3)"
MEXP1_TIME="$(echo $RESULTS | grep -m1 -oP '"Mexp1"\s*:\s*\[\s*[0-9]+\.*[0-9]*' | cut -d ' ' -f 3)"
MEXP2_TIME="$(echo $RESULTS | grep -m1 -oP '"Mexp2"\s*:\s*\[\s*[0-9]+\.*[0-9]*' | cut -d ' ' -f 3)"


#echo $PROOF_RESULT
#echo $PROOD_ID
#echo $PROOF_TIME
#echo $EVAL_TIME
#echo $H_TIME
#echo $MEXP1_TIME
#echo $MEXP2_TIME

add_dbelement "${PROOF_ID}, ${PROOF_RESULT}, ${PROOF_TIME}, ${EVAL_TIME}, ${H_TIME}, ${MEXP1_TIME}, ${MEXP2_TIME}"
