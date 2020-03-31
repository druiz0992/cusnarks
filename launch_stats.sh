#!/bin/sh

#set -ex

SCRIPT=$(readlink -f "$0")
CUSNARKS_HOME=$(dirname "$SCRIPT")
DBPATH="/tmp"
TABLE="cusnarks_table"
CSVF="${DBPATH}/.cusnarks.dat"

runpsql() {
    sudo -u postgres psql -t -c "${1}"
}


db2csv() {
    oldpwd="`pwd`"
    cd "${DBPATH}"
    ELS="$(runpsql "COPY (SELECT ${1} FROM ${TABLE} ORDER BY id DESC LIMIT ${2}) to '${CSVF}' DELIMITER ' ' CSV HEADER";)"
    cd "${oldpwd}"
}

get_dbNels() {
    oldpwd="`pwd`"
    cd "${DBPATH}"
    N_ELS="$(runpsql "SELECT COUNT(*) FROM ${TABLE}";)"
    cd "${oldpwd}"
    echo "$N_ELS"
}

if  command -v gnuplot >/dev/null 2>&1; then
  :
else
   sudo apt-get install gnuplot
fi

INITF=0
while true; do
  N_ELS=$(get_dbNels)
  if [ "$N_ELS" -gt 1 ]; then
    db2csv "*" "10"
    if [ "$INITF" -eq 0 ]; then
      INITF=1
      gnuplot liveplot.gnu&
    fi
  fi
  sleep 60
done
