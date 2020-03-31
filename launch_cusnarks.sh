#!/bin/sh

# Launch cusnarks server and client and collects stats
#
# Args : 
#  
#  $1 : Proving key file  
#  $2 : Verification key file
#  $3 : Witness File
#  $4 : Proof File
#  $4 : Public Data File

set -ex

SESSION="cusnarks-demo"
SCRIPT=$(readlink -f "$0")
CUSNARKS_HOME=$(dirname "$SCRIPT")
DBPATH="/tmp"
CSVF="${DBPATH}/.cusnarks.dat"

if [ -z $1] || [ "$#" -ne 5 ]; then
  PKF="${CUSNARKS_HOME}/circuits/test_cusnarks_pk.bin"
  VKF="${CUSNARKS_HOME}/circuits/test_cusnarks_vk.json"
  WF="${CUSNARKS_HOME}/circuits/test_cusnarks_w.dat"
  PF="${CUSNARKS_HOME}/circuits/test_cusnarks_p.json"
  PDF="${CUSNARKS_HOME}/circuits/test_cusnarks_pd.json"
else
  PKF="$1"
  VKF="$2"
  WF="$3"
  PF="$4"
  PDF="$5"
fi


#[ -e ${CSVF} ] && rm -f ${CSVF}

cd "${CUSNARKS_HOME}"


tmux kill-session -t $SESSION || true

tmux new-session -d -s $SESSION
tmux split-window -d -t 0 -v
tmux split-window -d -t 0 -h
tmux split-window -d -t 2 -h

tmux send-keys -t 0 "./launch_server.sh ${PKF} ${VKF}" enter
tmux send-keys -t 1 "while true; do sleep 15; ./launch_client.sh ${WF} ${PF} ${PDF}; done" enter
tmux send-keys -t 2 "./launch_stats.sh" enter


tmux attach -t $SESSION

tmux kill-session -t $SESSION || true
