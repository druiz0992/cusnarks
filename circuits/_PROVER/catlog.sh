#!/bin/bash

LOGFILE="$(ls -lrt log_* | tail -n 1 | rev | cut -d' ' -f1 | rev)"
cat ${LOGFILE}
