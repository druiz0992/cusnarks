#!/bin/bash 

# Launches http server.

# Check preconditions
if [ ! $CUSNARKS_NROOTS ]; then
	CUSNARKS_NROOTS=28
fi

if [ ! $CUSNARKS_CURVE ]; then
	CUSNARKS_CURVE=BN256
fi

# Generate Roots
cd /usr/src/app/cusnarks/config
python3 cusnarks_config.py ${CUSNARKS_NROOTS} ${CUSNARKS_CURVE}

# Starts Go-Cusnarks server
cd /usr/src/app/go-cusnarks

#Generate Cuf Files
go run cmd/gencuf/main.go

#Update server_config.yalm file
if [ ! -z $SERVICE_IPPORT ]; then
  sed -i '/serviceapi/c\serviceapi:${SERVICE_IPPORT}' server_config.yaml
fi

if [ ! -z $ADMIN_IPPORT ]; then
  sed -i '/adminapi/c\adminapi:${ADMIN_IPPORT}' server_config.yaml
fi

if [ ! -z $DEBUG_EN ]; then
  sed -i '/debug/c\debug: ${DEBUG_EN}' server_config.yaml
fi

if [ ! -z $SEED ]; then
  sed -i '/seed/c\seed : ${SEED}' server_config.yaml
fi

#Start server
go run . --config ./server_config.yaml start
