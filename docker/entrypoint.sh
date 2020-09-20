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

#Start server
go run . --config ./server_config.yaml start
