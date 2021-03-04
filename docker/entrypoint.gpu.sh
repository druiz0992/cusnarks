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

# Fix some issues
mv /usr/src/app/cusnarksdata/auxdata/runtime /usr/src/app/cusnarksdata/
mv /usr/src/app/cusnarksdata/auxdata/circuits /usr/src/app/cusnarksdata/
rmdir /usr/src/app/cusnarksdata/auxdata

#Fix parsing issue with nvgpu library
/bin/sed -i "s#if cuda_version <.*#if cuda_version < 12:#g" /usr/local/lib/python3.8/dist-packages/nvgpu/__init__.py
# Starts Go-Cusnarks server
cd /usr/src/app/go-cusnarks

#Generate Plugins
files=$(ls plugin/*.go)
for i in $files; do
	f=$(basename -- ${i%.go})
        go build -buildmode=plugin -o plugin/${f}.so plugin/${f}.go
done


#Update server_config.yalm file
if [ ! -z $SERVICE_IPPORT ]; then
  sed -i "/serviceapi/c\  serviceapi: ${SERVICE_IPPORT}" server_config.yaml
fi

if [ ! -z $ADMIN_IPPORT ]; then
  sed -i "/adminapi/c\  adminapi: ${ADMIN_IPPORT}" server_config.yaml
fi

if [ ! -z $DEBUG_EN ]; then
  sed -i "/debug/c\debug: ${DEBUG_EN}" server_config.yaml
fi

if [ ! -z $SEED ]; then
  sed -i "/seed/c\  seed : ${SEED}" server_config.yaml
fi

#Start server
go run . --config ./server_config.yaml start
