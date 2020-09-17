#!/bin/bash 

# Starts Go-Cusnarks server

cd /usr/src/app/go-cusnarks

#Generate Cuf Files
go run cmd/gencuf/main.go

#Start server
go run . --config ./server_config.yaml start

