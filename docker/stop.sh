
#!/bin/bash 

# Stops Go-Cusnarks server

cd /usr/src/app/go-cusnarks

go run . --config ./server_config.yaml stop
