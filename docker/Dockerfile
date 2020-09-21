FROM nvidia/cuda:10.2-cudnn7-devel as build_stage

ENV LD_LIBRARY_PATH=/usr/src/app/cusnarks/lib:$LD_LIBRARY_PATH
ENV CUSNARKS_PATH /usr/src/app/cusnarks
ENV CUSNARKS_NROOTS 20
ENV CUSNARKS_CURVE BN256

WORKDIR /usr/src/app

RUN mkdir -p cusnarks
COPY cusnarks ./cusnarks

WORKDIR /usr/src/app/cusnarks

RUN apt-get update && \
    apt-get install -y git curl build-essential libssl-dev clang libomp-dev g++ nasm python3 python3-pip && \
    curl -sL https://deb.nodesource.com/setup_12.x | bash - && \
    apt-get -yq install nodejs && \
    npm i -g circom snarkjs && \
    pip3 install Cython numpy future nvgpu && \
    sh -s -- --default-toolchain stable -y && \
    make docker_all

FROM nvidia/cuda:10.2-base

ENV LD_LIBRARY_PATH=/usr/src/app/cusnarks/lib:$LD_LIBRARY_PATH
ENV CUSNARKS_PATH /usr/src/app/cusnarks
ENV CUSNARKS_NROOTS 20
ENV CUSNARKS_CURVE BN256


WORKDIR /usr/src/app

COPY --from=build_stage /usr/src/app /usr/src/app

RUN apt-get update && \
    apt-get install -y git curl build-essential libssl-dev clang libomp-dev g++ nasm python3 python3-pip sudo libgmp3-dev nlohmann-json-dev vim && \
    curl -O https://dl.google.com/go/go1.15.1.linux-amd64.tar.gz && \
    tar xvf go1.15.1.linux-amd64.tar.gz -C . && \
    mv ./go /usr/local && \
    ln -s /usr/local/go/bin/go /usr/local/bin/go && \
    curl -sL https://deb.nodesource.com/setup_14.x | bash - && \
    apt-get -yq install nodejs && \
    npm i -g circom@0.5.23 snarkjs@0.3.33 && \
    pip3 install Cython numpy future nvgpu 

RUN rm -rf go1.15.1.linux-amd64.tar.gz

COPY entrypoint.sh /usr/local/bin/entrypoint
COPY stop.sh /usr/local/bin/stop

RUN mkdir -p go-cusnarks
COPY go-cusnarks ./go-cusnarks

WORKDIR /usr/src/app/go-cusnarks

RUN go build -o gocusnarks .

COPY server_config.yaml .

WORKDIR /usr/src/app
RUN rm -rf ./cusnarks/src/cuda
RUN rm -rf ./cusnarks/src/cython

ENTRYPOINT ["entrypoint"]
