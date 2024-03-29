# Cusnarks Overview
Cusnarks is an optimized CUDA implementation of ZK-SNARK setup and prover based on *Groth16* over curve [bn128][BN128].
It has been designed with the objective of computing proofs for up to 2^27 constraints in shortes time possible in a platform consisting of NxGPUs and MxCPU cores.
CUSNARKS is expected to work with [circom][] for the generation and compilation of circuits, and with [snarkjs][] for the computation witnesses, proof verification and parts of
 the trusted setup.  Additionally, cusnarks works with [circom2][], an optimized version of [circom][] that allows to compile and generate witnesses for very large circuits.


Cusnarks has been developed in C/C++/CUDA-C and Python. Python is the driving language where proof and 
setup scripts are launched.  Computation intensive functionality on the host (CPU) side has been written in C/C++.
 Computation intensive functionality on the the device (GPU) side has been writted in CUDA. Cython is used 
to build wrappers around C functions so that they can be called from Python and executed on CPU or GPU.
 An overview of the software architecture can be found [here](#architecture).

Elliptic curve scalar multiplication, the heaviest blocks in terms of 
computation requirements have been implemented to run on the GPU side. CPU is used during the conversion of Elliptic Curve points
from jacobian to affine cordinates and during QAP reduction in the prover and to perform polynomial multiplication (via FFT). 
In upcomming versions we will use CPUs as an additional
resource in the last multi-exponentiation stages as a way to reduce proof time.

The current partition of GPU vs. CPU functionality on the prover side is shown in below.

![proof_block_diagram](doc/block_diagram.png)

**NOTE:** Prover can be run entirely in CPU if no GPUs are available

The trusted setup is currently very slow. Implementation has been done using a single CPU core, except for the multi-exponentiation stage where we use a non-optimized algorithm for a single GPU.


## Outline
* [Installation][]
* [Launching Cusnarks](#Launching-Cusnarks)
* [Example][] 
* [Cusnarks Options][]
* [Architecture][]
* [Modules][]
* [Some Results][]
* [File Formats][]
* [Other Info](#Other-Info)
* [References][]
* [Next Steps][]
  
## Installation
1. Download repository www.github.com/iden3/cusnarks.git. From now on, the folder where cusnarks is downloaded will be called *$CUSNARKS_HOME*

2. Change to *$CUSNARKS_HOME* directory and execute cusnarks_build.sh (builds Cusnarks) and launch_cusnarks (launches a proof server and requests proofs) scripts
```sh
 cd $CUSNARKS_HOME
 ./cusnarks_build.sh
 ./launch_cusnarks.sh
```

Alternatively, you can install Cusnarks manually:

2. Ensure that all [dependencies][] are installed. 

Tested versions of required libraries and programs are provided below for reference.

    - Python3.6+
        - Cython (0.26.1)
        - numpy (1.17.1)
        - future (0.17.1)
        - nvgpu (0.8.0) : If GPU needed
    - g++ compiler (7.4.0)
    - nasm (2.13.02)
    - CUDA toolkit (10.1)
    - openmp (201511)
    - node (10.16.0)
    - npm (6.9.0)
    - rustup (1.20.2)
    - clang (6.0.0)
    - build-essential (12.4)
    - libssl-dev (1.1.1)

**NOTE:** Current version of cusnarks has been tested on x86_64 Linux with Ubuntu 16.04.06 and 18.04.2 distributions.

3. Add *$CUSNARKS_HOME/lib* to *LD_LIBRARY_PATH*

```sh
export LD_LIBRARY_PATH=$CUSNARKS_HOME/lib:$LD_LIBRARY_PATH
```

4. Build cusnarks to generate shared libraries in *$CUSNARKS_HOME/lib*.

```sh
make all
```
Two libraries are generated upon compilation of cusnarks
- *libcusnarks.so* : Cusnarks shared library
- *pycusnarks.so* : Cusnarks shared library wrapped with Cython wrapper so that it can be used from Python

5. Generate some metadata required for cusnarks, including :
- Roots of unity for curve [bn128][BN128]. Default option generates 2^20, which allows processing circuits of 
up to 2^19 constraints. When prompted, provide the desired number of roots to generate (maximum is 2^28).
- Location of folder to place input/output data. By default, location is *$CUSNARKS_HOME/data*

```sh
make config
```

6. Launch units tests (optional)

```sh
CUDA_VISIBLE_DEVICES=1,2 make test
```
**NOTE:** CUDA_VISIBLE_DEVICES is used to select the GPUs to be used. To get a numbered list of available GPUs type:

```sh
nvidia-smi
```
The output may look like this:

	Thu Dec 19 12:57:34 2019       
	+-----------------------------------------------------------------------------+
	| NVIDIA-SMI 440.33.01    Driver Version: 440.33.01    CUDA Version: 10.2     |
	|-------------------------------+----------------------+----------------------+
	| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
	| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
	|===============================+======================+======================|
	|   0  GeForce GTX 1080    Off  | 00000000:5E:00.0 Off |                  N/A |
	| 42%   62C    P2   106W / 180W |   2693MiB /  8119MiB |    100%      Default |
	+-------------------------------+----------------------+----------------------+
	|   1  GeForce GTX 1080    Off  | 00000000:B1:00.0 Off |                  N/A |
	| 27%   25C    P8     6W / 180W |   2687MiB /  8119MiB |      0%      Default |
	+-------------------------------+----------------------+----------------------+
	|   2  GeForce GTX 1080    Off  | 00000000:D9:00.0 Off |                  N/A |
	|  0%   24C    P8     9W / 240W |   2687MiB /  8119MiB |      0%      Default |
	+-------------------------------+----------------------+----------------------+
                                                                               	
	+-----------------------------------------------------------------------------+
	| Processes:                                                       GPU Memory |
	|  GPU       PID   Type   Process name                             Usage      |
	|=============================================================================|
	|    0     32195      C   python3                                     2683MiB |
	|    1     32195      C   python3                                     2677MiB |
	|    2     32195      C   python3                                     2677MiB |
	+-----------------------------------------------------------------------------+

In the example above, there are 3 GPUs numbered 0,1,2.

To launch GPU tests on GPUS 1 and 2 you need the following command: 
```sh
CUDA_VISIBLE_DEVICES=1,2 make test_system
```
To launch unit tests on CPU you need the following command:

```sh
make test_unit
```
## Launching Cusnarks
Launch setup and proof generation by running pysnarks.py. Following command displays help menu.

```sh
cd src/python
python3 pysnarks.py -h
```

### Trusted Setup
To generate a trusted setup from a .json/.bin compiled circuit, type:

```sh
CUDA_VISIBLE_DEVICES=<ordered list of GPUs> python3 pysnarks.py -m s -in_c <INPUT_CIRCUIT file> -pk <OUT_PROVING_KEY file> -vk <OUT_VERIFICATION_KEY file> -seed <RANDOM SEED> 
```

Mandatory arguments:
- **INPUT_CIRCUIT file** : location of file containing set of constraints generated by compiler. input file is expected to be in Extension needs to be .json, .bin or .r1cs. 
- **OUT_PROVING_KEY file** : Output Proving Key file generated during trusted setup. Extension needs to be .json or .bin. 
- **OUT_VERIFICATION_KEY file** : Output Verification Key file generated by trusted setup. Extension needs to be .json. 

A detailed description of different file formats can be found [ħere](#file-formats).

**Note** : if input or output files are placed in the *$CUSNARKS_HOME/circuits* directory, just provide file name. CUSNARKS wil automatically search in this directory

Running the trusted setup requires [snarkjs][] being installed to compute parts of the verification key. Cusnarks automatically downloads it into *$CUSNARKS_HOME/third_party_libs/snarkjs*

Optional arguments
- **-v** : Enable verification. If value is 1, setup verification is enabled. After setup is generated, cusnarks will call [snarkjs][] to generate setup. Verification is done by comparing both outputs
- **-t OUTPUT_TABLE_FILE.bin**: Precompute EC Tables and write them to OUTPUT_TABLE_FILE. 


### Prover

There are two mechanisms to generate a proof:
- Server mode : Launches a server that accepts requests to generate proofs for a given trusted setup by providing a witness file. This is the default and recommended mechanism (Prover initialization is slow)
- One off : Launches a single Prover

#### Server Mode
To launch Proof Server:
```sh
CUDA_VISIBLE_DEVICES=<ordered list of GPUs> python3 pysnarks.py -m p -pk <INPUT_PROVING_KEY file> -vk <INPUT_VERIFICATION_KEY file> 
```

Mandatory arguments:
- **INPUT_PROVING_KEY file** : Input Proving Key file generated by trusted setup. Extension needs to be .json or .bin.
- **INPUT_VERIFICATION_KEY file** : Input Verification file generated by trusted setup if verification is required when launching proof. Extension is .json

Optinal arguments:
- **-gpu** : Number of GPUs. Ex. -gpu 2 will use two GPUS. -gpu 0 will use 0 GPUs and tests will be run on CPU

To request a proof: 

```sh
CUDA_VISIBLE_DEVICES=<ordered list of GPUs> python3 pysnarks.py -m p -w <INPUT_WITNESS file file> -p <OUTPUT_PROOF file> -pd <OUTPUT_PUBLIC_DATA file> -seed <RANDOM SEED> -v <0|1>
```
Mandatory arguments:
- **INPUT_WITNESS file** : Input Witness file. Extension needs to be .bin or .dat.  
- **OUTPUT_PROOF file** :   Output file containing proof. Extension needs to be .json.  
- **OUTPUT_PUBLIC_DATA file** : Output file containing public data. Extension needs to be .json. 

A detailed description of different file formats can be found [here](#file-formats).

Optional arguments
- **-v** : Enable verification. If value is 1, proof verification is enabled. After proof is generated, cusnarks will call [snarkjs][] to verifty proof. Cusnarks will return the output 
- **INPUT_VERIFICATION_KEY file**: If verification is required, location of verification file generated during trusted setup needs to be provided. Extension is .json. 

#### Non-Server Mode
To request a proof in non-server mode (assumes server is not launched. If it has been launched, it will run on Server Mode):

```sh
CUDA_VISIBLE_DEVICES=<ordered list of GPUs> python3 pysnarks.py -m -pk<INPUT_PROVING_KEY file> -vk <INPUT_VERIFICATION_KEY file> -w <INPUT_WITNESS file file> -p <OUTPUT_PROOF file> -pd <OUTPUT_PUBLIC_DATA file> -seed <RANDOM SEED> -v <0|1>
```

## Example
In this section we will go through an example on how to compile a circuit using [circom2][] to obtain a valid set of R1CS constraints and a valid witness, and how to call cusnarks to compute the Trusted Setup and the Proof.

### Circuit Compilation
Cusnarks provides a compiler [circom2][] to generate R1CS constraints and witness from a given circuit. [circom2][] is located in *$CUSNARKS_HOME/third_party_libs/za/*. 

If you have followed the installation steps indicated in this manual rust compiler is already installed in your system. If this is the case, you can directly jumt to step 3. Otherwise, keep reading.

Steps to follow :

1. Install rust compiler 

```sh
curl https://sh.rustup.rs -sSf | sh
sudo apt-get install clang build-essential libssl-dev
source $HOME/.cargo/env
```

2. Compile [circom2][]

From *$CUSNARKS_HOME/* type:

```sh
make all
```

After this step is completed, cusnarks and [circom2][] are ready to use.

3. Review circuit

[circom2][] provides a test circuit that can be used to generate test proofs with an arbitrary number of constraints. The circuit can be found in *$CUSNARKS_HOME/third_party_libs/za/interop/circuits/cuda/circuit.circom*. The circuit is shown below.

	template T(N) {
		signal private input p;
		signal output q;

		signal tmp[N];	
	
		tmp[0] <== p;
		tmp[1] <== p + 1;
		tmp[2] <== p + 2;
		
		for (var i=0;i<N-3;i+=1) {
			tmp[i+3] <== (tmp[i] + tmp[i+1]) * (tmp[i+1] + tmp[i+2]);
		}
	
		q <== tmp[N-1];
	}
	
	#[test]
	template test1() {
		component main = T(10000);
		#[w] {
			main.p <== 2;
		}
	}
	
	component main = T(10000);

*circuit.circom* is a very simple circuit that generates *N+3* constraints from a single input *p* and a single output *q*.

In this particular circuit, *N* is equal to 10000.

To modify the number of constraints, set the number *N* in T(N) to the desired number of constraints. In our example we will set *N* to be 4000000 by changing both occurrences to **component main = T(400000)**

**NOTE:** Modified circuit name needs to be *circuit.circom*

4. Generate constraints
From the same directory where *circuit.circom* is placed, type:

```sh
../../../target/release/circom2 compile --cuda=r1cs4M_c.bin
```

Wait a few minutes and the output will be a constraint file called *r1cs4M_c.bin* located in the same folder as *circuit.circom*. The format of *r1cs4M_c.bin* is described in [constraints-circom-format][].

5. Generate witness
From the same directory where *circuit.circom* is placed, type:

```sh
../../../target/release/circom2 test --skipcompile --outputwitness
```
Wait a few minutes and the output will be a witness file called *test1.binwitness*. Unfortunately, this witness file is in big-endian format and we need to convert it to little-endian.

```sh
objcopy -I binary -O binary --reverse-bytes=4 test1.binwitness r1cs4M_w.dat
```

Move *r1cs4M_c.bin* and *r1cs4M_w.dat* to *$CUSNARKS_HOME/circuits/* folder

After this last step is completed, you have generated a constraint file and a valid witness. Next step is to generate the Trusted Setup.

The format of *r1cs4M_w.dat* is described in [witness-format][]

### Trusted-Setup
Once constraint and witness file have been generated and place in *$CUSNARKS_HOME/circuits/*, run the cusnarks to compute the Trusted Setup.

Go to directory *$CUSNARKS_HOME/src/python/* to launch Trusted-Setup.

```sh
CUDA_VISIBLE_DEVICES=1,2 python3 pysnarks.py -m s -in_c r1cs4M_c.bin -pk r1cs4M_pk.bin -vk r1cs4M_vk.json 
```

Wait a few minutes until cusnarks has finished computing the Trusted Setup. When it is done the proving key and verification key files will be computed and placed in *$CUSNARKS_HOME/circuits/* folder.

To check Trusted-Setup progress type:

```sh
tail -f $CUSNARKS_HOME/circuits/_SETUP/log
```

**NOTE:** Throughout this example we are using GPUs 1 and 2 and that is why we are setting CUDA_VISIBLE_DEVICES. This will likely change in your particular environment.

### Proof 
Once the Trusted Setup step has been computed, it is time to compute the proof. Computing the proof is a two step process.
Even though in this specific case it is not very useful to launch the prover in server/client mode as there is only a single witness created, we will show how you should proceed to launch prover in server/client mode as it is the normal operation mode.

1. Launch Proof server

Go to directory *$CUSNARKS_HOME/src/python/* to launch Proof server

```sh
CUDA_VISIBLE_DEVICES=1,2 python3 pysnarks.py -m p -pk r1cs4M_pk.bin -vk r1cs4M_vk.json
```

Wait a few seconds until the message below appears on the screen.

	GP init : 112.70901942253113
	Server listening on port 8193 ready...
	Server listening on port 8192 ready...


To check Proof server progress type:

```sh
tail -f $CUSNARKS_HOME/circuits/_PROOF/log
```
During this stage prover is being initialized with the contents of the proving key. 

2. Launch Proof client

```sh
CUDA_VISIBLE_DEVICES=1,2 python3 pysnarks.py -m p -w r1cs4M_w.dat -p r1cs4M_proof.json -pd r1cs4M_pd.json -v 1
```
Wait a few seconds until final completion message appears on screen. At the end of the process the proof and the public data files will be written to *$CUSNARKS_HOME/circuits/* folder.

To check Proof server progress type:

```sh
tail -f $CUSNARKS_HOME/circuits/_PROOF/log
```

When the proof is generated, you can launch another Proof client following the same steps as described in this section.

When the proof is generated the client will display aome type of message on screen:

	{
  	  "witness_f": "/mnt/disk2/druiz/cusnarks/circuits/r1cs4M_w.dat",
  	  "proof_f": "/mnt/disk2/druiz/cusnarks/circuits/r1cs4M_proof.json",
  	  "public_data_f": "/mnt/disk2/druiz/cusnarks/circuits/r1cs4M_pd.json",
  	  "verify_en": 1,
  	  "proof_id": 0,
  	  "result": 1,
  	  "init": [
    	    0.0102,
    	    0.04
  	    ],
  	  "Read_W": [
    	    0.3188,
    	    1.14
  	    ],
  	  "Eval": [
    	    9.4916,
    	    34.05
  	    ],
  	  "Mexp": [
    	    27.1099,
    	    97.25
  	    ],
  	  "H": [
    	    6.6682,
    	    23.92
  	    ],
  	  "Proof": 27.8771
	}

* **witness_f** : location of witness file
* **proof_f** : location of output proof file
* **public_data_f** : location of output public data file
* **verify_en** : Flag to indicate if verification was requested (1: Verification Enabled/ 0: Verification Disabled)
* **proof_id** : Proof unique identifier
* **result** : Proof result (-2: Aborted, -1: Ongoing, 0 :Failed, 1: Sucess, 2: Verification not requested) 
* **init** : Array containing two numbers. First number is time taken by initialization in seconds. Second number is the relative initialization time with respect to complete proof time
* **Read_W** : Array containing two numbers. First number is time taken by reading witness file in seconds. Second number is the relative witness reading time with respect to complete proof time
* **Eval** : Array containing two numbers. First number is time taken by poly evaluation phase in seconds. Second number is the relative poly evaluation time with respect to complete proof time
* **Mexp** : Array containing two numbers. First number is time taken by multi-exponentiation phase in seconds. Second number is the relative multi-exponentiation time with respect to complete proof time
* **H** : Array containing two numbers. First number is time taken by FFT phase in seconds. Second number is the relative FFT time with respect to complete proof time
* **Proof** : Total proof time in seconds

3. Stop Proof server

If you want to launch a new Proof server with a different proving/verification key, you need to first stop the Proof server, and then follow the steps outlined above to generate a new set of constraints, witness and Trusted Setup.

To stop the server:

```sh
python3 pysnarks.py -stop_server 1
```
## Cusnarks Options

This section describes the some of the options embedded in cusnarks via command arguments.
General usage is:
 
```sh
python pysnaks.py [OPTIONS]
```
**NOTE** Remember that cusnarks needs to be launched from *$CUSNARKS_HOME/src/python* directory.

In section [Example](#example) we described the minimum arguments required to launch cusnarks. In this section we will further detail some additional options.

To list all available options, type:

```sh
python3 pysnarks.py -h
```
Options can be divided in three groups:
* Generic mode : General info
* Setup mode : Configure Trusted setup
* Proof mode : Configure Proof generation

### Deprecated 
This section enumerates options that are either deprecated and its use may produce unintended results, or functionality that is still not implemented. At some point, all deprecated options will be deleted from the menu.

- snarkjs <SNARKJS location> : describes current location of snarkjs.
- ml <MIN_LEVELS>  : Unclear use.
- Ml <MAX_LEVELS>  : Unclear use.
- rep <N_REP>      : Unclear use.
- b <BENCHMARK>    : Unclear use.
- r_cpus RESERVED_CPUS : Mechanism to limit the number of CPUs used. 
- wf <WITNESS_FORMAT> : Input witness format (Montgomery/Normal). Currently, only normal format is accepted.
- df <DATA_FOLDER> :
	
### Generic mode
* -h

### Setup mode
Setup is launched with the following command:

```sh
python3 pysnarks.py -m s [SETUP_OPTIONS]
```

- in_c <INPUT_CIRCUIT> : 
- out_c <OUTPUT_CIRCUIT> :
- out_cf <OUTPUT_CIRCUIT_FORMAT> :
- pk <PROVING_KEY>
- vk <VERIFICATION KEY>
- kf <KEYS_FORMAT>
- d <DEBUG>
- seed <SEED

### Proof mode
Proof is launched with the following command:

```sh
CUDA_VISIBLE_DEVICES=0,1,2 python3 pysnarks.py -m p [PROOF_OPTIONS]
```
- pk PROVING_KEY
- vk VERIFICTION_KEY
- seed SEED
- w <WITNESS_FILE>
- p PROOF
- pd PUBLIC_DATA
- v VERIFY
- gpu MAX_GPUs
- stream MAX_STREAMS
- server <START_SERVER>

Additionally, it is possible to launch several comands without *-m p* option that affect the behavior of a previously launched proof server. Examples of this include:

- stop_server <STOP_SERVER> : 
- stop_client <STOP_CLIENT> :
- alive <IS_ALIVE> :
- l <LIST> :


## Architecture

Modules are divided into 4 categories depending on functionality:

1. **Infrastructure Layer** : Modules in this class have no dependencies and perform basic functionality common to all project (constants and type definition mostly). Infrastructure modules can be accessed by both host and device.

2. **Service Layer** : Modules in this class implement non core functionality used by higher layers (logging, random number generation or CUDA kernel launch abstraction). Service modules are implemented in C or CUDA C. All services layer modules are accessible by host side. Logging can be access by both host and device.

3. **Core Layer**  ; Modules in this category implement Snarks core functionalty,  including modular and elliptic curve implementation.
 Core functionality is duplicated in C and Python. Python was used as a fast prototyping implementation that could be used to validate C
 version and not as an efficient Snarks implementation. On the other hand, CUDA-C was designed with the main objective of being a very efficient
 implementation in terms of execution time. Thus, most modules are executed in the device side. Host side C core layer modules are mainly used to
 define kernel function handlers, although functionality will be extended to have CPUs accelerate computation.

4. **Applications Layer** : User applications. For now only setup and prover functionality is included, but in the future witness generation will be in in this layer.
 Application layer modules are implemented in Python and can launch CUDA kernels via cusnarks_kernel module and host side accelerated C functions via utils_host Cython
 wrapped module

Below you can see a block diagram of the software arhictecture.
![Architecture](doc/architecture.png)

## Modules

### Application

| Module Name | Executed on | Description |
|-------------|-------------|-------------|
| pysnarks | CPU | Main script that launches the setup or the proof depending on the arguments passed|
|groth_setup | CPU | Implementation of GrothSetup class |  
|groth_prover | CPU | Implementation of GrothProver class |

### Core

| Module Name | Executed on | Description |
|-------------|-------------|-------------|
| utils_host  | CPU         | Implementation of several field and elliptic curve operations. Most of them are warpped in Cython (*$CUSNARKS_HOME/src/cython/_cusnarks_kernel.pyx*) so that they can be called from Python |
| cusnarks_kernel | CPU | CUDA resource manager (memory allocation, kernel launching,...) |
| u256     | CPU | Initialization of object U256 (256 bit integer) |
|u256_device  | GPU | Implementation of u256 finite field arithmetic, including mod, addm, subm, montgomery multiplicationm... |
| zpoly    | CPU | Initialization of object Field Polynomial |
| zpoly_device   | GPU | Implementation of arithmetic on polinomials of finite field elements including FFT |
| ecbn128  | CPU | Initialization of object Elliptic Curve in G1|
| ecbn128_device  | GPU | Implementation of G1 arithmetic over BN128 curve |
| ec2bn128 | CPU | Initialization of object Elliptic curve in G2 |
| ec2bn128_device | GPU | Implementation of G2 arithmetic over BN128 curve |


Additionally, there are some additional Python modules:

| Module Name | Executed on | Description |
|-------------|-------------|-------------|
| bigint      | CPU         | Implementation of big integer functionaliy |
| zutils      | CPU         | Several utilities for field and group arithmetic |
| zfield      | CPU         | Finitie field arithmetic |
| z2field_element     | CPU    | Extended finite field arithmentic | 
|zpoly       | CPU |  Arithmetic of polynomials with finite field elements |
| ecc  | CPU |  Elliptic curve functionality |


### Service

| Module Name | Executed on | Description |
|-------------|-------------|-------------|
| pysnarks_utils | CPU      | Utilities for conversion of file formats used all around cusnarks |
| json_socket    | CPU      | Communication library to establish a server and a client |
| cuda_wrapper   | CPU      | Python wrapper to be able to launch CUDA kernels from Python |
| rng   | CPU  | Encapsulation of Random Number Generator pcg |
| log   | CPU/GPU  | Logging functionality. See [Logging][logging] for more details on how logging works |


### Infrastucture

| Module Name | Executed on | Description |
|-------------|-------------|-------------|
| types       | CPU/GPU     | basic type definition |
| constants   | CPU/CPU     | Constant definition |
| cuda        | GPU         | Some error check functionality for CUDA|

## Some Results
This section provides some performance results generated with Dell T640 server with 2xIntel Xeon Silver 4110@2.1GHz (32 cores), 128GB of memory and 2XGPU GTX 1080.
Maximum number of constraints achieved is 2^26 and took 230 seconds. Proof performance for different number of constraints is shown below.

![Some Results-performance](doc/cusnarks_performance.png)

Below it is displayed the relative performance per functional block (Mexp, FFT, Witness Read and Initialization).
![Some Results-block](doc/cusnarks_block_performance.png)

|Log2 N constraints | Time Cusnarks [s] | Time Libsnarks [s] |
|-------------------|-------------------|--------------------|
| 22                | 18                |  50                |
| 23                | 30                |  96                |
| 24                | 69                |  176               |
| 25                | 116               |  327               |
| 26                | 234               |  675               |

## File Formats
CUSNARKs requires and generates different files. The picture below shows a block diagram containing the three main actors (Setup, Proof Server and Prover) and how they relate to the 
different files. 

![File Formats](doc/file_formats.png)

In this section we will describe the different files and their formats. Below you can see a table summarizing the different 
formats.

|File type        | Extensions|
|-----------------|-----------|
| Constraints     | .bin, .json, .r1cs|
|Proving Key      | .json, .bin |
|Verification Key | .json                   |
|Witness          | .txt, .json, .dat, .bin |
|EC Tables           | .bin |
|Proof | .json    |
|Public Data | .json |
|Results | stdout |


Trusted setup consumes constraint files and produces Proving Key and Verification ket. Proof server consumes the Proving Key. The Prover consumes Verification Key and
generates the Proof, Public data and Results. Below there is a figure summarizing the structure of which block generates and consumes which file.

### Constraints
Constraint files are generated by a SNARK compiler and consumed by cusnarks trusted setup.

#### .json
Constraint system generated by [circom][].  JSON file includes the following keys:

|Key | Description |
|----|-------------|
| constraints | set of constraints. Every set is  3xN list thatincludes the R1CSA, R1CSB and R1CSC as coeff, value tupple |
| nPubInputs | Number of public inputs in the circuit |
| nOutputs | Number of outpus in the circuit |
| nVars|Number of wires|
| cirformat | Representation of values in the constraint set. 0 is 256 number. 1 is Montgomery representation |
| protocol | 'groth' |

#### .bin
Constraint system generated by [circom2][]. 32 bit words are represented as Little Endian.

|Key | Bit size | Description |
|----|----------|-------------|
| nWordsa | 64 bits | Length of file in 32 bit words |
| nPubInputs | 64 bits | Number of public inputs in the circuit |
| nOutputs | 64 bits| Number of outputs in the circuit |
| nVars|64 bits | Number of wires in the circuit |
| nConstraints | 64 bits| Number of constraints |
| cirformat | 64 bits | Representation of values in the constraint set. 0 is 256 number. 1 is Montgomery representation | 
| R1CSA_nWords | 64 bits |  Length of R1CSA constraints in 32 bit words |
| R1CSB_nWords | 64 bits | Length of R1CSB constraints in 32 bit words |
| R1CSC_nWords | 64 bits | Length of R1CSC constraints in 32 bit words | 
| R1CSA | RBCSA_nWords | R1CSA constraints |
| R1CSB | R1CSB_nWords | R1CSB constraints |
| R1CSC | R1CSC_nWords | R1CSC constraints |

R1CS constraints are represented as follows:

|Field | Bit size | Description |
|------|----------|-------------|
| nConstraints | 32 bit| Total Number of constraints |
| Number coefficients constraint_0 | 32 bits |   |
| Cum. number coefficients constraint_1 | 32 bits | N coeff constraints_0 + N coeff constraints_1 |
|...| 32 bit | ... |
| Cum. number coefficients constraint_N-1 | 32 bits | N coeff constraints_0 + N coeff constraints_1 +...+ N constraints_N-1|
| Coeff_0_0 | 32 bit| constraint 0, coefficient 0|
| Coeff_0_1 | 32 bit | constraint 0, coefficient 1|
| ... | 32 bit |... |
| Coeff_0_M-1 | 32 bit | constraint 0, coefficient M-1|
| Value_0_0 | 256 bit | constraint 0, value 0|
| Value_0_1 | 256 bit | constraint 0, value 1|
|...| 256 bit |...|
| Value_0_M-1 | 256 bit| constraint 0, value M-1|
| Coeff_1_0 | 32 bit | constraint 1, coefficient 0|
| Coeff_1_1 | 32 bit | constraint 1, coefficient 1|
|...| 32 bit |...|
| Coeff_N-1_0 | 32 bit | constraint N-1, coefficient 0|
| Coeff_N-1_1 | 32 bit | constraint N-1, coefficient 1|
|...| 32 bit | |
| Coeff_N-1_Mn-1 | 32 bit | constraint N-1, coefficient Mn-1|
| Value_N-1_0 |256 bit | constraint N-1, value 0|
| Value_N-1_1 | 256 bit | constraint N-1, value 1|
|...|256 bit|...|
| Value_0_Mn-1 | 256 bit | constraint N-1, value Mn-1|


#### .r1cs
Constraint system generated by [circom][]. See [r1cs][] for a detailed description.

### Proving Key
Trusted setup generated by cusnarks Trusted Setup and consumed by cusnarks Prover. 

#### .json
JSON includes the following keys:

|Key | Description |
|----|-------------|
| ftype | 'PK_FILE' |
| protocol |'groth' |
| Rbitlen | Size of Field elements. For BN128 Rbit len is '256'|
| k_binformat | Representation of field elements ('normal'/'montgomery')|
| k_ecformat | Representation of group elements ('affine') |
| nVars |  Number of wires in circuit|
| nPublic | Number of public data in circuit (nPubInputs + nOutputs)|
| domainBits |: Number of bits|
| domainSize | Size (power(2,domainBits))|
| field_r | Order of field (for BN128 field_r is "21888242871839275222246405745257275088548364400416034343698204186575808495617")|
| group_q | Order of group (for BN128 group_1 is "21888242871839275222246405745257275088696311157297823662689037894645226208583")|
| A | Set of elliptic curve points A in G1|
| B1 | Set of elliptic curve points B1 in G1|
| B2 | Set of elliptic curve points B2 in G2|
| C | Set of elliptic curve points C in G1|
| hExps | Set of elliptic curve points K[nInputs+1:nVars] in G1|
| polsA | QAP A |
| polsB | QAP B |
| polsC | QAP C |
| vk_alfa_1 |Elliptic curve point alpha in G1|
| vk_beta_1 | Elliptic curve point beta in G1|
| vk_beta_2 | Elliptic curve point beta in G2|
| vk_delta_1 | Elliptic curve point delta in G1|
| vk_delta_2 | Elliptic curve point delta in G2|

**Note** : Elliptic curve points in G1 are represented with three coordinate system (X, Y, Z=1). Elliptic curve points in G2
are represented with three coordinate system (X, Y, Z = [1,0])

#### .bin
32 bit words are represented as Little Endian in .bin format. The format of .bin files is:

|Field | Bit size | Description |
|------|----------|-------------|
| nWords | 32 bit | Unused |
| ftype |32 bit | File type (1) |
| protocol |32 bit | Protocol type (Groth -> 1)|
| Rbitlen |32 bit | Size of field/group (256)|
| k_binformat | 32 bit | Representation of field elements (1 -> montgomery)|
| k_ecformat | 32 bit | Representation of group elements (2 -> affine)|
| nVars | 32 bit | Number of wires in circuit|
| nPublic |32 bit| Number of public inputs in circuit|
| domainBits |32 bit | Number of bits |
| domainSize |32 bit | Size (power(2,domainBits))|
| field_r | 32 bit | Field order |
| group_q | 32 bit | Group order |
| polsA_nWords | 64 bit | Number of 32 bit words in representation of polsA|
| polsB_nWords | 64 bit | Number of 32 bit words in representation of polsB|
| polsC_nWords | 64 bit | Number of 32 bit words in representation of polsC|
| A_nWords | 64 bit| Number of 32 bit workds in representation of EC point A|
| B1_nWords | 64 bit | Number of 32 bit workds in representation of EC point B1|
| B2_nWords | 64 bit | Number of 32 bit workds in representation of EC point B2|
| C_nWords | 64 bit | Number of 32 bit workds in representation of EC point C|
| hExps_nWords | 64 bit | Number of 32 bit words in representation of EC point K[nInputs+1:nVars]|
| polsA | n bit | QAP A |
| polsB | n bit | QAP B |
| polsC | n bit | QAP C |
| alfa_1 | 512 bit | Elliptic curve point alpha in G1|
| beta_1 | 512 bit | Elliptic curve point beta in G1|
| beta_2 | 1024 bit | Elliptic curve point beta in G2|
| delta_1 | 512 bit | Elliptic curve point delta in G1|
| delta_2 | 1024 bit | Elliptic curve point delta in G2|
| A | n bit | Set of elliptic curve points A in G1|
| B1 | n bit | set of elliptic curve points B1 in G1|
| B2 | n bit | set of elliptic curve points B2 in G2 |
| C | n bit | set of elliptic curve points C in G1|
| hExps | n bit | set of elliptic curve points K[nInputs+1:nVars] in G1|

**Note** : Elliptic curve points in G1 are represented with two coordinate system (X, Y). Elliptic curve points in G2 are
also represented with two coordinate system (X, Y)

Formt of polsA, polsB and polsC is:

|Field | Bit size | Description |
|------|----------|-------------|
| nVars | 32 bit | Number of embedded polys|
| nCoeff_0 | 32 bit | Number of coefficients in poly 0 |
| nCoeff_1 | 32 bit | Number of coefficients in poly 1| 
| ... | 32 bit |...|
| nCoeff_N-1 | 32 bit | Number of coefficients in poly N-1|
| Value_0_0  | 256 bit | Value 0, Poly 0|
| Value_1_0 | 256 bit |  Value 1, Poly 0|
| ...| 256 bit|...|
| Value M-1,0 | 256 bit| Value M-1, Poly 0|
| Value 0, 1 | 256 bit| Value 1, Poly 1|
| ... | 256 bit|...|
| Value Mn-1, N-1|256 bit| Value Mn-1, Poly N-1|

### Verification Key
Verification key is  generated by CUSNSAKRS during trusted setup and consumed by [snarkjs][] during proof verification.

#### .json
JSON contains the following keys:

|Key | Description |
|----|-------------|
| Rbitlen | Size of field/group (for BN128 Rbitlen is "256")|
| binFormat | Format of field elements ('normal')|
| domainBits | Number of bits|
| domainSize | Size of FFT. (power(2,domainSize))|
| ecFormat | Format of Elliptic curve elements ('affine')|
| field_r | Order of field (for BN128 field_r is "21888242871839275222246405745257275088548364400416034343698204186575808495617")|
| group_q | Order of group (for BN128 group_1 is "21888242871839275222246405745257275088696311157297823662689037894645226208583")|
| nPublic | Number of public variables (Number of public input + number of output)|
| nVars | Number of wires in the circuit|
| protocol | 'groth'|
| vk_alfa_1 | Elliptic curve element alpha in G1|
| vk_beta_2 | Elliptic curve element beta in G2|
| vk_delta_2 | Elliptic curve element delta in G2|
| vk_gamma_2 | Elliptic curve element gamma in G2|
| vk_alfabeta_12 | Pairing e(alpha, beta)|
| IC | Elliptic curve elements K[0:nInputs] in G1|


### Witness
Witness is required for cusnarks prover. There are four possible formats (.txt, .json, .bin, .dat). There are two binary
represenations (.bin and .dat) because different applications generate different formats.

#### .txt
Text file including a witness in every line.

#### .json
JSON file including a comma separated witness in very line. Witness sequences is cintained between square brackets ([..])

#### .bin
Generated by [circom][]. Format is :

|Field | Bit size | Description |
|------|----------|-------------|
| nWords | 32 bit |  Number of witness input |
| wSize | 32 bit |  Size of witness in 32 bit words |
| other | 32 bit |  Empty |
| witness_0 | n bit | Witness 0 |
| witness_1 | n bit | Witness 1 |
| ... | n bit | ... |
| witness_N-1 | n bit | Witness N-1|

#### .dat
Generated by [circom2][]. Format is :

|Field | Bit size | Description |
|------|----------|-------------|
|nWords|32 bit| Number of witness input |
|wSize |32 bit| Size of witness in 32 bit words|
| other |64 bit| Empty|
| other |128 bit| Empty|
| witness_0|n bit | Witness 0|
| witness_1|n bit | Witness 1|
| ...| n bit | ... |
| witness_N-1[n bit] : Witness N-1 |

### EC Tables
OUtput from cusnarks Setup and consumed by cusnarks Prover.

#### .bin
Precomputed table binary file has the following format:

|Field | Bit size | Description |
|------|----------|-------------|
|Order|32 bit| Table generation order. Tables will have 2^Order elements |
|Offset T A|64 bit| Offset in 32 bit words of A Table 1 in file|
|Offset T B2|64 bit| Offset in 32 bit words of B2 Table 1 in file|
|Offset T B1|64 bit| Offset in 32 bit words of B1 Table 1 in file|
|Offset T C|64 bit| Offset in 32 bit words  of C Table 1 in file|
|Offset T hExps|64 bit| Offset in 32 bit words  of hExps Table 1 in file|
|T1 A|Var| A Table  (from 0 to nVars) |
|T1 B2|Var| B2 Table  (from 0 to nVars)|
|T1 B1|Var| B1 Table 1 (from 0 to nVars)|
|T1 C|Var| C Table 1 (from nPublic+1 to nVars)|
|T1 hExps|Var| hhExps Table 1 (from 0 to domainSize-1)| 

### Proof
Output from cusnarks Prover and consumed by verifier [snarkjs][]. 

#### .json
JSON file contains the following keys:

|Key | Description |
|----|-------------|
| pi_a |  Elliptic curve point in G1|
| pi_b |  Elliptic curve point in G2|
| pi_c |  Elliptic curve point in G1|
| protocol | groth|

### Public Data
Public data 

#### .json
List containining the public elements in the input data separated by a comma.

### Results
Results are generated by the Prover. The format is  Python dictionary including the following keys:

|Key | Description |
|----|-------------|
| Proof | Total proof time |
| status | Result of proof verification. 0 -> Proof is incorrect. 1 -> Proof is correct, 2 -> No verification requested |
| init | Prover initialization time list. First element in the list is the absolute time. Second element is the relative time with respect to the total proof time|
| read W | Time spent reading witness file. First element in the list is the absolute time. Second element is the relative time with respect to the total proof time|
| Mexp | Time spent cAomputing multi-exponentiation in G1 and G2. First element in the list is the absolute time. Second element is the relative time with respect to the total 
| Eval | Time spent computing QAP from R1CS. First element in the list is the absolute time. Second element is the relative time with respect to the total proof time|
| H | Time spent computing H. First element in the list is the absolute time. Second element is the relative time with respect to the total proof time|

**Note** : the sum of all the time spent in different submodules is less than the total proof time. This is because some
things are done in parallel.


## Other Info
### Directory Structure
* **build**\    : Object files
* **data**\     : Auxiliary files (precomputed roots of unity,...)
* **lib**\      : Generated dynamic libraries
* **src**\
  - *cuda*\     : C/C++/CUDA sources (.cpp, .c, .cu, .h)
  - *cython*\   : Cython files (.pyx, .pxd)
  - *python*\   : Python source files (.py)
* **test**\
  - *python*\   : Python unit test. They mainly test Python library using [unittest][] unit testing framework. 
   However, there are some files (*xxx_cu_xxx.py*) that test CUDA functions as well.
  - *c*\        : C unit tests for host side functionality.
     -*aux_data*\ : Files used during tests.
  - *ideas*\    : Folder containing small scripts testing some ideas to be implmented in main code
* **profiling**\ : Profiling information
   - *python*\   : Collection of scripts to measure time of CUDA functions 
* **third_party_libs**\ : Exteral libraries used will be automatically downloaded here
   - *pcg-cpp*\  : implementation of PCG family of random number generators. Full details can be found at the [PCG-Random website].
   - *snarkjs*\  : snarkjs repository
   - *za*\ : rust circom repository
* **circuits**\   : Default location where circuits are processed by cusnarks.
   - *_SETUP*\    : Default location where Trusted setup files are copied.
   - *_PROOF*\    : Default location where Proof fils are copied.
* **config**\     : Location of initial configuration scripts.

### Logging
There are two types of logging.
From Python side, logs during trusted setup are recorded in *$CUSNARKS_HOME/circuits/_SETUP/log*, and during proof in 
*$CUSNARKS_HOME/circuits/_PROOF/log*

From C/CUDA side, logs are available as well, but are disabled by default. To enable logs, edit *$CUSNARKS_HOME/src/cuda/log.h* file.
There are two constants defined :
- **LOG_LEVEL** : There are 5 levels of logging (nolog, error, warning, info, debug). To enable logs, define *LOG_LEVEL* constant as 
one of these types. Use function *LogError()*, *LogWarning()*, *LogInfo()*, *LogDebug()* to log a message. Message is displayed in standard 
output. These functions are only to be used in CPU side.
- **LOG_TID** : Define the kernel thread id that you want to display the message. To log a message for a particular TID, use 
functions *LogXXXBigNumberTid* or *LogXXXTid*, where XXX is Error, Warning, Info or Debug depending on the severity. 
*LogXXXBigNumberTid* displays a U256 integer. *LogXXXTid* can be used to display other data types. Its use is similar to printf. Both functions are only to be used in GPU side.

After modifying *log.h* you will need to do a *make clean build* to recompile with the new changes.

**NOTE:** if logging is disabled, logging function calls are not compiled and thus add no overhead.

## References
The following is a non-comprehensive list of interesting material:
* [DIZK][] : Paper on distributed snarks
* [Elliptic curve primer][] : elliptic curve crypto primer book
* [snarkjs][]  : javascript implementation of zksnarks
* [bellman][]  : bellman rust implemenation of zksnarks
* [libsnark][] : libsnark library
* [belle_cuda][] : Alternative CUDA implementation of Snarks


## Next Steps
Future direction of cusnarks will include:
1. Increase speed. Target is to be able to compute 2^27 constraints under 10 minutes in a single PC
2. Increase curve coverage. Currently, only curve [bn128][] is supported. Next versions will include BLS12-381, MNT4 and MNT6.
3. Adapt cusnarks to multi-server architecture so that multiple PCs can be used to compute a proof.

[dependencies]: #Requirements 
[snarkjs]: https://www.github.com/iden3/snarkjs
[circom]: https://www.github.com/iden3/circom
[circom2]:https://www.github.com/iden3/za
[PCG-Random website]: http://www.pcg-random.org
[unittest]: https://python.org/3/library/unittest.html
[Architecture]: #Architecture
[Modules]: #Modules
[Installation]: #Installation
[Using Cusnarks]: #Using-Cusnarks
[Other Info]: #Other
[File Formats]: ##File-Formats
[r1cs]:https://github.com/iden3/circom/blob/c_build/doc/r1cs_bin_format.md
[bn128]:https://github.com/ethereum/py_ecc/tree/master/py_ecc/bn128
[Overview]: #Overview
[Some Results]: #Some-Results
[Logging]: ###Logging
[constraints-circom-format]: ###Constraints
[witness-format]: ###Witness
[Example]: #Example
[Cusnarks Options]: #Cusnarks-Options
[Next Steps]: #Next-Steps
[References]: #References
[DIZK]: https://eprint.iacr.org/2018/691.pdf 
[Elliptic curve primer]: http://tomlr.free.fr/Math%E9matiques/Math%20Complete/Cryptography/Guide%20to%20Elliptic%20Curve%20Cryptography%20-%20D.%20Hankerson,%20A.%20Menezes,%20S.%20Vanstone.pdf 
[bellman]: https://github.com/zkcrypto/bellman
[libsnark]:  https://github.com/scipr-lab/libsnark 
[belle_cuda]: https://github.com/matter-labs/belle_cuda
