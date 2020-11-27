server:
  serviceapi: :3000
  adminapi: :3001
debug: true
proofservice:
  loopinterval: 5
proverparams:
  cusnarkspath: "/usr/src/app/cusnarks"
  cufiledescriptor :
     circuitcpp :  "hermez.cpp"
     circuitdat :  "hermez_witbuild.dat"
     inputsfile:   "hermez_input.json"
     pkfile   : "hermez_pk.zkey"
     vkfile   : 
  witdatafile  : "hermez_w.wshm"
  pdfile   : "hermez_pd.json"
  pfile    : "hermez_p.json"
  seed: 436442495 
  verifyen   : "1"
  gpuen : 1
  zken : 0
  batchsize : 15
  maxbatchsize : 20
  nstreams : 5
logfile: "/var/log/proof_service/proof_service.log"

