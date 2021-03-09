server:
  serviceapi: :3000
  adminapi: :3001
debug: true
proofservice:
  loopinterval: 5
provertype: "cusnarks"
proverparams:
  path: "/usr/src/app/cusnarks"
  proverfiledescriptor :
     circuitcpp :  "circuit-352-32-256-64.cpp"
     circuitdat :  "circuit-352-32-256-64.dat"
     inputsfile:   "input-352-32-256-64.json"
     pkfile   : "circuit-352-32-256-64_hez2.zkey"
     vkfile   : 
  witdatafile  : "circuit-352-32-256-64_w.wshm"
  pdfile   : "circuit-352-32-256-64_pd.json"
  pfile    : "circuit-352-32-256-64_p.json"
  seed: 436442495 
  verifyen   : 1
  gpu : 4
  zken : 0
  batchsize : 15
  maxbatchsize : 20
  nstreams : 5
  cpu : 64
  timeout : 5
  log : 0
logfile: "/var/log/proof_service/proof_service.log"

