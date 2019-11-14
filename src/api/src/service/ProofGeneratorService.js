'use strict';

const fs                    = require("fs")
const zkSnark               = require("snarkjs");
const { writeWitnessBin }   = require("./witnessBin");
const cu                    = require("./cusnarkCLI");

const config = JSON.parse(fs.readFileSync(`${__dirname}/../../config.json`, "utf8"));
// UNCOMMENT FOR REALNESS
// const circuitDef  = JSON.parse(fs.readFileSync(config.circuitFile, "utf8"));
// const circuit     = new zkSnark.Circuit(circuitDef);

// Strings matching API doc
const state = {
    IDLE:     "Idle",
    ERROR:    "Error",
    PENDING:  "Pending",
    FINISHED: "Finished",
};
let currentState = state.IDLE;
var currentProof = {};

/**
 * Get the status of the proof generator.
 * Returns the status of the proof generator.
 *
 * returns status
 **/
exports.getStatus = function() {
  return new Promise(function(resolve, reject) {
    let status = {status: currentState}
    if (currentState === state.FINISHED) {
      status.proof = currentProof.proof;
      status.pubData = currentProof.pubData;
    }
    return resolve(status);
  });
}


/**
 * Abort the proof that is currentl being generated
 * Send a petition to abort the creation process of a proof
 *
 * no response value expected for this operation
 **/
exports.postCancel = function() {
  return new Promise(function(resolve, reject) {
    var examples = {};
    examples['application/json'] = true;
    // TODO: RETURNING A MOCKUP, NOT IMPLEMENTED YET
    reject({
      notImplemented: true,
      message: "This feature is not fully implemented yet. You can use the mockup data provided in this response but keep in mind that the values are reandomly generated.",
      mockup: examples[Object.keys(examples)[0]]
    })
    // return resolve(answer);
  });
}


/**
 * Send inputs to generate a proof
 * Send a petition of proo generation based on inputs. Note that this endpoint will just schedule the proof generation, not return it.
 *
 * input Input Input for the proof generation circuit.
 * no response value expected for this operation
 **/
exports.postInput = function(input) {
  return new Promise(function(resolve, reject) {
    // UNCOMMENT FOR REALNESS: 
    // const witness = circuit.calculateWitness(input);
    // writeWitnessBin(witness, config.witnessFile)
    //   .then(() => {
    //     genProof(true); // generate the proof, don't wait for it!
    //     resolve();
    //   })
    //   .catch((e) => {
    //     console.error("ERROR GENERATING WITNESS: ", e);
    //     reject({
    //       internal: true,
    //       message: "An error has occured while preparing data for proof generation. Please check the corectness of the submited values"
    //     });
    //   });
    genProof();
    resolve();
  });
}

/**
 * Generate a poof given a binary input
 *
 * input Input Input for the proof generation circuit in binary format as expected by cuSnarks.
 * no response value expected for this operation, however when the operation is finished, it will change the state
 **/
function genProof() {
  // start execution
  currentState = state.PENDING;
  cu.generateProof(config)
    .then((cmdRes) => {
      let response = 0;
      try {
        response = JSON.parse(cmdRes.stdout).status;
      } finally {
        if (response === 1) {
          // PROOF GENERATED SUCCESSFULY
          currentProof = {
            proof: JSON.parse(fs.readFileSync(config.proofFile, "utf8")),
            pubData: JSON.parse(fs.readFileSync(config.publicDataFile, "utf8"))
          };
          currentState = state.FINISHED;
          console.log("PROOF GENERATED!");
          return
        } else {
          // ERROR GENERATING PROOF
          console.error("ERROR GENERATING PROOF; UNEXPECTED RESPONSE: ", cmdRes);
          console.error("MAKE SURE cusnarks IS RUNNING!!");
          currentState = state.ERROR;
        }
      }
    })
    .catch((e) => {
      console.error("ERROR GENERATING PROOF: ", e);
      currentState = state.ERROR;
    });
}