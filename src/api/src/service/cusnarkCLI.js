const exec                  = require('child_process').exec;
const psTree                = require('ps-tree');

const kill = async function (pid) {
    const signal = 'SIGKILL';
    psTree(pid, function (err, children) {
        [pid].concat(
            children.map(function (p) {
                return p.PID;
            })
        ).forEach(function (tpid) {
            try { 
                process.kill(tpid, signal);
                console.log("KILL PROCESS: ", tpid);
            }
            catch (err) {
                console.log("ERROR CANCELLING CUSNARK PROOF GENERATION");
             }
        });
    });
  };

const noProcess = "NO CURRENT PROCESS";

let resolvePRomise;
let child = noProcess;

exports.generateProof = (config) => {
  // build cuSNARKs call
  return new Promise( (resolve, reject) => {
        const cmd = `cd ${config.pysnarkPath} && \
            CUDA_VISIBLE_DEVICES=${config.cudaList} \
            python3 pysnarks.py -m p \
            -w ${config.witnessFile} \
            -p ${config.proofFile} \
            -pd ${config.publicDataFile} \
            -v 1`;

        console.error("EXECUTING cudaProofGenerator: ", cmd);
        resolvePromise = resolve;
        child = exec(cmd, (error, stdout, stderr) => {
            if (error) {
                child = noProcess;
                reject(error);
                return;
            }
            child = noProcess;
            resolve(stdout.trim());
        }); 
  });
}

exports.cancelProof = async () => {
    if (child !== noProcess){
        await kill(child.pid);
        resolvePromise(JSON.stringify({ status: 2 }));
        child = noProcess;
    } 
};