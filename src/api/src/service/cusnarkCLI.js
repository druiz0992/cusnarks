const util                  = require('util');
const exec                  = util.promisify(require('child_process').exec);

exports.generateProof = async (config) => {
// build cuSNARKs call
    const cmd = `cd ${config.pysnarkPath} && \
CUDA_VISIBLE_DEVICES=${config.cudaList} \
python3 pysnarks.py -m p \
-w ${config.witnessFile} \
-p ${config.proofFile} \
-pd ${config.publicDataFile} \
-v 1`;
    console.error("EXECUTING cudaProofGenerator: ", cmd);
    return await exec(cmd);
}
// export LD_LIBRARY_PATH=/home/tester/production/cusnarks/lib:$LD_LIBRARY_PATH