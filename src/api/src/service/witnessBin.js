const bigInt = require("snarkjs").bigInt;
const fs     = require("fs");

function writeUint32(h, val) {
    h.dataView.setUint32(h.offset, val, true);
    h.offset += 4;
}

function writeBigInt(h, bi) {
    for (let i = 0; i < 8; i++) {
        let v = Number(bi.shr(i*32).and(bigInt("0xFFFFFFFF")));
        writeUint32(h, v);
    }
}

function calculateWitnessLen(witness) {
    let size = 0;
    // beta2, delta2
    size += witness.length * 32;
    return size;
}

function addHeader(h, witness){
    // witness element size (calculated in number of words(32 bits))
    const witnessSize = 8; // witness size = 256 bits = 8 words
    // witness elements indicated by witnessSize
    const witnessLen = witness.length;
    // write witness size
    writeUint32(h, witnessLen);
    writeUint32(h, witnessSize);
    writeUint32(h, 0);
    writeUint32(h, 0);
}

function buildWitnessBin(witness) {
    const headerLen = 4;

    const witnessLen = calculateWitnessLen(witness);
    const buffLen = witnessLen + headerLen * 32;
    const buff = new ArrayBuffer(buffLen);

    const h = {
        dataView: new DataView(buff),
        offset: 0
    };

    addHeader(h, witness);

    for (let i = 0; i < witness.length; i++) {
        writeBigInt(h, witness[i]);
    }
    return buff;
}

async function writeWitnessBin(witness, file) {
    return new Promise( (resolve, reject) => {
        let bin;
        try {
            bin = buildWitnessBin(witness);
        } catch (error) {
            reject(error);
            return
        }
        let wstream = fs.createWriteStream(file);
        wstream.write(Buffer.from(bin));
        wstream.end();
        wstream.on('finish', resolve);
        wstream.on('error', reject);
    });
}

module.exports = {
    writeWitnessBin,
};