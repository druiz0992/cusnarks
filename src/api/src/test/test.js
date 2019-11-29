/* 
BEFORE RUNNING THE TEST:

1. Make sure cusnakrs is running:
export LD_LIBRARY_PATH=/mnt/disk2/tester/cusnarks/lib:$LD_LIBRARY_PATH
cd /mnt/disk2/tester/cusnarks/src/python
python3 pysnarks.py -alive IS_ALIVE

2. If the output of the command is not {success: 1} you must start cusnark. Open a new terminal and run:
export LD_LIBRARY_PATH=/mnt/disk2/tester/cusnarks/lib:$LD_LIBRARY_PATH
cd /mnt/disk2/tester/cusnarks/src/python
CUDA_VISIBLE_DEVICES=1,2 python3 pysnarks.py -m p -pk /mnt/disk2/tester/cusnarks/circuits/rollup_1x8_pk.bin -vk /mnt/disk2/tester/cusnarks/circuits/rollup_1x8_vk.json

3. Wait until you see "...server is ready", it will take a bit more than a minute. Then repeat step 1 to check that the terminal session where the test will be run can access cusnark.
*/

const supertest     = require("supertest");
const child_process = require("child_process");
const should        = require("should");

const url   = "http://localhost:8080";
const host  = supertest.agent(url);

// WARNING: THIS INPUT CORRESPONDS TO A 1 X 8 ROLLUP CIRCUIT!
const input = {"oldStRoot":"0","feePlanCoins":0,"feePlanFees":0,"txData":["0"],"rqTxData":[0],"s":[0],"r8x":[0],"r8y":[0],"loadAmount":[0],"ethAddr":[0],"ax":[0],"ay":[0],"step":[0],"ax1":[0],"ay1":[0],"amount1":[0],"nonce1":[0],"ethAddr1":[0],"siblings1":[[0,0,0,0,0,0,0,0,0]],"isOld0_1":[0],"oldKey1":[0],"oldValue1":[0],"ax2":[0],"ay2":[0],"amount2":[0],"nonce2":[0],"ethAddr2":[0],"siblings2":[[0,0,0,0,0,0,0,0,0]],"isOld0_2":[0],"oldKey2":[0],"oldValue2":[0]}

describe("cusnarks API test. ",function(){

    it("start the API server", () => require("../index"));

    it("should start with idle status", (done) => checkStatus("Idle", done));

    it("should receive a proof generation reqest", (done) => {
        host
        .post("/input")
        .send(input)
        .expect("Content-type",/json/)
        .expect(200) // THis is HTTP response
        .end((err,res) => {
            // HTTP status should be 200
            res.status.should.equal(200);
            done();
        });
    });

    it("should be pending status. If not cusnarks probably is not initialized", (done) => checkStatus("Pending", done));

    it("Should cancel the proof generation", async () => {
        const res = await cancelProof();
        res.status.should.equal(200);
        while (await getStatus() === "Pending") {
            console.error("...Waiting for server to cancel the proof generation");
            child_process.execSync("sleep 2");
        }
        (await getStatus()).should.equal("Idle");
    });

    it("should receive a second proof generation request", (done) => {
        host
        .post("/input")
        .send(input)
        .expect("Content-type",/json/)
        .expect(200) // THis is HTTP response
        .end((err,res) => {
            // HTTP status should be 200
            res.status.should.equal(200);
            done();
        });
    });

    it("Wait until proof generation finishes, it should be succesfull! (it will take a while)", async () => {
        while (await getStatus() === "Pending") {
            console.error("...Waiting for server to finish proof generation");
            child_process.execSync("sleep 2");
        }
        (await getStatus()).should.equal("Finished");
    });

    it("stop the server", () => {
        console.log("stopping the server");
        process.exit(0)
    });
});

// HELPER FUNCTIONS
async function getStatus() {
    return new Promise ((resolve, reject) => {
        host
        .get("/status")
        .end((err,res) => {
            if(err !== null) reject("_error_geting_status");
            resolve(res.body.status);
        });
    }); 
}

async function cancelProof() {
    return new Promise ((resolve, reject) => {
        host
        .post("/cancel")
        .end((err, res) => {
            if(err !== null) reject("_error_geting_status");
            resolve(res);
        });
    }); 
}

function checkStatus(expected, done) {
    host
    .get("/status")
    .expect("Content-type",/json/)
    .expect(200) // THis is HTTP response
    .end((err,res) => {
        // HTTP status should be 200
        res.status.should.equal(200);
        // Error key should be false.
        console.log(res.body.status);
        
        res.body.status.should.eql(expected);
        done();
    });
}