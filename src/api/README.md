# Proof Service
API server used to integrate [cusnark](https://github.com/iden3/cusnarks) for generating proofs using GPU acceleration.

## API

The API specification is written following the [Open API Specification (OAS) 3.0](https://swagger.io/specification/).
In order to make life easier some [open source tools](https://swagger.io/tools/open-source/) are used to:
- Edit the specification.
- Generate server code: endpoints and validation.
- Host a nice looking and interactive documentation.

###  View / Edit the specification (/src/api/spec.yaml)

Of course you can use your favorite IDE to edit /api/spec.yaml. However using a OAS specific editor is recommended as it provides nice features such as validation and frontend documentation preview. The recommended editor is [Swagger editor](https://github.com/swagger-api/swagger-editor):

- Using docker:
0. Install Docker (if you don't have it already)
1. `docker pull swaggerapi/swagger-editor`
2. `docker run --rm --name swagger-editor -d -p 80:8080 swaggerapi/swagger-editor`
3. Use your browser: http://localhost:80
4. Import the file: File/Import file => (path to the repo)/proof-generator/spec.yaml
5. Save changes: File/Save as YAML => Move the file from your Downloads directory to the path of the repo and replace the old version. ***NOTE: Docker will keep changes made to browser sessions, even if the container is restarted, however the only way to persist changes is to export the file as described in this point.***
6. To stop the server: `docker kill swagger-editor`

- Run locally *Not tested*:
0. Install Node 6.x and NPM 3.x (if you don't have it already)
1. Clone the [repo](https://github.com/swagger-api/swagger-editor)
2. Run `npm start`

- Use a hosted solution like [swagger hub](https://swagger.io/tools/swaggerhub/)

### Generate code

The API server code is based on the generated code of [swagger-codegen](https://github.com/swagger-api/swagger-codegen).

By doing so, consistency between documentation and the deployed service is ensured. Additionally you get effortless input validation, mock ups when the functionalities are not implemented and 0 boilerplate writing.

***When changes to spec.yaml are made, the generated code should be updated***, without overwriting the actual code and changes should be merged. To do so:

0. Install Docker (if you don't have it already)
1. Export code in nodejs-server language: `docker run --rm --name swagger-codegen -v ${PWD}/src/api/:/local swaggerapi/swagger-codegen-cli generate -i /local/spec.yaml -l nodejs-server -o /local/codegen` ***Note that you can use other languages, for instance to generate client code***
2. Check differences between fresh generated code (api/codegen) and current server code (api/src). It's recommended to use a diff tool such as [Meld](http://meldmerge.org/) to see changes and merge them. In general the changing parts are: api/src/api/swagger.yaml (take all the changes from the codegen version), files under the controller directory (only take the function definitions and inputs from codegen), files under the service directory (only take the function definitions and the example values in case the logic is not implemented yet and inputs from codegen) *If permission errors are thrown, run:* `sudo chown -R $USER api/codegen`

### Test / Run the API

PRE: Compile a circuit using circom, run a trusted setup to generate proving and verifying keys.
1. Start cusnarks as described on the README at the root of this repo.
2. copy /src/api/config-example.json to /src/api/config.json an edit as needed:

| Key            | Description                                                         | Example                                    |
|----------------|---------------------------------------------------------------------|--------------------------------------------|
| circuitFile    | Path of the compiled circuit                                        | /home/user/circuit.json                    |
| proofFile      | Path where the generated proof will be stored                       | /home/user/proof.json                      |
| publicDataFile | Path where the public data will be stored                           | /home/user/publicData.json                 |
| witnessFile    | Path where the binary witness will be stored to be used by cusnarks | /home/user/witness.dat                     |
| pysnarks       | Path to pysnarks.py                                                 | /home/user/cusnarks/src/python/pysnarks.py |
| cudaList       | List of CUDA GPUs to be used during proof generation                | 1,2                                        |

* For testing: `npm install` (just once), then every time you want to test: `npm test`
* For running: `npm start`
***The npm ... commands must be run from /src/api/src***

### View the documentation

In order to offer access to the API documentation to consumers and developers, [Swagger UI](https://github.com/swagger-api/swagger-ui) is used. This can be hosted in different ways:

- Publish the documentation (/src/api/spec.yaml) on a hosted service such as [Swagger Hub](https://app.swaggerhub.com)

- Self host using docker:
1. Install Docker (if you don't have it already)
2. `docker run --rm --name swagger-ui -d -p 80:8080 -e SWAGGER_JSON=/doc/spec.yaml -v ${PWD}/src/api:/doc swaggerapi/swagger-ui`
3. Use your browser: http://localhost:80
4. To stop the server: `docker kill swagger-ui`

## TO DO
0. Test
1. Back to realness (HAVE A LOOK AT: "// UNCOMMENT FOR REALNESS"), right now part of the code is commented for testing with hardcoded values.
2. Implement cancel call
3. Improve validation and descriptions on the API doc (spec.yaml)