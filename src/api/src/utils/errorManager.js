var writer = require('./writer.js');

const logInfo = process.env.NODE_ENV === 'development';
const defaultCode = 500;
const defaultResponse = {
  code: "UNKNOWN_ERROR",
  message: "This is emabarrasing, something went wrong while processing your request. Try again later, if the rror persist please contact support."
};

exports.onError = function(err, req, res, next) {
  let code;
  let response;

  // Validation error
  if (err.failedValidation) {
    response = {
      code: "INPUT_VALIDATION_ERROR",
    };
    code = 401;
    if (Array.isArray(err.results.errors)) {
      response.message = ""
      for (var error of err.results.errors) {
        response.message += `\n\t- Incorrect value for ${error.path}: ${error.message}.`
      }
    }
    else response.message = `There was an error on the provided parameters.`;
  }

  if (err.notImplemented) {
    response = {
      code: "FEATURE_NOT_IMPLEMENTED_YET",
      message: `This feature is not fully implemented yet. You can use the mockup data provided in this response but keep in mind that the values are reandomly generated.`,
      mockup: err.mockup
    };
    code = 501;
  }

  if (err.name == "TypeError [ERR_INVALID_ARG_TYPE]") {
    console.error("req: ", req);
    console.error("-------------------------------------------------------");
    console.error("res: ", res);
    response = {
      code: "UNEXPECTED_RESPONSE",
      message: `The response had an unconsistent format with the documentation. Please inform the developers of this error`
    };
    code = 500;
  }

  // controlled internal server error
  if (err.internal) {
    response = {
      code: "INTERNAL_SERVER_ERROR",
      message: err.message
    };
    code = 500;
  }

  // Unknown error
  if (response == undefined) {
    console.error(err);
    code = defaultCode;
    response = defaultResponse;
  }

  // Response
  writer.writeJson(res, response, code);
}
