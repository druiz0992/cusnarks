'use strict';

var utils = require('../utils/writer.js');
var ProofGenerator = require('../service/ProofGeneratorService');
var onError = require('../utils/errorManager.js').onError;

module.exports.getStatus = function getStatus (req, res, next) {
  ProofGenerator.getStatus()
    .then(function (response) {
      utils.writeJson(res, response);
    })
    .catch(function (err) {
      onError(err, req, res);
    });
};

module.exports.postCancel = function postCancel (req, res, next) {
  ProofGenerator.postCancel()
    .then(function (response) {
      utils.writeJson(res, response);
    })
    .catch(function (err) {
      onError(err, req, res);
    });
};

module.exports.postInput = function postInput (req, res, next) {
  var input = req.swagger.params['Input'].value;
  ProofGenerator.postInput(input)
    .then(function (response) {
      utils.writeJson(res, response);
    })
    .catch(function (err) {
      onError(err, req, res);
    });
};
