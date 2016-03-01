"use strict";

function MultivariateNormal(mean, cov) {
  if (mean.hasOwnProperty('mean')) {
    var params = mean;
    this.mean = params.mean;
    this.dim = this.mean.length;
    this.constant = -0.5 * Math.log(2.0 * Math.PI) * this.dim;
    if (params.hasOwnProperty('covL') && !params.hasOwnProperty('cov')) {
      this.covL = params.covL;
      this.cov = this.covL.multiply(this.covL.transpose());
    } else {
      this.cov = params.cov;
    }
    if (params.hasOwnProperty('covL') && params.hasOwnProperty('logDet')) {
      this.covL = params.covL;
      this.logDet = params.logDet;
    } else {
      this.setCovariance(params.cov);
    }
    if (params.hasOwnProperty('invCov')) {
      this.invCov = params.invCov;
    }
  } else {
    this.mean = mean;
    this.dim = mean.rows;
    this.setCovariance(cov);
  }
  this.constant = -0.5 * Math.log(2.0 * Math.PI) * this.dim;
  this.logDet = this.covL.diagonal().map(Math.log).sum();
}

MultivariateNormal.getNormal = function() {
  var x, y, w;
  do {
    x = Math.random() * 2 - 1;
    y = Math.random() * 2 - 1;
    w = x * x + y * y;
  } while (w >= 1.0)
  return x * Math.sqrt(-2 * Math.log(w) / w);
};

MultivariateNormal.getSample = function(dim) {
  var dist = new MultivariateNormal(zeros(dim), eye(dim));
  return dist.getSample();
};

MultivariateNormal.prototype.setCovariance = function(cov) {
  this.cov = cov;
  this.covL = cov.chol();
  this.logDet = this.covL.diagonal().map(Math.log).sum();
};

MultivariateNormal.prototype.getSample = function() {
  var z = Float64Array.build(MultivariateNormal.getNormal, this.dim, 1);
  return this.mean.add(this.covL.multiply(z));
};

MultivariateNormal.prototype.logDensity = function(x) {
  var diff = this.mean.subtract(x);
  return this.constant - this.logDet - 0.5 * (this.covL.bsolve_inplace(this.covL.fsolve_inplace(diff), {transpose: true})).norm2();
};

MultivariateNormal.prototype.gradLogDensity = function(x) {
  var diff = x.subtract(this.mean);
  // return this.cov.llt_solve(diff);
  return this.covL.bsolve_inplace(this.covL.fsolve_inplace(diff), {transpose: true}).scale(-1);
};

MultivariateNormal.prototype.toString = function() {
  return 'mean: ' + this.mean.transpose().toString() +
    '\ncov:  \n' + this.cov.toString() +
    '\ncovL: \n' + this.covL.toString() +
    '\nlogDet: ' + this.logDet;

};