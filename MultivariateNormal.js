"use strict";

function MultivariateNormal(mean, cov) {
  this.mean = mean;
  this.dim = mean.rows;
  this.constant = -0.5 * Math.log(2.0 * Math.PI) * this.dim;
  this.setCovariance(cov);
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

MultivariateNormal.prototype.setCovariance = function(cov) {
  this.cov = cov;
  this.L = cov.chol();
  this.logdet = this.L.diagonal().map(Math.log).sum();
};

MultivariateNormal.prototype.getSample = function() {
  var z = Float64Array.build(MultivariateNormal.getNormal, this.dim, 1);
  return this.mean.add(this.L.multiply(z));
};

MultivariateNormal.prototype.logDensity = function(x) {
  var diff = this.mean.subtract(x);
  return this.constant - this.logdet - 0.5 * (this.L.bsolve_inplace(this.L.fsolve_inplace(diff), {transpose: true})).norm2();
};

MultivariateNormal.prototype.gradLogDensity = function(x) {
  var diff = x.subtract(this.mean);
  // return this.cov.llt_solve(diff);
  return this.L.bsolve_inplace(this.L.fsolve_inplace(diff), {transpose: true}).scale(-1);
};