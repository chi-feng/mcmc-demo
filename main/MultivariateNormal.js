"use strict";

class MultivariateNormal {
  constructor(mean, cov) {
    if (mean.hasOwnProperty("mean")) {
      const params = mean;
      this.mean = params.mean;
      this.dim = this.mean.length;
      this.constant = -0.5 * Math.log(2.0 * Math.PI) * this.dim;
      if (params.hasOwnProperty("covL") && !params.hasOwnProperty("cov")) {
        this.covL = params.covL;
        this.cov = this.covL.multiply(this.covL.transpose());
      } else {
        this.cov = params.cov;
      }
      if (params.hasOwnProperty("covL") && params.hasOwnProperty("logDet")) {
        this.covL = params.covL;
        this.logDet = params.logDet;
      } else {
        this.setCovariance(params.cov);
      }
      if (params.hasOwnProperty("invCov")) {
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
  setCovariance(cov) {
    this.cov = cov;
    this.covL = cov.chol();
    this.logDet = this.covL.diagonal().map(Math.log).sum();
  }
  getSample() {
    const z = Float64Array.build(MultivariateNormal.getNormal, this.dim, 1);
    return this.mean.add(this.covL.multiply(z));
  }
  logDensity(x) {
    const diff = this.mean.subtract(x);
    return (
      this.constant -
      this.logDet -
      0.5 * this.covL.fsolve_inplace(diff).norm2()
    );
  }
  gradLogDensity(x) {
    const diff = x.subtract(this.mean);
    // equivalent to this.cov.llt_solve(diff);
    return this.covL.bsolve_inplace(this.covL.fsolve_inplace(diff), { transpose: true }).scale(-1);
  }
  toString() {
    return (
      "mean: " +
      this.mean.transpose().toString() +
      "\ncov:  \n" +
      this.cov.toString() +
      "\ncovL: \n" +
      this.covL.toString() +
      "\nlogDet: " +
      this.logDet
    );
  }
  static getNormal() {
    let x, y, w;
    do {
      x = Math.random() * 2 - 1;
      y = Math.random() * 2 - 1;
      w = x * x + y * y;
    } while (w >= 1.0);
    return x * Math.sqrt((-2 * Math.log(w)) / w);
  }
  static getSample(dim) {
    const dist = new MultivariateNormal(zeros(dim), eye(dim));
    return dist.getSample();
  }
}
