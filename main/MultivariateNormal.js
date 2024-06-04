"use strict";

/**
 * Class representing a multivariate normal distribution.
 */
class MultivariateNormal {
  /**
   * Create a MultivariateNormal distribution.
   * @param {Object|Matrix} mean - Mean vector or parameters object.
   * @param {Matrix} [cov] - Covariance matrix (if mean is a vector).
   */
  constructor(mean, cov) {
    if (mean.hasOwnProperty("mean")) {
      // Parameter object case
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
      // Mean and covariance matrix case
      this.mean = mean;
      this.dim = mean.rows;
      this.setCovariance(cov);
    }
    this.constant = -0.5 * Math.log(2.0 * Math.PI) * this.dim;
    this.logDet = this.covL.diagonal().map(Math.log).sum();
  }

  /**
   * Set the covariance matrix and compute its Cholesky decomposition and log determinant.
   * @param {Matrix} cov - Covariance matrix.
   */
  setCovariance(cov) {
    this.cov = cov;
    this.covL = cov.chol();
    this.logDet = this.covL.diagonal().map(Math.log).sum();
  }

  /**
   * Generate a random sample from the distribution.
   * @returns {Matrix} Random sample vector.
   */
  getSample() {
    const z = Float64Array.build(MultivariateNormal.getNormal, this.dim, 1);
    return this.mean.add(this.covL.multiply(z));
  }

  /**
   * Compute the log density of a point.
   * @param {Matrix} x - Point to evaluate.
   * @returns {number} Log density at point x.
   */
  logDensity(x) {
    const diff = this.mean.subtract(x);
    return (
      this.constant -
      this.logDet -
      0.5 *
        this.covL
          .bsolve_inplace(this.covL.fsolve_inplace(diff), { transpose: true })
          .norm2()
    );
  }

  /**
   * Compute the gradient of the log density at a point.
   * @param {Matrix} x - Point to evaluate.
   * @returns {Matrix} Gradient of the log density at point x.
   */
  gradLogDensity(x) {
    const diff = x.subtract(this.mean);
    return this.covL
      .bsolve_inplace(this.covL.fsolve_inplace(diff), { transpose: true })
      .scale(-1);
  }

  /**
   * Get a string representation of the distribution.
   * @returns {string} String representation.
   */
  toString() {
    return (
      `mean: ${this.mean.transpose().toString()}\n` +
      `cov:  \n${this.cov.toString()}\n` +
      `covL: \n${this.covL.toString()}\n` +
      `logDet: ${this.logDet}`
    );
  }

  /**
   * Generate a standard normal random variable using the Box-Muller transform.
   * @returns {number} Standard normal random variable.
   */
  static getNormal() {
    let x, y, w;
    do {
      x = Math.random() * 2 - 1;
      y = Math.random() * 2 - 1;
      w = x * x + y * y;
    } while (w >= 1.0);
    return x * Math.sqrt((-2 * Math.log(w)) / w);
  }

  /**
   * Generate a random sample from a standard normal distribution of given dimension.
   * @param {number} dim - Dimension of the sample.
   * @returns {Matrix} Random sample vector.
   */
  static getSample(dim) {
    const dist = new MultivariateNormal(zeros(dim), eye(dim));
    return dist.getSample();
  }
}
