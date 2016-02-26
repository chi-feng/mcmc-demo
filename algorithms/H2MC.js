'use strict';

MCMC.registerAlgorithm('H2MC', {

  description: 'Hessian-Hamiltonian Monte Carlo',

  about: function() {
    window.open('https://people.csail.mit.edu/tzumao/h2mc/');
  },

  init: function(self) {
    self.sigma = 1.0;

    self.L = Math.PI / 2;
    self.posScaleFactor = Math.pow(0.5 * (Math.exp(self.L) - Math.exp(-self.L)), 2);
    self.posOffsetFactor = 0.5 * (Math.exp(self.L) + Math.exp(-self.L) - 1);
    self.negScaleFactor = Math.sin(self.L) * Math.sin(self.L);
    self.negOffsetFactor = -(Math.cos(self.L) - 1);
    self.epsilon = 1e-8;

    self.computeGaussian = function(grad, hess) {
      var dim = self.dim;

      var sigma = Float64Array.constant(self.sigma, dim);
      var sigmaMax = sigma.maxCoeff();
      var sigmaSq = sigma.cwiseProduct(sigma);
      var invSigmaSq = sigmaSq.cwiseInverse();
      if (hess.norm() < 0.5 / (sigmaMax * sigmaMax)) {
        return {mean: zeros(dim), covL: sigma.asDiagonal(), invCov: invSigmaSq, logDet: invSigmaSq.map(Math.log).sum()};
      }

      var eigenSolver = hess.jacobiRotation({maxIter:10, tolerance: self.epsilon});
      var hEigenvector = eigenSolver.eigenvectors;
      var hEigenvalues = eigenSolver.eigenvalues;

      var eigenBuff = zeros(dim, 1);
      var offsetBuff = zeros(dim, 1);
      var postInvCovEigenvalues = zeros(dim, 1);

      for (var i = 0; i < dim; i++) {
        if (Math.abs(hEigenvalues[i]) > self.epsilon) {
          eigenBuff[i] = 1.0 / Math.abs(hEigenvalues[i]);
        } else {
          eigenBuff[i] = 0.0;
        }
      }

      offsetBuff = eigenBuff.asDiagonal().multiply(hEigenvector.transpose().multiply(grad));
      for (var i = 0; i < dim; i++) {
        var scale = 1.0;
        var offset = 0.0;
        if (Math.abs(hEigenvalues[i]) > self.epsilon) {
          offset = offsetBuff[i];
          if (hEigenvalues[i] > 0.0) {
            scale = self.posScaleFactor;
            offset *= self.posOffsetFactor;
          } else {
            scale = self.negScaleFactor;
            offset *= self.negOffsetFactor;
          }
        } else {
          scale = self.L * self.L;
          offset = 0.5 * offsetBuff[i] * self.L * self.L;
        }
        eigenBuff[i] *= scale;
        if (eigenBuff[i] > self.epsilon) {
          eigenBuff[i] = 1.0 / eigenBuff[i];
        } else {
          eigenBuff[i] = 0.0;
        }
        offsetBuff[i] = offset;
      }

      postInvCovEigenvalues = eigenBuff.add(invSigmaSq);
      var gaussian = {
        invCov: hEigenvector.multiply(postInvCovEigenvalues.asDiagonal().multiply(hEigenvector.transpose())),
        mean: hEigenvector.multiply(eigenBuff.cwiseQuotient(postInvCovEigenvalues).asDiagonal().multiply(offsetBuff)),
        covL: hEigenvector.multiply(postInvCovEigenvalues.cwiseInverse().cwiseSqrt().asDiagonal()),
        logDet: postInvCovEigenvalues.map(Math.log).sum()
      };
      return gaussian;
    };
  },

  reset: function(self) {
    self.chain = [ MultivariateNormal.getSample(self.dim) ];
    self.lastGaussian = { mean: zeros(self.dim, 1), covL: eye(self.dim, self.dim), logDet: 0.0 };
  },

  attachUI: function(self, folder) {
    folder.add(self, 'sigma', 0.1, 5).step(0.1).name('&sigma;');
    folder.open();
  },

  step: function(self, visualizer) {
    var x = self.chain.last();
    var gaussian = self.computeGaussian(self.gradLogDensity(x), self.hessLogDensity(x));
    gaussian.mean.increment(x);
    var proposalDist = new MultivariateNormal(gaussian);
    var y = proposalDist.getSample();
    var z = y.subtract(x);
    var Phi_y = proposalDist;
    var Phi_x = new MultivariateNormal(self.lastGaussian);
    var logAcceptRatio = self.logDensity(y) - self.logDensity(x); // + Phi_y.logDensity(x) - Phi_x.logDensity(y);
    visualizer.queue.push({type: 'proposal', proposal: y, proposalCov: proposalDist.cov, proposalMean: proposalDist.mean, logAcceptRatio: logAcceptRatio});
    if (Math.random() < Math.exp(logAcceptRatio)) {
      self.chain.push(y.copy());
      visualizer.queue.push({type: 'accept', proposal: y});
      self.lastGaussian = gaussian;
    } else {
      self.chain.push(x.copy());
      visualizer.queue.push({type: 'reject', proposal: y});
    }
  }

});