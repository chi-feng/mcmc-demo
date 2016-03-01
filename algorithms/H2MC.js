'use strict';

MCMC.registerAlgorithm('H2MC', {

  description: 'Hessian-Hamiltonian Monte Carlo',

  about: function() {
    window.open('https://people.csail.mit.edu/tzumao/h2mc/');
  },

  init: function(self) {
    self.sigma = 1.0;
    self.L = Math.PI / 2;
    self.epsilon = 1e-8;
    /**
     * Based on C++ implementation by Tzu-Mao Li
     * https://github.com/BachiLi/dpt
     */
    self.computeGaussian = function(x, grad, hess) {
      self.posScaleFactor  = Math.pow(0.5 * (Math.exp(self.L) - Math.exp(-self.L)), 2);
      self.posOffsetFactor = 0.5 * (Math.exp(self.L) + Math.exp(-self.L) - 1);
      self.negScaleFactor  = Math.sin(self.L) * Math.sin(self.L);
      self.negOffsetFactor = -(Math.cos(self.L) - 1);
      var dim = self.dim;
      var sigma = Float64Array.constant(self.sigma, dim);
      var sigmaMax = sigma.maxCoeff();
      var sigmaSq = sigma.cwiseProduct(sigma);
      var invSigmaSq = sigmaSq.cwiseInverse();
      if (hess.norm() < 0.5 / (sigmaMax * sigmaMax)) {
        return new MultivariateNormal({offset: zeros(dim), mean: x, covL: sigma.asDiagonal(), invCov: invSigmaSq, logDet: invSigmaSq.map(Math.log).sum()});
      }
      var eigenSolver  = hess.jacobiRotation({maxIter:10, tolerance: self.epsilon});
      var hEigenvector = eigenSolver.eigenvectors;
      var hEigenvalues = eigenSolver.eigenvalues;
      var eigenBuff    = zeros(dim, 1);
      var offsetBuff   = zeros(dim, 1);
      var postInvCovEigenvalues = zeros(dim, 1);
      for (var i = 0; i < dim; i++) {
        eigenBuff[i] = (Math.abs(hEigenvalues[i]) > self.epsilon) ? 1.0 / Math.abs(hEigenvalues[i]) : 0;
      }
      offsetBuff = eigenBuff.asDiagonal().multiply(hEigenvector.transpose().multiply(grad));
      for (var i = 0; i < dim; i++) {
        var scale = 1.0;
        var offset = 0.0;
        if (Math.abs(hEigenvalues[i]) > self.epsilon) {
          offset = offsetBuff[i];
          scale = (hEigenvalues[i] > 0.0) ? self.posScaleFactor : self.negScaleFactor;
          offset *= (hEigenvalues[i] > 0.0) ? self.posOffsetFactor : self.negOffsetFactor;
        } else {
          scale = self.L * self.L;
          offset = 0.5 * offsetBuff[i] * self.L * self.L;
        }
        eigenBuff[i] *= scale;
        eigenBuff[i] = (eigenBuff[i] > self.epsilon) ? 1.0 / eigenBuff[i] : 0.0;
        offsetBuff[i] = offset;
      }
      postInvCovEigenvalues = eigenBuff.add(invSigmaSq);
      var gaussianParams = {
        invCov: hEigenvector.multiply(postInvCovEigenvalues.asDiagonal().multiply(hEigenvector.transpose())),
        offset: hEigenvector.multiply(eigenBuff.cwiseQuotient(postInvCovEigenvalues).asDiagonal().multiply(offsetBuff)),
        covL:   hEigenvector.multiply(postInvCovEigenvalues.cwiseInverse().cwiseSqrt().asDiagonal()),
        logDet: postInvCovEigenvalues.map(Math.log).sum()
      };
      gaussianParams.mean = x.add(gaussianParams.offset);
      return new MultivariateNormal(gaussianParams);
    };
  },

  reset: function(self) {
    self.chain = [ MultivariateNormal.getSample(self.dim) ];
  },

  attachUI: function(self, folder) {
    folder.add(self, 'sigma', 0.1, 5).step(0.1).name('&sigma;');
    folder.add(self, 'L', 0.1, 6.28).step(0.1).name('L');
    folder.open();
  },

  step: function(self, visualizer) {
    var x = self.chain.last();
    var proposalDist = self.computeGaussian(x, self.gradLogDensity(x), self.hessLogDensity(x));
    var y = proposalDist.getSample();
    visualizer.queue.push({ type: 'proposal', proposal: y, proposalMean: proposalDist.mean, proposalCov: proposalDist.cov });
    if (Math.random() < Math.exp(self.logDensity(y) - self.logDensity(x))) {
      self.chain.push(y.copy());
      visualizer.queue.push({type: 'accept', proposal: y});
    } else {
      self.chain.push(x.copy());
      visualizer.queue.push({type: 'reject', proposal: y});
    }
  }

});