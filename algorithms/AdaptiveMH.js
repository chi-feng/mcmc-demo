'use strict';

MCMC.registerAlgorithm('AdaptiveMH', {

  description: 'Adaptive Metropolis-Hastings',

  about: function() {
    window.open('http://projecteuclid.org/euclid.bj/1080222083');
  },

  init: function(self) {
    self.sigma = 1;
    self.adaptStride = 10;
    self.adaptProbability = 0.9;
  },

  reset: function(self) {
    self.mhDist = new MultivariateNormal(zeros(self.dim, 1), eye(self.dim));
    self.amDist = new MultivariateNormal(zeros(self.dim, 1), eye(self.dim));
    self.chainScatter = eye(self.dim, self.dim).scale(1e-6);
    self.chainSum = zeros(self.dim, 1);
    self.chain = [MultivariateNormal.getSample(self.dim)];
  },

  attachUI: function(self, folder) {
    folder.add(self, 'adaptProbability', 0, 1).step(0.05).name('Adapt Probability');
    folder.open();
  },

  step: function(self, visualizer) {
    var lastIndex = self.chain.length - 1;
    // update proposal covariance using rank-1 covariance update
    if (self.chain.length % self.adaptStride == 0) {
      for (var i = 0; i < self.adaptStride; ++i) {
        self.chainScatter.increment(Float64Array.outer(self.chain[lastIndex-i], self.chain[lastIndex-i]));
        self.chainSum.increment(self.chain[lastIndex-i]);
        var covariance = self.chainScatter.subtract(Float64Array.outer(self.chainSum, self.chainSum).scale(1.0 / self.chain.length)).scale(1.0 / self.chain.length);
        self.amDist.setCovariance(covariance.scale(2.38 * 2.38 / self.dim));
      }
    }
    var proposalDist = (Math.random() < self.adaptProbability) ? self.amDist : self.mhDist;
    var proposal = self.chain.last().add(proposalDist.getSample());
    var logAcceptRatio = self.logDensity(proposal) - self.logDensity(self.chain.last());
    visualizer.queue.push({type: 'proposal', proposal: proposal, proposalCov: proposalDist.cov});
    if (Math.random() < Math.exp(logAcceptRatio)) {
      self.chain.push(proposal);
      visualizer.queue.push({type: 'accept', proposal: proposal});
    } else {
      self.chain.push(self.chain.last());
      visualizer.queue.push({type: 'reject', proposal: proposal});
    }
  }

});