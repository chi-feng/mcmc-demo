'use strict';

MCMC.registerAlgorithm('MALA', {

  description: 'Metropolis-adjusted Langevin algorithm',

  init: function(self) {
    self.sigma = 1;
    self.reset(self);
  },

  reset: function(self) {
    self.chain = [Float64Array.zeros(self.dim, 1)];
  },

  step: function(self, visualizer) {
    var lastIndex   = self.chain.length - 1;
    var gradient    = self.gradLogDensity(self.chain[lastIndex]);
    var Z           = Float64Array.build(MultivariateNormal.getNormal, self.dim, 1).scale(self.sigma);
    var proposal    = self.chain[lastIndex].add(Z).add(gradient.scale(self.sigma * self.sigma / 2));

    var logProposalDensity = function(x, y) {
      return -y.subtract(x).subtract(self.gradLogDensity(x).scale(self.sigma * self.sigma / 2)).norm2() / (2 * self.sigma * self.sigma) - self.dim / 2 * Math.log(2 * Math.PI * self.sigma * self.sigma);
    };

    var logNumerator = self.logDensity(proposal) + logProposalDensity(proposal, self.chain[lastIndex]);
    var logDenominator = self.logDensity(self.chain[lastIndex]) + logProposalDensity(self.chain[lastIndex], proposal);
    var logAcceptRatio = logNumerator - logDenominator;

    visualizer.queue.push({
      type:        'mala-proposal',
      proposal:    proposal.copy(),
      Z:           Z.copy(),
      gradient:    gradient.scale(self.sigma * self.sigma / 2),
      last:        self.chain[lastIndex].copy(),
      acceptRatio: Math.exp(logAcceptRatio)
    });

    if (Math.log(Math.random()) < logAcceptRatio) {
      self.chain.push(proposal);
      visualizer.queue.push({type: 'accept', proposal: proposal.copy(), last: this.chain[lastIndex].copy()});
    } else {
      self.chain.push(self.chain[lastIndex]);
      visualizer.queue.push({type: 'reject', proposal: proposal.copy(), last: this.chain[lastIndex].copy()});
    }
  },

  attachUI: function(self, folder) {
    folder.add(self, 'sigma', 0.05, 2).step(0.05).name('Proposal &sigma;');
    folder.open();
  }
});