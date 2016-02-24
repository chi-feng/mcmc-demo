'use strict';

MCMC.registerAlgorithm('MALA', {

  description: 'Metropolis-adjusted Langevin algorithm',

  init: function(self) {
    self.sigma = 1;
  },

  reset: function(self) {
    self.chain = [MultivariateNormal.getSample(self.dim)];
  },

  attachUI: function(self, folder) {
    folder.add(self, 'sigma', 0.05, 2).step(0.05).name('Proposal &sigma;');
    folder.open();
  },

  step: function(self, visualizer) {
    var gradient    = self.gradLogDensity(self.chain.last());
    var Z           = Float64Array.build(MultivariateNormal.getNormal, self.dim, 1).scale(self.sigma);
    var proposal    = self.chain.last().add(Z).add(gradient.scale(self.sigma * self.sigma / 2));

    var logProposalDensity = function(x, y) {
      return -y.subtract(x).subtract(self.gradLogDensity(x).scale(self.sigma * self.sigma / 2)).norm2() / (2 * self.sigma * self.sigma) - self.dim / 2 * Math.log(2 * Math.PI * self.sigma * self.sigma);
    };

    var logNumerator = self.logDensity(proposal) + logProposalDensity(proposal, self.chain.last());
    var logDenominator = self.logDensity(self.chain.last()) + logProposalDensity(self.chain.last(), proposal);
    var logAcceptRatio = logNumerator - logDenominator;

    visualizer.queue.push({
      type:        'proposal',
      proposal:    proposal.copy(),
      Z:           Z.copy(),
      gradient:    gradient.scale(self.sigma * self.sigma / 2),
    });

    if (Math.random() < Math.exp(logAcceptRatio)) {
      self.chain.push(proposal);
      visualizer.queue.push({type: 'accept', proposal: proposal.copy()});
    } else {
      self.chain.push(self.chain.last());
      visualizer.queue.push({type: 'reject', proposal: proposal.copy()});
    }
  }
});