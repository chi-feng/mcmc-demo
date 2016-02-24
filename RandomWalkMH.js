'use strict';

MCMC.registerAlgorithm('RandomWalkMH', {

  description: 'Random walk Metropolis-Hastings',

  init: function(self) {
    self.sigma = 1;
  },

  reset: function(self) {
    self.chain = [zeros(self.dim)];
  },

  attachUI: function(self, folder) {
    folder.add(self, 'sigma', 0.05, 2).step(0.05).name('Proposal &sigma;');
    folder.open();
  },

  step: function(self, visualizer) {
    var lastIndex  = self.chain.length - 1;
    var proposalDist = new MultivariateNormal(self.chain[lastIndex], Float64Array.eye(self.dim).scale(self.sigma * self.sigma));
    var proposal = proposalDist.getSample();
    var logAcceptRatio = self.logDensity(proposal) - self.logDensity(self.chain[lastIndex]);
    visualizer.queue.push({type: 'proposal', proposal: proposal.copy(), proposalCov: proposalDist.cov.copy(), last: self.chain[lastIndex].copy()});
    if (Math.log(Math.random()) < logAcceptRatio) {
      self.chain.push(proposal);
      visualizer.queue.push({type: 'accept', proposal: proposal.copy(), last: self.chain[lastIndex].copy()});
    } else {
      self.chain.push(self.chain[lastIndex]);
      visualizer.queue.push({type: 'reject', proposal: proposal.copy(), last: self.chain[lastIndex].copy()});
    }
  },

});