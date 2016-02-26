'use strict';

MCMC.registerAlgorithm('RandomWalkMH', {

  description: 'Random walk Metropolis-Hastings',

  about: function() {
    window.open('https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm');
  },

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
    var proposalDist    = new MultivariateNormal(self.chain.last(), eye(self.dim).scale(self.sigma * self.sigma));
    var proposal        = proposalDist.getSample();
    var logAcceptRatio  = self.logDensity(proposal) - self.logDensity(self.chain.last());
    visualizer.queue.push({type: 'proposal', proposal: proposal, proposalCov: proposalDist.cov});
    if (Math.random() < Math.exp(logAcceptRatio)) {
      self.chain.push(proposal);
      visualizer.queue.push({type: 'accept', proposal: proposal});
    } else {
      self.chain.push(self.chain.last());
      visualizer.queue.push({type: 'reject', proposal: proposal});
    }
  },

});