'use strict';

MCMC.registerAlgorithm('HMC', {

  description: 'Hamiltonian Monte Carlo',

  init: function(self) {
    self.leapfrogSteps  = 30;
    self.dt             = 0.1;
    self.proposalScale  = 1;
    self.reset(self);
  },

  reset: function(self) {
    self.accepted     = 0;
    self.chain        = [ Float64Array.zeros(self.dim, 1) ];
    self.logDensities = [ self.logDensity(self.chain[0]) ];
    self.gradients    = [ self.gradLogDensity(self.chain[0]) ];
  },

  step: function(self, visualizer) {

    var lastIndex   = self.chain.length - 1;
    var last        = self.chain[lastIndex];

    var logDensity  = self.logDensities[lastIndex];
    var gradient    = self.gradients[lastIndex];

    var initialMomentum = Float64Array.build(NormalDistribution.getSample, self.dim, 1).scale(self.proposalScale);;
    var momentum    = initialMomentum.copy();
    var hamiltonian = logDensity + momentum.norm2() / 2;

    var proposal = last.copy();
    self.proposalTrajectory = [ proposal.copy() ];
    self.momentumTrajectory = [ momentum.copy() ];
    for (var i = 0; i < self.leapfrogSteps; ++i) {
      momentum.increment(gradient.scale(self.dt / 2));
      proposal.increment(momentum.scale(self.dt));
      gradient = self.gradLogDensity(proposal);
      momentum.increment(gradient.scale(self.dt / 2));
      self.proposalTrajectory.push(proposal.copy());
      self.momentumTrajectory.push(momentum.copy());
    }

    if (visualizer.animateProposal) {
      visualizer.queue.push({type: 'hmc-animation-start', last: last, initialMomentum: initialMomentum});
      for (var i = 0; i < self.leapfrogSteps+1; ++i)
        visualizer.queue.push({type: 'hmc-animation', proposal: self.proposalTrajectory[i], index: i, last: last, initialMomentum: initialMomentum});
      visualizer.queue.push({type: 'hmc-animation-end'});
    } else {
      visualizer.queue.push({type: 'proposal', proposal: proposal, last: last, initialMomentum: initialMomentum});
    }

    var newLogDensity = self.logDensity(proposal);
    var newHamiltonian = newLogDensity + momentum.norm2() / 2;
    var delta = hamiltonian - newHamiltonian;

    if (delta < 0 || Math.random() < Math.exp(delta)) {
      self.accepted++;
      self.chain.push(proposal.copy());
      self.logDensities.push(newLogDensity);
      self.gradients.push(gradient.copy());
      visualizer.queue.push({type: 'accept', proposal: proposal, last: last});
    } else {
      self.chain.push(last.copy());
      self.logDensities.push(self.logDensities[lastIndex]);
      self.gradients.push(self.gradients[lastIndex].copy());
      visualizer.queue.push({type: 'reject', proposal: proposal, last: last});
    }
  },

  attachUI: function(self, folder) {
    folder.add(self, 'leapfrogSteps', 5, 50).step(1).name('Leapfrog Steps');
    folder.add(self, 'dt', 0.02, 0.3).step(0.02).name('Leapfrog &Delta;t');
    folder.add(self, 'proposalScale', 0.1, 5).name('Momentum Scale');
    folder.open();
  }
}
);