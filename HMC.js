'use strict';

MCMC.registerAlgorithm('HMC', {

  description: 'Hamiltonian Monte Carlo',

  init: function(self) {
    self.leapfrogSteps  = 20;
    self.dt             = 0.1;
    self.reset(self);
  },

  reset: function(self) {
    self.chain        = [ Float64Array.zeros(self.dim, 1) ];
  },

  step: function(self, visualizer) {

    var lastIndex   = self.chain.length - 1;
    var last        = self.chain[lastIndex];

    var logDensity  = self.logDensity(last);
    var gradient    = self.gradLogDensity(last);

    var initialMomentum = Float64Array.build(MultivariateNormal.getNormal, self.dim, 1);
    var momentum    = initialMomentum.copy();
    var hamiltonian = logDensity + momentum.norm2() / 2;

    var proposal = last.copy();
    self.proposalTrajectory = [ proposal.copy() ];
    for (var i = 0; i < self.leapfrogSteps; ++i) {
      momentum.increment(gradient.scale(self.dt / 2));
      proposal.increment(momentum.scale(self.dt));
      gradient = self.gradLogDensity(proposal);
      momentum.increment(gradient.scale(self.dt / 2));
      self.proposalTrajectory.push(proposal.copy());
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
      self.chain.push(proposal.copy());
      visualizer.queue.push({type: 'accept', proposal: proposal, last: last});
    } else {
      self.chain.push(last.copy());
      visualizer.queue.push({type: 'reject', proposal: proposal, last: last});
    }
  },

  attachUI: function(self, folder) {
    folder.add(self, 'leapfrogSteps', 5, 100).step(1).name('Leapfrog Steps');
    folder.add(self, 'dt', 0.05, 0.5).step(0.025).name('Leapfrog &Delta;t');
    folder.open();
  }
}
);