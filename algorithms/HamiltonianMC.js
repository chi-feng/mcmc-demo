"use strict";

/**
 * Hamiltonian Monte Carlo (HMC) Algorithm
 *
 * Uses Hamiltonian dynamics to propose distant states while maintaining high
 * acceptance rates. Simulates a physical system where the negative log density
 * acts as potential energy and momentum provides kinetic energy.
 *
 * @see https://en.wikipedia.org/wiki/Hybrid_Monte_Carlo
 */
MCMC.registerAlgorithm("HamiltonianMC", {
  description: "Hamiltonian Monte Carlo",

  about: () => {
    window.open("https://en.wikipedia.org/wiki/Hybrid_Monte_Carlo");
  },

  init: (self) => {
    self.leapfrogSteps = 37;
    self.dt = 0.1;
  },

  reset: (self) => {
    self.chain = [MultivariateNormal.getSample(self.dim)];
  },

  attachUI: (self, folder) => {
    folder.add(self, "leapfrogSteps", 5, 100).step(1).name("Leapfrog Steps");
    folder.add(self, "dt", 0.05, 0.5).step(0.025).name("Leapfrog &Delta;t");
    folder.open();
  },

  step: (self, visualizer) => {
    const q0 = self.chain.last();
    const p0 = MultivariateNormal.getSample(self.dim);

    // Use leapfrog integration to find proposal
    const q = q0.copy();
    const p = p0.copy();
    const trajectory = [q.copy()];

    for (let i = 0; i < self.leapfrogSteps; i++) {
      p.increment(self.gradLogDensity(q).scale(self.dt / 2));
      q.increment(p.scale(self.dt));
      p.increment(self.gradLogDensity(q).scale(self.dt / 2));
      trajectory.push(q.copy());
    }

    // Add integrated trajectory to visualizer animation queue
    visualizer.queue.push({
      type: "proposal",
      proposal: q,
      trajectory: trajectory,
      initialMomentum: p0,
    });

    // Calculate acceptance ratio
    const H0 = -self.logDensity(q0) + p0.norm2() / 2;
    const H = -self.logDensity(q) + p.norm2() / 2;
    const logAcceptRatio = -H + H0;

    // Accept or reject proposal
    if (Math.random() < Math.exp(logAcceptRatio)) {
      self.chain.push(q.copy());
      visualizer.queue.push({ type: "accept", proposal: q });
    } else {
      self.chain.push(q0.copy());
      visualizer.queue.push({ type: "reject", proposal: q });
    }
  },
});
