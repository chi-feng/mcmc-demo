"use strict";

MCMC.registerAlgorithm("MicrocanonicalHamiltonianMC", {
  description: "Microcanonical Hamiltonian Monte Carlo",

  about: () => {
    window.open("https://arxiv.org/pdf/2212.08549.pdf");
  },

  init: (self) => {
    self.leapfrogSteps = 37;
    self.dt = 0.2;
  },

  reset: (self) => {
    self.chain = [MultivariateNormal.getSample(self.dim)];
  },

  attachUI: (self, folder) => {
    folder.add(self, "leapfrogSteps", 5, 120).step(1).name("Leapfrog Steps");
    folder.add(self, "dt", 0.05, 0.5).step(0.025).name("Leapfrog &Delta;t");
    folder.open();
  },

  step: (self, visualizer) => {
    
    var updateMomentum = function (eps, u, grad_logp) {
      const g_norm = Math.sqrt(grad_logp.norm2());
      const e = grad_logp.scale(-1.0 / g_norm);
      const ue = u.dot(e);
      const delta = eps * g_norm / (self.dim - 1);
      const zeta = Math.exp(-delta);
      const uu = e.scale((1 - zeta) * (1 + zeta + ue * (1 - zeta)) + 2 * zeta);
      return uu.scale(1.0 / Math.sqrt(uu.norm2()));
    }
    
    const q0 = self.chain.last();
    const p0 = MultivariateNormal.getSample(self.dim);
      
    // Normalize p0
    const p0Norm = Math.sqrt(p0.norm2());
    p0.scale(1.0 / p0Norm);

    // use leapfrog integration to find proposal
    const q = q0.copy();
    var p = p0.copy();
    const trajectory = [q.copy()];
    for (let i = 0; i < self.leapfrogSteps; i++) {
      p = updateMomentum(self.dt / 2, p, self.gradLogDensity(q));
      q.increment(p.scale(self.dt));
      p = updateMomentum(self.dt / 2, p, self.gradLogDensity(q));
      trajectory.push(q.copy());
    }

    // add integrated trajectory to visualizer animation queue
    visualizer.queue.push({
      type: "proposal",
      proposal: q,
      trajectory: trajectory,
      initialMomentum: p0,
    });

    // accept proposal always in MCHMC
    self.chain.push(q.copy());
    visualizer.queue.push({ type: "accept", proposal: q });

  },
});
