"use strict";

/**
 * Microcanonical Hamiltonian Monte Carlo (MCHMC)
 *
 * A variant of HMC that operates on the microcanonical ensemble, where the
 * total energy is conserved exactly. Uses a modified momentum update that
 * preserves the norm of the velocity vector.
 *
 * @see https://arxiv.org/pdf/2212.08549.pdf
 */
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
    // Momentum update from Robnik et al. (arXiv:2212.08549), eq. 16:
    // the tangential component 2*zeta*u must survive alongside the e term.
    // The reference negates the energy gradient, which is already the negative
    // log-density gradient, so e points uphill in log density: e = +grad/|grad|
    const updateMomentum = (eps, u, grad_logp) => {
      const g_norm = Math.sqrt(grad_logp.norm2());
      if (g_norm === 0) return u;
      const e = grad_logp.scale(1.0 / g_norm);
      const ue = u.dot(e);
      const delta = (eps * g_norm) / (self.dim - 1);
      const zeta = Math.exp(-delta);
      const uu = u.scale(2 * zeta).add(e.scale((1 - zeta) * (1 + zeta + ue * (1 - zeta))));
      return uu.scale(1.0 / Math.sqrt(uu.norm2()));
    };

    const q0 = self.chain.last();
    // MCHMC uses a unit velocity; scale returns a copy, so assign the result
    const p0 = MultivariateNormal.getSample(self.dim);
    const u0 = p0.scale(1.0 / Math.sqrt(p0.norm2()));

    // Use leapfrog integration to find proposal
    const q = q0.copy();
    let p = u0.copy();
    const trajectory = [q.copy()];

    for (let i = 0; i < self.leapfrogSteps; i++) {
      p = updateMomentum(self.dt / 2, p, self.gradLogDensity(q));
      q.increment(p.scale(self.dt));
      p = updateMomentum(self.dt / 2, p, self.gradLogDensity(q));
      trajectory.push(q.copy());
    }

    // Add integrated trajectory to visualizer animation queue
    visualizer.queue.push({
      type: "proposal",
      proposal: q,
      trajectory: trajectory,
      initialMomentum: u0,
    });

    // Accept proposal always in MCHMC
    self.chain.push(q.copy());
    visualizer.queue.push({ type: "accept", proposal: q });
  },
});
