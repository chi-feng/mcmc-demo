"use strict";

/**
 * Differential Evolution MCMC (Z variant)
 *
 * A population-based MCMC method that uses the difference between randomly
 * selected past states to generate proposals. The "Z" variant samples from
 * the entire chain history rather than a fixed population.
 *
 * @see https://link.springer.com/article/10.1007/s11222-008-9104-9
 */
MCMC.registerAlgorithm("DE-MCMC-Z", {
  description: "Differential Evolution Metropolis (Z)",

  about: () => {
    window.open("https://link.springer.com/article/10.1007/s11222-008-9104-9");
  },

  init: (self) => {
    self.lambda = 2.38 / Math.sqrt(self.dim);
    self.scaling = 0.1;
  },

  reset: (self) => {
    self.chain = [MultivariateNormal.getSample(self.dim)];
  },

  attachUI: (self, folder) => {
    folder.add(self, "lambda", 0.1, 3).step(0.1).name("Lambda &lambda;");
    folder.add(self, "scaling", 0.001, 0.2).step(0.01).name("Scaling &epsilon;");
    folder.open();
  },

  step: (self, visualizer) => {
    const N = self.chain.length;
    let iz1 = Math.floor(Math.random() * N);
    let iz2 = Math.floor(Math.random() * N);

    if (N > 1) {
      while (iz2 === iz1) {
        iz2 = Math.floor(Math.random() * N);
      }
    }

    const q0 = self.chain.last();
    const z1 = self.chain[iz1];
    const z2 = self.chain[iz2];

    const epsilonDist = new MultivariateNormal(zeros(self.dim, 1), eye(self.dim).scale(self.scaling * self.scaling));
    const epsilon = epsilonDist.getSample();
    const vec = z2.subtract(z1);
    const proposal = q0.add(vec.scale(self.lambda)).add(epsilon);

    const logAcceptRatio = self.logDensity(proposal) - self.logDensity(self.chain.last());

    visualizer.queue.push({
      type: "proposal",
      proposal: proposal,
      inspiration: {
        from: z1,
        to: z2,
      },
    });

    if (Math.random() < Math.exp(logAcceptRatio)) {
      self.chain.push(proposal);
      visualizer.queue.push({ type: "accept", proposal: proposal });
    } else {
      self.chain.push(q0);
      visualizer.queue.push({ type: "reject", proposal: proposal });
    }
  },
});
