"use strict";

/**
 * Gibbs Sampling
 *
 * A component-wise MCMC method that samples each dimension in turn from its
 * full conditional distribution. Uses numerical integration to approximate
 * the conditionals.
 *
 * @see https://en.wikipedia.org/wiki/Gibbs_sampling
 */
MCMC.registerAlgorithm("GibbsSampling", {
  description: "Gibbs Sampling",

  about: () => {
    window.open("https://en.wikipedia.org/wiki/Gibbs_sampling");
  },

  init: (self) => {},

  reset: (self) => {
    self.chain = [MultivariateNormal.getSample(self.dim)];
  },

  attachUI: (self, folder) => {
    folder.open();
  },

  step: (self, visualizer) => {
    const sampleFullConditional = (logDensity, point, index) => {
      point = point.copy();

      // Add some noise to avoid grid pattern in samples; use the target's extents
      const width = self.xmax - self.xmin;
      const Xs = linspace(
        self.xmin - (Math.random() * width) / 256,
        self.xmax + (Math.random() * width) / 256,
        256
      );
      const densities = [];
      let marginal = 0;

      for (let i = 0; i < 256; i++) {
        point[index] = Xs[i];
        const density = Math.exp(logDensity(point));
        densities.push(density);
        marginal += density;
      }

      const threshold = marginal * Math.random();
      let sum = 0;
      let i = 0;

      while (sum < threshold) {
        sum += densities[i++];
      }

      point[index] = Xs[i - 1];
      return point;
    };

    let last = self.chain.last();
    const trajectory = [last.copy()];

    for (let i = 0; i < 2; i++) {
      last = sampleFullConditional(self.logDensity, last, i);
      trajectory.push(last);
    }

    visualizer.queue.push({
      type: "proposal",
      proposal: last,
      trajectory: trajectory,
    });
    visualizer.queue.push({ type: "accept", proposal: last });
    self.chain.push(last);
  },
});
