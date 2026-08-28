"use strict";

/**
 * Stein Variational Gradient Descent (SVGD)
 *
 * A particle-based variational inference method that iteratively transports
 * a set of particles to approximate the target distribution. Uses a
 * reproducing kernel to balance between fitting the target and maintaining
 * particle diversity.
 *
 * @see http://www.cs.dartmouth.edu/~dartml/project.html?p=vgd
 */
MCMC.registerAlgorithm("SVGD", {
  description: "Stein Variational Gradient Descent",

  about: () => {
    window.open("http://www.cs.dartmouth.edu/~dartml/project.html?p=vgd");
  },

  init: (self) => {
    self.chain = [];
    self.n = 200; // number of particles
    self.epsilon = 0.01; // step size
    self.h = 0.15; // bandwidth
    self.use_median = true;
    self.use_adagrad = true;
    self.alpha = 0.9;
    self.fudge_factor = 1e-2;
    self.iter = 0;
    self.reset(self);
  },

  reset: (self) => {
    // Initialize chain with samples from standard normal
    self.chain = [];
    self.gradx = [];
    self.historical_grad = [];
    self.gradLogDensities = [];
    self.iter = 0;

    for (let i = 0; i < self.n; i++) {
      self.chain.push(MultivariateNormal.getSample(self.dim));
      self.gradx.push(Float64Array.zeros(self.dim, 1));
      self.historical_grad.push(Float64Array.zeros(self.dim, 1));
      self.gradLogDensities.push(0);
    }
  },

  attachUI: (self, folder) => {
    folder.add(self, "use_median").name("Median heuristic").listen();
    folder
      .add(self, "h", 0.05, 2)
      .step(0.05)
      .name("bandwidth")
      .listen()
      .onChange(() => {
        self.use_median = false;
      });
    folder.add(self, "use_adagrad").name("Adagrad");
    folder.add(self, "epsilon", 0.001, 0.1).step(0.001).name("stepsize");
    folder.add(self, "n", 10, 400).step(1).name("numParticles");
    folder.open();
  },

  step: (self, visualizer) => {
    // Resize samples appropriately
    if (self.n > self.chain.length) {
      // Capture the deficit first: each push grows chain.length and would shrink the bound
      const missing = self.n - self.chain.length;
      for (let i = 0; i < missing; i++) {
        self.chain.push(MultivariateNormal.getSample(self.dim));
        self.gradx.push(Float64Array.zeros(self.dim, 1));
        self.historical_grad.push(Float64Array.zeros(self.dim, 1));
        self.gradLogDensities.push(0);
      }
    } else if (self.n < self.chain.length) {
      self.chain = self.chain.slice(0, self.n);
      self.gradx = self.gradx.slice(0, self.n);
      self.historical_grad = self.historical_grad.slice(0, self.n);
      self.gradLogDensities = self.gradLogDensities.slice(0, self.n);
    }

    const n = self.chain.length;

    // Precompute log densities
    for (let i = 0; i < n; i++) {
      self.gradLogDensities[i] = self.gradLogDensity(self.chain[i]);
      for (let k = 0; k < self.dim; k++) {
        self.gradx[i][k] = 0;
      }
    }

    // Pairwise distances trick
    const dist2 = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < i; j++) {
        let delta = 0;
        for (let k = 0; k < self.dim; k++) {
          delta += Math.pow(self.chain[i][k] - self.chain[j][k], 2);
        }
        dist2[i * n + j] = delta;
        dist2[j * n + i] = delta;
      }
    }

    if (self.use_median) {
      const dist2copy = new Float64Array(dist2);
      dist2copy.sort();
      const median = dist2copy[Math.floor(dist2copy.length / 2)];
      self.h = median / Math.log(n);
    }

    // Compute gradient
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const rbf = Math.exp(-dist2[i * n + j] / self.h);
        for (let k = 0; k < self.dim; k++) {
          const grad_rbf = ((self.chain[i][k] - self.chain[j][k]) * 2 * rbf) / self.h;
          self.gradx[i][k] += self.gradLogDensities[j][k] * rbf + grad_rbf;
        }
      }
      for (let k = 0; k < self.dim; k++) {
        self.gradx[i][k] /= n;
      }
    }

    // Adagrad
    if (self.use_adagrad) {
      for (let i = 0; i < n; i++) {
        for (let k = 0; k < self.dim; k++) {
          self.historical_grad[i][k] =
            self.alpha * self.historical_grad[i][k] + (1 - self.alpha) * Math.pow(self.gradx[i][k], 2);
        }
      }
      for (let i = 0; i < n; i++) {
        for (let k = 0; k < self.dim; k++) {
          self.gradx[i][k] /= self.fudge_factor + Math.sqrt(self.historical_grad[i][k]);
        }
      }
    }

    for (let i = 0; i < n; i++) {
      for (let k = 0; k < self.dim; k++) {
        self.gradx[i][k] *= self.epsilon;
      }
    }

    visualizer.queue.push({
      type: "svgd-step",
      x: self.chain,
      gradx: self.gradx,
      h: self.h,
    });

    // Update particles
    for (let i = 0; i < n; i++) {
      self.chain[i].increment(self.gradx[i]);
    }

    self.iter++;
  },
});
