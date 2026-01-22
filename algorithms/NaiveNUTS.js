"use strict";

/**
 * Naive No-U-Turn Sampler (NUTS)
 *
 * Automatically tunes the number of leapfrog steps by building a binary tree
 * of states and stopping when the trajectory starts to double back (U-turn).
 * This is the naive version from Algorithm 2 of the NUTS paper.
 *
 * @see http://arxiv.org/abs/1111.4246
 */
MCMC.registerAlgorithm("NaiveNUTS", {
  description: "Naive No-U-Turn Sampler",

  about: () => {
    window.open("http://arxiv.org/abs/1111.4246");
  },

  init: (self) => {
    self.dt = 0.1;
    self.Delta_max = 1000;
  },

  reset: (self) => {
    self.chain = [MultivariateNormal.getSample(self.dim)];
  },

  attachUI: (self, folder) => {
    folder.add(self, "dt", 0.025, 0.6).step(0.025).name("Leapfrog &Delta;t");
    folder.open();
  },

  // Notation adopted from http://arxiv.org/pdf/1111.4246v1.pdf
  step: (self, visualizer) => {
    const trajectory = [];

    /**
     * BuildTree from Algorithm 2: Naive No-U-Turn Sampler
     * @param {Matrix} q - position
     * @param {Matrix} p - momentum
     * @param {number} u - Uniform([0, exp{L(q0)-½p0·p0}])
     * @param {number} v - direction of integration (±1)
     * @param {number} j - depth of tree/recursion
     * @returns {Object} { q_minus, p_minus, q_plus, p_plus, C_prime, s_prime }
     */
    const buildTree = (q, p, u, v, j) => {
      const q0 = q.copy();
      q = q.copy();
      p = p.copy();

      if (j === 0) {
        // Base case - take one leapfrog step in the direction v
        p.increment(self.gradLogDensity(q).scale((v * self.dt) / 2));
        q.increment(p.scale(v * self.dt));
        p.increment(self.gradLogDensity(q).scale((v * self.dt) / 2));

        const C_prime = [];
        if (u < Math.exp(self.logDensity(q) - p.norm2() / 2)) {
          C_prime.push([q.copy(), p.copy()]);
          trajectory.push({ type: "accept", from: q0.copy(), to: q.copy() });
        } else {
          trajectory.push({ type: "reject", from: q0.copy(), to: q.copy() });
        }

        const s_prime = u < Math.exp(self.Delta_max + self.logDensity(q) - p.norm2() / 2) ? 1 : 0;

        return {
          q_plus: q,
          p_plus: p,
          q_minus: q,
          p_minus: p,
          C_prime: C_prime,
          s_prime: s_prime,
        };
      } else {
        // Recursion - build the left and right subtrees
        let result = buildTree(q, p, u, v, j - 1);
        let { q_minus, p_minus, q_plus, p_plus, C_prime, s_prime } = result;
        let C_pprime, s_pprime;

        if (v === -1) {
          result = buildTree(q_minus, p_minus, u, v, j - 1);
          q_minus = result.q_minus;
          p_minus = result.p_minus;
          C_pprime = result.C_prime;
          s_pprime = result.s_prime;
        } else {
          result = buildTree(q_plus, p_plus, u, v, j - 1);
          q_plus = result.q_plus;
          p_plus = result.p_plus;
          C_pprime = result.C_prime;
          s_pprime = result.s_prime;
        }

        const I1 = q_plus.subtract(q_minus).dot(p_minus) >= 0 ? 1 : 0;
        const I2 = q_plus.subtract(q_minus).dot(p_plus) >= 0 ? 1 : 0;
        s_prime = s_prime * s_pprime * I1 * I2;

        // C' = C' ∪ C''
        for (let i = 0; i < C_pprime.length; ++i) {
          C_prime.push(C_pprime[i]);
        }

        return {
          q_plus: q_plus,
          p_plus: p_plus,
          q_minus: q_minus,
          p_minus: p_minus,
          C_prime: C_prime,
          s_prime: s_prime,
        };
      }
    };

    const p0 = MultivariateNormal.getSample(self.dim);
    const u = Math.random() * Math.exp(self.logDensity(self.chain.last()) - p0.norm2() / 2);

    let q_minus = self.chain.last().copy();
    let q_plus = self.chain.last().copy();
    let p_minus = p0.copy();
    let p_plus = p0.copy();
    let j = 0;
    const C = [[self.chain.last().copy(), p0.copy(), 0]];
    let s = 1;

    while (s === 1) {
      const v = Math.sign(Math.random() - 0.5);
      let C_prime, s_prime;

      if (v === -1) {
        trajectory.push({ type: "left" });
        const result = buildTree(q_minus, p_minus, u, v, j);
        q_minus = result.q_minus;
        p_minus = result.p_minus;
        C_prime = result.C_prime;
        s_prime = result.s_prime;
      } else {
        trajectory.push({ type: "right" });
        const result = buildTree(q_plus, p_plus, u, v, j);
        q_plus = result.q_plus;
        p_plus = result.p_plus;
        C_prime = result.C_prime;
        s_prime = result.s_prime;
      }

      // If s' == 1, then C = C ∪ C'
      if (s_prime === 1) {
        for (let i = 0; i < C_prime.length; ++i) {
          C.push([C_prime[i][0], C_prime[i][1], j]);
        }
      }

      const I1 = q_plus.subtract(q_minus).dot(p_minus) >= 0 ? 1 : 0;
      const I2 = q_plus.subtract(q_minus).dot(p_plus) >= 0 ? 1 : 0;
      s = s_prime * I1 * I2;
      j = j + 1;
    }

    // Sample (q, p) uniformly at random from C
    const index = Math.floor(Math.random() * C.length);
    const q = C[index][0];

    self.chain.push(q.copy());

    visualizer.queue.push({
      type: "proposal",
      proposal: q,
      nuts_trajectory: trajectory,
      initialMomentum: p0,
    });
    visualizer.queue.push({ type: "accept", proposal: q });
  },
});
