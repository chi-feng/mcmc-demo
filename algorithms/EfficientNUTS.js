"use strict";

/**
 * Efficient No-U-Turn Sampler (NUTS)
 *
 * An optimized version of NUTS that avoids storing all candidate points.
 * Uses slice sampling within the tree to select the next state.
 * This is Algorithm 3 from the NUTS paper.
 *
 * @see http://arxiv.org/abs/1111.4246
 */
MCMC.registerAlgorithm("EfficientNUTS", {
  description: "Efficient No-U-Turn Sampler",

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

    // BuildTree from Algorithm 3: Efficient No-U-Turn Sampler
    const buildTree = (q, p, u, v, j) => {
      q = q.copy();
      // Copy p too: the leapfrog below mutates it, and without a copy the second
      // recursive call overwrites the momentum kept as the first subtree's endpoint
      p = p.copy();
      const q0 = q.copy();

      if (j === 0) {
        // Base case - take one leapfrog step in the direction v
        p.increment(self.gradLogDensity(q).scale((v * self.dt) / 2));
        q.increment(p.scale(v * self.dt));
        p.increment(self.gradLogDensity(q).scale((v * self.dt) / 2));

        const n_ = u < Math.exp(self.logDensity(q) - p.norm2() / 2) ? 1 : 0;
        const s_ = u < Math.exp(self.Delta_max + self.logDensity(q) - p.norm2() / 2) ? 1 : 0;

        trajectory.push({
          type: n_ === 1 ? "accept" : "reject",
          from: q0.copy(),
          to: q.copy(),
        });

        return { q_p: q, p_p: p, q_m: q, p_m: p, q_: q, n_: n_, s_: s_ };
      } else {
        // Recursion - build the left and right subtrees
        let result = buildTree(q, p, u, v, j - 1);
        let { q_m, p_m, q_p, p_p, q_, n_, s_ } = result;

        if (s_ === 1) {
          let n__, s__, q__;

          if (v === -1) {
            result = buildTree(q_m, p_m, u, v, j - 1);
            q_m = result.q_m;
            p_m = result.p_m;
            q__ = result.q_;
            n__ = result.n_;
            s__ = result.s_;
          } else {
            result = buildTree(q_p, p_p, u, v, j - 1);
            q_p = result.q_p;
            p_p = result.p_p;
            q__ = result.q_;
            n__ = result.n_;
            s__ = result.s_;
          }

          if (Math.random() < n__ / (n_ + n__)) {
            q_ = q__;
          }

          s_ = s_ * s__ * (q_p.subtract(q_m).dot(p_m) >= 0 ? 1 : 0) * (q_p.subtract(q_m).dot(p_p) >= 0 ? 1 : 0);
          n_ = n_ + n__;
        }

        return {
          q_p: q_p,
          p_p: p_p,
          q_m: q_m,
          p_m: p_m,
          q_: q_,
          n_: n_,
          s_: s_,
        };
      }
    };

    const p0 = MultivariateNormal.getSample(self.dim);
    const u = Math.random() * Math.exp(self.logDensity(self.chain.last()) - p0.norm2() / 2);

    let q = self.chain.last().copy();
    let q_m = self.chain.last().copy();
    let q_p = self.chain.last().copy();
    let p_m = p0.copy();
    let p_p = p0.copy();
    let j = 0;
    let n = 1;
    let s = 1;

    while (s === 1) {
      const v = Math.sign(Math.random() - 0.5);
      let q_, n_, s_;

      if (v === -1) {
        const result = buildTree(q_m, p_m, u, v, j);
        q_m = result.q_m;
        p_m = result.p_m;
        q_ = result.q_;
        n_ = result.n_;
        s_ = result.s_;
      } else {
        const result = buildTree(q_p, p_p, u, v, j);
        q_p = result.q_p;
        p_p = result.p_p;
        q_ = result.q_;
        n_ = result.n_;
        s_ = result.s_;
      }

      if (s_ === 1 && Math.random() < n_ / n) {
        q = q_.copy();
      }

      s = s_ * (q_p.subtract(q_m).dot(p_m) >= 0 ? 1 : 0) * (q_p.subtract(q_m).dot(p_p) >= 0 ? 1 : 0);
      n = n + n_;
      j = j + 1;
    }

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
