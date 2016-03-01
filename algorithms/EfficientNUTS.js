'use strict';

MCMC.registerAlgorithm('EfficientNUTS', {

  description: 'Efficient No-U-Turn Sampler',

  about: function() {
    window.open('http://arxiv.org/abs/1111.4246');
  },

  init: function(self) {
    self.dt = 0.1;
    self.Delta_max = 1000;
  },

  reset: function(self) {
    self.chain = [ MultivariateNormal.getSample(self.dim) ];
  },

  attachUI: function(self, folder) {
    folder.add(self, 'dt', 0.025, 0.6).step(0.025).name('Leapfrog &Delta;t');
    folder.open();
  },

  // Notation adopted from http://arxiv.org/pdf/1111.4246v1.pdf
  step: function(self, visualizer) {

    var trajectory = [ ];

    // BuildTree from Algorithm 3: Efficient No-U-Turn Sampler
    function buildTree(q, p, u, v, j) {
      var q  = q.copy(), q0 = q.copy();
      if (j == 0) {
        // base case - take one leapfrog step in the direction v
        p.increment(self.gradLogDensity(q).scale(v * self.dt / 2));
        q.increment(p.scale(v * self.dt));
        p.increment(self.gradLogDensity(q).scale(v * self.dt / 2));
        var n_ = (u < Math.exp(self.logDensity(q) - p.norm2() / 2)) ? 1 : 0;
        var s_ = (u < Math.exp(self.Delta_max + self.logDensity(q) - p.norm2() / 2)) ? 1 : 0;
        trajectory.push({type: n_ == 1 ? 'accept' : 'reject', from: q0.copy(), to: q.copy()});
        return {q_p: q, p_p: p, q_m: q, p_m: p, q_: q, n_: n_, s_: s_};
      } else {
        // recursion - build the left and right subtrees
        var result = buildTree(q, p, u, v, j - 1);
        var q_m = result.q_m, p_m = result.p_m, q_p = result.q_p, p_p = result.p_p, q_ = result.q_, n_ = result.n_, s_ = result.s_;
        if (s_ == 1) {
          var n__, s__, q__;
          if (v == -1) {
            var result = buildTree(q_m, p_m, u, v, j - 1);
            q_m = result.q_m; p_m = result.p_m; q__ = result.q_; n__ = result.n_; s__ = result.s_;
          } else {
            var result = buildTree(q_p, p_p, u, v, j - 1);
            q_p = result.q_p; p_p = result.p_p; q__ = result.q_; n__ = result.n_; s__ = result.s_;
          }
          if (Math.random() < n__ / (n_ + n__))
            q_ = q__;
          s_ = s_ * s__ * (q_p.subtract(q_m).dot(p_m) >= 0 ? 1 : 0) * (q_p.subtract(q_m).dot(p_p) >= 0 ? 1 : 0);
          n_ = n_ + n__;
        }
        return {q_p: q_p, p_p: p_p, q_m: q_m, p_m: p_m, q_: q_, n_: n_, s_: s_};
      }
    }

    var p0 = MultivariateNormal.getSample(self.dim);
    var u  = Math.random() * Math.exp(self.logDensity(self.chain.last()) - p0.norm2() / 2);

    var q   = self.chain.last().copy(),
        q_m = self.chain.last().copy(),
        q_p = self.chain.last().copy(),
        p_m = p0.copy(), p_p = p0.copy(),
        j = 0, n = 1, s = 1;

    while (s == 1) {
      var v = Math.sign(Math.random() - 0.5);
      var q_, n_, s_;
      if (v == -1) {
        var result = buildTree(q_m, p_m, u, v, j);
        q_m = result.q_m; p_m = result.p_m; q_ = result.q_; n_ = result.n_; s_ = result.s_;
      } else {
        var result = buildTree(q_p, p_p, u, v, j);
        q_p = result.q_p; p_p = result.p_p; q_ = result.q_; n_ = result.n_; s_ = result.s_;
      }
      if (s_ == 1 && Math.random() < n_ / n)
        q = q_.copy();
      s = s_ * (q_p.subtract(q_m).dot(p_m) >= 0 ? 1 : 0) * (q_p.subtract(q_m).dot(p_p) >= 0 ? 1 : 0);
      n = n + n_;
      j = j + 1;
    }

    self.chain.push(q.copy());

    visualizer.queue.push({type: 'proposal', proposal: q, nuts_trajectory: trajectory, initialMomentum: p0});
    visualizer.queue.push({type: 'accept', proposal: q});
  }

});