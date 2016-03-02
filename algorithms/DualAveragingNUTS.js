'use strict';

MCMC.registerAlgorithm('DualAveragingNUTS', {

  description: 'No-U-Turn Sampler with Dual Averaging',

  about: function() {
    window.open('http://arxiv.org/abs/1111.4246');
  },

  init: function(self) {
    self.Delta_max = 1000;

    self.delta = 0.65;
    self.M_adapt = 200;

    self.dt = 0;

    viz.animateProposal = false;

    self.joint = function(q, p) {
      return Math.exp(self.logDensity(q) - p.norm2() / 2);
    };

    self.leapFrog = function(theta, r, epsilon) {
      var r_ = r.add(self.gradLogDensity(theta).scale(epsilon / 2));
      var theta_ = theta.add(r_.scale(epsilon));
      r_.increment(self.gradLogDensity(theta_).scale(epsilon / 2));
      return {theta: theta_, r: r_};
    };

    self.findReasonableEpsilon = function(theta) {
      var epsilon = 0.1;
      var r = MultivariateNormal.getSample(self.dim);
      var result = self.leapFrog(theta, r, epsilon);
      var a = 2 * (self.joint(result.theta, result.r) / self.joint(theta, r) > 0.5 ? 1 : 0) - 1;
      while (Math.pow(self.joint(result.theta, result.r) / self.joint(theta, r), a) > Math.pow(2.0, -a)) {
        epsilon = Math.pow(2, a) * epsilon;
        result = self.leapFrog(result.theta, result.r, epsilon);
      }
      return Math.max(0.1, epsilon);
    };
  },

  reset: function(self) {
    self.chain = [ MultivariateNormal.getSample(self.dim) ];
    self.epsilon = [ self.findReasonableEpsilon(self.chain.last()) ];
    self.mu = Math.log(10 * self.epsilon[0]);
    self.epsilon_bar = [ 1.0 ];
    self.H_bar = [ 1.0 ];

    self.gamma = 0.2;
    self.t0 = 10;
    self.kappa = 0.75;

  },

  attachUI: function(self, folder) {
    folder.add(self, 'delta', 0.2, 1).step(0.05).name('&delta;').onChange(function(value) {
      sim.reset();
    });
    folder.open();
  },

  // Notation adopted from http://arxiv.org/pdf/1111.4246v1.pdf
  step: function(self, visualizer) {

    var trajectory = [ ];

    // BuildTree from Algorithm 3: Efficient No-U-Turn Sampler
    function buildTree(q, p, u, v, j, dt, q0, p0) {
      var q  = q.copy(), q0 = q.copy();
      if (j == 0) {
        // base case - take one leapfrog step in the direction v
        p.increment(self.gradLogDensity(q).scale(v * dt / 2));
        q.increment(p.scale(v * dt));
        p.increment(self.gradLogDensity(q).scale(v * dt / 2));
        var n_ = (u < Math.exp(self.logDensity(q) - p.norm2() / 2)) ? 1 : 0;
        var s_ = (u < Math.exp(self.Delta_max + self.logDensity(q) - p.norm2() / 2)) ? 1 : 0;
        trajectory.push({type: n_ == 1 ? 'accept' : 'reject', from: q0.copy(), to: q.copy()});
        var alpha = Math.min(1, Math.exp(self.logDensity(q) - p.norm2() / 2 - self.logDensity(q0) + p0.norm2() / 2));
        var n_alpha = 1;
        return {q_p: q, p_p: p, q_m: q, p_m: p, q_: q, n_: n_, s_: s_, alpha_: alpha, n_alpha_: n_alpha};
      } else {
        // recursion - build the left and right subtrees
        var result = buildTree(q, p, u, v, j - 1, dt, q0, p0);
        var q_m = result.q_m, p_m = result.p_m, q_p = result.q_p, p_p = result.p_p, q_ = result.q_, n_ = result.n_, s_ = result.s_, alpha_ = result.alpha_, n_alpha_ = result.n_alpha_ ;
        if (s_ == 1) {
          var n__, s__, q__, alpha__, n_alpha__;
          if (v == -1) {
            var result = buildTree(q_m, p_m, u, v, j - 1, dt, q0, p0);
            q_m = result.q_m; p_m = result.p_m; q__ = result.q_; n__ = result.n_; s__ = result.s_; alpha__ = result.alpha_; n_alpha__ = result.n_alpha_;
          } else {
            var result = buildTree(q_p, p_p, u, v, j - 1, dt, q0, p0);
            q_p = result.q_p; p_p = result.p_p; q__ = result.q_; n__ = result.n_; s__ = result.s_; alpha__ = result.alpha_; n_alpha__ = result.n_alpha_;
          }
          if (Math.random() < n__ / (n_ + n__))
            q_ = q__;
          alpha_ = alpha_ + alpha__;
          n_alpha_ = n_alpha_ + n_alpha__;
          s_ = s_ * s__ * (q_p.subtract(q_m).dot(p_m) >= 0 ? 1 : 0) * (q_p.subtract(q_m).dot(p_p) >= 0 ? 1 : 0);
          n_ = n_ + n__;
        }
        return {q_p: q_p, p_p: p_p, q_m: q_m, p_m: p_m, q_: q_, n_: n_, s_: s_, alpha_: alpha_, n_alpha_: n_alpha_};
      }
    }

    var p0 = MultivariateNormal.getSample(self.dim);
    var u  = Math.random() * Math.exp(self.logDensity(self.chain.last()) - p0.norm2() / 2);

    var q   = self.chain.last().copy(),
        q_m = self.chain.last().copy(),
        q_p = self.chain.last().copy(),
        p_m = p0.copy(), p_p = p0.copy(),
        j = 0, n = 1, s = 1, alpha, n_alpha;

    while (s == 1) {
      var v = Math.sign(Math.random() - 0.5);
      var q_, n_, s_;
      if (v == -1) {
        var result = buildTree(q_m, p_m, u, v, j, self.epsilon.last(), self.chain.last(), p0);
        q_m = result.q_m; p_m = result.p_m; q_ = result.q_; n_ = result.n_; s_ = result.s_; alpha = result.alpha_; n_alpha = result.n_alpha_;
      } else {
        var result = buildTree(q_p, p_p, u, v, j, self.epsilon.last(), self.chain.last(), p0);
        q_p = result.q_p; p_p = result.p_p; q_ = result.q_; n_ = result.n_; s_ = result.s_; alpha = result.alpha_; n_alpha = result.n_alpha_;
      }
      if (s_ == 1 && Math.random() < n_ / n)
        q = q_.copy();
      s = s_ * (q_p.subtract(q_m).dot(p_m) >= 0 ? 1 : 0) * (q_p.subtract(q_m).dot(p_p) >= 0 ? 1 : 0);
      n = n + n_;
      j = j + 1;
    }
    var epsilon = ((self.epsilon.last() * 1000) | 0) / 1000;
    visualizer.queue.push({type: 'proposal', proposal: q, nuts_trajectory: trajectory, initialMomentum: p0, epsilon: epsilon, alpha:self.delta - self.H_bar.last() });
    self.chain.push(q.copy());

    var m = self.chain.length;
    if (m <= self.M_adapt) {
      self.H_bar.push((1 - 1 / (m + self.t0)) * self.H_bar.last() + (1 / (m + self.t0)) * (self.delta - alpha / n_alpha));
      var log_epsilon = self.mu - Math.sqrt(m) / self.gamma * self.H_bar.last();
      log_epsilon = Math.max(log_epsilon, -4.5);
      self.epsilon.push(Math.exp(log_epsilon));
      self.epsilon_bar.push(Math.exp(Math.pow(m, -self.kappa) * log_epsilon + (1 - Math.pow(m, -self.kappa)) * Math.log(self.epsilon_bar.last())));
    } else {
      self.epsilon.push(self.epsilon_bar.last());
    }

    visualizer.queue.push({type: 'accept', proposal: q});
  }

});