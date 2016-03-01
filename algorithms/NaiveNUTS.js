'use strict';

MCMC.registerAlgorithm('NaiveNUTS', {

  description: 'Naive No-U-Turn Sampler',

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

    /**
     * BuildTree from Algorithm 2: Naive No-U-Turn Sampler
     * @param  q position
     * @param  p momentum
     * @param  u Uniform([0, exp{L(q0-½p0·p0)}])
     * @param  v direction of integration (±1)
     * @param  j depth of tree/recursion
     * @return { q_minus, p_minus, q_plus, p_plus, C_prime, s_prime }
     */
    function buildTree(q, p, u, v, j) {
      var q0 = q.copy();
      var p0 = p.copy();
      q = q.copy();
      p = p.copy();
      if (j == 0) {
        // base case - take one leapfrog step in the direction v
        p.increment(self.gradLogDensity(q).scale(v * self.dt / 2));
        q.increment(p.scale(v * self.dt));
        p.increment(self.gradLogDensity(q).scale(v * self.dt / 2));
        var C_prime = [ ];
        if (u < Math.exp(self.logDensity(q) - p.norm2() / 2)) {
          C_prime.push([q.copy(), p.copy()]);
          trajectory.push({type: 'accept', from: q0.copy(), to: q.copy()});
        } else {
          trajectory.push({type: 'reject', from: q0.copy(), to: q.copy()});
        }
        var s_prime = (u < Math.exp(self.Delta_max + self.logDensity(q) - p.norm2() / 2)) ? 1 : 0;
        return {q_plus: q, p_plus: p, q_minus: q, p_minus: p, C_prime: C_prime, s_prime: s_prime};
      } else {
        // recursion - build the left and right subtrees
        var result = buildTree(q, p, u, v, j - 1);
        var q_minus = result.q_minus,
            p_minus = result.p_minus,
            C_prime = result.C_prime,
            s_prime = result.s_prime,
            q_plus  = result.q_plus,
            p_plus  = result.p_plus;
        var C_pprime, s_pprime;
        if (v == -1) {
          var result = buildTree(q_minus, p_minus, u, v, j - 1);
          q_minus = result.q_minus;
          p_minus = result.p_minus;
          C_pprime = result.C_prime;
          s_pprime = result.s_prime;
        } else {
          var result = buildTree(q_plus, p_plus, u, v, j - 1);
          q_plus = result.q_plus;
          p_plus = result.p_plus;
          C_pprime = result.C_prime;
          s_pprime = result.s_prime;
        }
        var I1 = q_plus.subtract(q_minus).dot(p_minus) >= 0 ? 1 : 0;
        var I2 = q_plus.subtract(q_minus).dot(p_plus)  >= 0 ? 1 : 0;
        s_prime = s_prime * s_pprime * I1 * I2;
        // C' = C' ∪ C''
        for (var i = 0; i < C_pprime.length; ++i)
          C_prime.push(C_pprime[i]);
        return {q_plus: q_plus, p_plus: p_plus, q_minus: q_minus, p_minus: p_minus, C_prime: C_prime, s_prime: s_prime};
      }
    }

    var p0 = MultivariateNormal.getSample(self.dim);
    var u  = Math.random() * Math.exp(self.logDensity(self.chain.last()) - p0.norm2() / 2);

    var q_minus = self.chain.last().copy(),
        q_plus  = self.chain.last().copy(),
        p_minus = p0.copy(),
        p_plus  = p0.copy(),
        j       = 0,
        C       = [[self.chain.last().copy(), p0.copy(), 0]],
        s       = 1;

    while (s == 1) {
      var v = Math.sign(Math.random() - 0.5);
      var C_prime, s_prime;
      if (v == -1) {
        trajectory.push({type:'left'});
        var result = buildTree(q_minus, p_minus, u, v, j);
        q_minus = result.q_minus;
        p_minus = result.p_minus;
        C_prime = result.C_prime;
        s_prime = result.s_prime;
      } else {
        trajectory.push({type:'right'});
        var result = buildTree(q_plus, p_plus, u, v, j);
        q_plus  = result.q_plus;
        p_plus  = result.p_plus;
        C_prime = result.C_prime;
        s_prime = result.s_prime;
      }
      // if s' == 1, then C = C ∪ C'
      if (s_prime == 1) {
        for (var i = 0; i < C_prime.length; ++i)
          C.push([C_prime[i][0], C_prime[i][1], j]);
      }
      var I1 = q_plus.subtract(q_minus).dot(p_minus) >= 0 ? 1 : 0;
      var I2 = q_plus.subtract(q_minus).dot(p_plus)  >= 0 ? 1 : 0;
      s = s_prime * I1 * I2;
      j = j + 1;
    }

    // sample (q, p) uniformly at random from C
    var index = Math.floor(Math.random() * C.length);
    var q = C[index][0];
    var p = C[index][1];

    self.chain.push(q.copy());

    visualizer.queue.push({type: 'proposal', proposal: q, nuts_trajectory: trajectory, initialMomentum: p0});
    visualizer.queue.push({type: 'accept', proposal: q});
  }

});