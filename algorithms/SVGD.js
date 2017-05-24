'use strict';

MCMC.registerAlgorithm('SVGD', {

  description: 'Stein Variational Gradient Descent',

  about: function() {
    window.open('http://www.cs.dartmouth.edu/~dartml/project.html?p=vgd');
  },

  init: function(self) {
    self.chain = [ ];
    self.n = 200; // number of particlese
    self.epsilon = 0.01; // step size
    self.h = 0.15; // bandwidth
    self.use_median = true;
    self.use_adagrad = true;
    self.alpha = 0.9;
    self.fudge_factor = 1e-2;
    self.iter = 0;
    self.reset(self);
  },

  reset: function(self) {
    // initialize chain with samples from standard normal
    self.chain = [ ];
    self.gradx = [];
    self.historical_grad = [];
    self.gradLogDensities = [];
    self.iter = 0;
    for (var i = 0; i < self.n; i++) {
      self.chain.push(MultivariateNormal.getSample(self.dim));
      self.gradx.push(Float64Array.zeros(self.dim,1));
      self.historical_grad.push(Float64Array.zeros(self.dim,1));
      self.gradLogDensities.push(0);
    }
  },

  attachUI: function(self, folder) {
    folder.add(self, 'use_median').name('Median heuristic').listen();
    folder.add(self, 'h', 0.05, 2).step(0.05).name('bandwidth').listen().onChange(function(value) {
      self.use_median = false;
    });
    folder.add(self, 'use_adagrad').name('Adagrad');
    folder.add(self, 'epsilon', 0.001, 0.1).step(0.001).name('stepsize');
    // folder.add(self, 'alpha', 0.01, 1.0).step(0.01).name('alpha');
    // folder.add(self, 'fudge_factor', 0.0001, 0.05).step(0.0001).name('fudge_factor');
    folder.add(self, 'n', 10, 400).step(1).name('numParticles');
    folder.open();
  },

  step: function(self, visualizer) {

    // resize samples appropriately
    if (self.n > self.chain.length) {
      for (var i = 0; i < self.n - self.chain.length; i++) {
        self.chain.push(MultivariateNormal.getSample(self.dim));
        self.gradx.push(Float64Array.zeros(self.dim,1));
        self.historical_grad.push(Float64Array.zeros(self.dim,1));
        self.gradLogDensities.push(0);
      }
    } else if (self.n < self.chain.length) {
      self.chain = self.chain.slice(0, self.n);
      self.gradx = self.gradx.slice(0, self.n);
      self.historical_grad = self.historical_grad.slice(0, self.n);
      self.gradLogDensities = self.gradLogDensities.slice(0, self.n);
    }

    var n = self.chain.length;

    // precompute log densities
    for (var i = 0; i < n; i++) {
      self.gradLogDensities[i] = self.gradLogDensity(self.chain[i]);
      for (var k = 0; k < self.dim; k++) { self.gradx[i][k] = 0; }
    }

    // pairwise distances trick
    var dist2 = new Float64Array(n * n);
    for (var i = 0; i < n; i++) {
      for (var j = 0; j < i; j++) {
        var delta = 0;
        for (var k = 0; k < self.dim; k++)
          delta += Math.pow(self.chain[i][k] - self.chain[j][k], 2);
        dist2[i * n + j] = delta;
        dist2[j * n + i] = delta;
      }
    }

    if (self.use_median) {
      var dist2copy = new Float64Array(dist2);
      dist2copy.sort();
      var median = dist2copy[Math.floor(dist2copy.length/2)];
      self.h = median / Math.log(n);
    }

    // compute gradient
    for (var i = 0; i < n; i++) {
      for (var j = 0; j < n; j++) {
        var rbf = Math.exp(-dist2[i*n+j]  / (self.h));
        for (var k = 0; k < self.dim; k++) {
          var grad_rbf = (self.chain[i][k] - self.chain[j][k]) * 2 * rbf / (self.h);
          self.gradx[i][k] += self.gradLogDensities[j][k] * rbf + grad_rbf;
        }
      }
      for (var k = 0; k < self.dim; k++) { self.gradx[i][k] /= n; }
    }

    // adagrad
    if (self.use_adagrad) {
      for (var i = 0; i < n; i++)
        for (var k = 0; k < self.dim; k++)
          self.historical_grad[i][k] = self.alpha * self.historical_grad[i][k] + (1 - self.alpha) * Math.pow(self.gradx[i][k], 2);
      for (var i = 0; i < n; i++)
        for (var k = 0; k < self.dim; k++)
          self.gradx[i][k] /= (self.fudge_factor + Math.sqrt(self.historical_grad[i][k]));
    }

    for (var i = 0; i < n; i++) {
      for (var k = 0; k < self.dim; k++) {
        self.gradx[i][k] *= self.epsilon;
      }
    }

    visualizer.queue.push({ type: 'svgd-step', x: self.chain, gradx: self.gradx, h: self.h});

    // update particles
    for (var i = 0; i < n; i++) {
      self.chain[i].increment(self.gradx[i]);
    }

    self.iter++;

  }

});
