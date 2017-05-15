'use strict';

MCMC.registerAlgorithm('SVGD', {

  description: 'Stein Variational Gradient Descent',

  about: function() {
    window.open('http://www.cs.dartmouth.edu/~dartml/project.html?p=vgd');
  },

  init: function(self) {
    self.chain = [ ];
    self.n = 125; // number of particlese
    self.epsilon = 0.25; // step size
    self.h = 0.15; // bandwidth
    self.reset(self);
  },

  reset: function(self) {
    // initialize chain with samples from standard normal
    self.chain = [ ];
    for (var i = 0; i < self.n; i++) {
      self.chain.push(MultivariateNormal.getSample(self.dim));
    }
  },

  attachUI: function(self, folder) {
    folder.add(self, 'h', 0.05, 1).step(0.05).name('bandwidth');
    folder.add(self, 'epsilon', 0.05, 0.5).step(0.05).name('stepsize');
    folder.add(self, 'n', 10, 400).step(1).name('numParticles');
    folder.open();
  },

  step: function(self, visualizer) {

    var gradx = [];

    // resize samples appropriately
    if (self.n > self.chain.length) {
      for (var i = 0; i < self.n - self.chain.length; i++) {
        self.chain.push(MultivariateNormal.getSample(self.dim));
      }
    } else if (self.n < self.chain.length) {
      self.chain = self.chain.slice(0, self.n);
    }
    
    var x = self.chain;
    var h = self.h;

    // precompute log densities
    var gradLogDensities = [ ];
    x.forEach(function(x_i, i) {
      gradLogDensities.push(self.gradLogDensity(x_i));
    });

    // compute gradient of x as empirical average of gradx_i for i = 1...n
    for (var i = 0; i < self.chain.length; i++) {
      var x_i = self.chain[i];
      var gradx_i = Float64Array.zeros(self.dim);
      for (var j = 0; j < self.chain.length; j++) {
        var x_j = self.chain[j];
        var dist2 = 0;
        for (var k = 0; k < self.dim; k++) 
          dist2 += Math.pow(x_i[k] - x_j[k], 2);
        var rbf = Math.exp(-dist2 / (2 * h * h));
        for (var k = 0; k < self.dim; k++) {
          var grad_rbf = (x_i[k] - x_j[k]) * 2 * rbf / (h * h);
          gradx_i[k] += gradLogDensities[j][k] * rbf + grad_rbf;
        }
      }
      for (var k = 0; k < self.dim; k++) 
        gradx_i[k] /= self.chain.length;
      gradx.push(gradx_i);
    }
    
    /*
    x.forEach(function(x_i, i) {
      var gradx_i = Float64Array.zeros(self.dim);
      x.forEach(function(x_j, j) {
        // compute rbf and gradient
        var rbf = Math.exp(-x_i.dist2(x_j) / (2 * h * h));
        var grad_rbf = x_i.subtract(x_j).scale(2 * rbf / (h * h));
        gradx_i.increment(gradLogDensities[j].scale(rbf).add(grad_rbf));
      });
      gradx.push(gradx_i.scale(1 / self.n)); // average over x_j
    });
    */

    visualizer.queue.push({ type: 'svgd-step', x: x, gradx: gradx, h: h});

    // update particles
    for (var i = 0; i < self.chain.length; i++) {
      x[i].increment(gradx[i].scale(self.epsilon));
    }

  }

});
