"use strict";

MCMC.registerAlgorithm("GibbsSampling", {
  description: "Gibbs Sampling",

  about: function () {
    window.open("https://en.wikipedia.org/wiki/Gibbs_sampling");
  },

  init: function (self) {},

  reset: function (self) {
    self.chain = [MultivariateNormal.getSample(self.dim)];
  },

  attachUI: function (self, folder) {
    folder.open();
  },

  step: function (self, visualizer) {
    function sampleFullConditional(logDensity, point, index) {
      var point = point.copy();
      // add some noise to avoid grid pattern in samples
      var Xs = linspace(-6 - (Math.random() * 12) / 256, 6 + (Math.random() * 12) / 256, 256);
      var densities = [];
      var marginal = 0;
      for (var i = 0; i < 256; i++) {
        point[index] = Xs[i];
        var density = Math.exp(logDensity(point));
        densities.push(density);
        marginal += density;
      }
      var threshold = marginal * Math.random();
      var sum = 0;
      var i = 0;
      while (sum < threshold) {
        sum += densities[i++];
      }
      point[index] = Xs[i - 1];
      return point;
    }

    var last = self.chain.last();
    var trajectory = [last.copy()];
    for (var i = 0; i < 2; i++) {
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
