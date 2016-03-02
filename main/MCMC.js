'use strict';

var MCMC = {
  algorithmNames:[],
  algorithms:{},
  targetNames:[],
  targets: {},
  registerAlgorithm: function(name, methods) {
    MCMC.algorithmNames.push(name);
    MCMC.algorithms[name] = methods;
  },
  computeMean: function(chain) {
    var mean = chain[0].copy();
    for (var i = 1; i < chain.length; ++i)
      mean.increment(chain[i]);
    return mean.scale(1.0 / chain.length);
  },
  computeAutocorrelation: function(chain, lag) {
    var mean = MCMC.computeMean(chain);
    var autocovariance = zeros(lag, 1);
    for (var k = 0; k <= lag; ++k)
      for (var i = k; i < chain.length; ++i)
        autocovariance[k] += chain[i].subtract(mean).dot(chain[i-k].subtract(mean));
    return autocovariance.scale(1.0 / autocovariance[0]);
  }
};

// Banana distribution
var bananaDist = new MultivariateNormal(matrix([[0],[4]]), matrix([[1,0.5],[0.5,1]]));
MCMC.targetNames.push('banana');
MCMC.targets['banana'] = {
  logDensity: function(x) {
    var a = 2, b = 0.2;
    var y = zeros(2,1);
    y[0] = x[0] / a;
    y[1] = x[1] * a + a * b * (x[0] * x[0] + a * a);
    return bananaDist.logDensity(y);
  },
  gradLogDensity: function(x) {
    var a = 2, b = 0.2;
    var y = zeros(2,1);
    y[0] = x[0] / a;
    y[1] = x[1] * a + a * b * (x[0] * x[0] + a * a);
    var grad = bananaDist.gradLogDensity(y);
    var gradx0 = grad[0] / a + grad[1] * a * b * 2 * x[0];
    var gradx1 = grad[1] * a;
    grad[0] = gradx0;
    grad[1] = gradx1;
    return grad;
  },
};

// Donut
MCMC.targetNames.push('donut');
MCMC.targets['donut'] = {
  radius: 2.6,
  sigma2: 0.033,
  logDensity: function(x) {
    var r = x.norm();
    return -Math.pow(r - MCMC.targets.donut.radius, 2) / MCMC.targets.donut.sigma2;
  },
  gradLogDensity: function(x) {
    var r = x.norm();
    if (r == 0) return zeros(2);
    return matrix([[x[0] * (MCMC.targets.donut.radius / r - 1) * 2 / MCMC.targets.donut.sigma2],
                   [x[1] * (MCMC.targets.donut.radius / r - 1) * 2 / MCMC.targets.donut.sigma2]]);
  }
};

// Bivariate normal distribution with no correlation
MCMC.targetNames.push('standard');
var dist = new MultivariateNormal(zeros(2,1), eye(2));
MCMC.targets['standard'] = {
  logDensity: function(x) {
    return dist.logDensity(x);
  },
  gradLogDensity: function(x) {
    return dist.gradLogDensity(x);
  }
};

// Mixture distribution with three components
var mixtureComponents = [
  new MultivariateNormal(matrix([[-1.5],[-1.5]]), eye(2).scale(0.8)),
  new MultivariateNormal(matrix([[1.5],[1.5]]), eye(2).scale(0.8)),
  new MultivariateNormal(matrix([[-2],[2]]), eye(2).scale(0.5))
];
MCMC.targetNames.push('multimodal');
MCMC.targets['multimodal'] = {
  logDensity: function(x) {
    return Math.log(Math.exp(mixtureComponents[0].logDensity(x)) + Math.exp(mixtureComponents[1].logDensity(x)) + Math.exp(mixtureComponents[2].logDensity(x)));
  },
  gradLogDensity: function(x) {
    var p1 = Math.exp(mixtureComponents[0].logDensity(x));
    var p2 = Math.exp(mixtureComponents[1].logDensity(x));
    var p3 = Math.exp(mixtureComponents[2].logDensity(x));
    return (mixtureComponents[0].gradLogDensity(x).scale(p1).add(mixtureComponents[1].gradLogDensity(x).scale(p2)).add(mixtureComponents[2].gradLogDensity(x).scale(p3))).scale(1 / (p1 + p2 + p3));
  }
};
// fillin to get last element of array
if (!Array.prototype.last){
  Array.prototype.last = function(){ return this[this.length - 1]; };
};


// Squiggle distribution
var squiggleDist = new MultivariateNormal(matrix([[0],[0]]), matrix([[2,0.25],[0.25,0.5]]));
MCMC.targetNames.push('squiggle');
MCMC.targets['squiggle'] = {
  logDensity: function(x) {
    var y = zeros(2, 1);
    y[0] = x[0];
    y[1] = x[1] + Math.sin(5 * x[0]);
    return squiggleDist.logDensity(y);
  },
  gradLogDensity: function(x) {
    var y = zeros(2, 1);
    y[0] = x[0];
    y[1] = x[1] + Math.sin(5 * x[0]);
    var grad = squiggleDist.gradLogDensity(y);
    var gradx0 = grad[0] + grad[1] * 5 * Math.cos(5 * x[0]);
    var gradx1 = grad[1];
    grad[0] = gradx0;
    grad[1] = gradx1;
    return grad;
  }
};