'use strict';

var MCMC = {
  algorithmNames:[],
  algorithms:{},
  targetNames:[],
  targets: {},
  registerAlgorithm: function(name, methods) {
    MCMC.algorithmNames.push(name);
    MCMC.algorithms[name] = methods;
  }
};

// Standard bimodal distribution
MCMC.targetNames.push('standard');
var dist = new MultivariateNormal(Float64Array.zeros(2,1), Float64Array.eye(2));
MCMC.targets['standard'] = {
  logDensity: function(x) {
    return dist.logDensity(x);
  },
  gradLogDensity: function(x) {
    return dist.gradLogDensity(x);
  }
};


// Mixture distribution with two components
var mixtureComponents = [
  new MultivariateNormal(Float64Array.matrix([[-1.5],[-1.5]]), Float64Array.eye(2).scale(0.8)),
  new MultivariateNormal(Float64Array.matrix([[1.5],[1.5]]), Float64Array.eye(2).scale(0.8))
];
MCMC.targetNames.push('bimodal');
MCMC.targets['bimodal'] = {
  logDensity: function(x) {
    return Math.log(Math.exp(mixtureComponents[0].logDensity(x)) + Math.exp(mixtureComponents[1].logDensity(x)));
  },
  gradLogDensity: function(x) {
    var p1 = Math.exp(mixtureComponents[0].logDensity(x));
    var p2 = Math.exp(mixtureComponents[1].logDensity(x));
    return (mixtureComponents[0].gradLogDensity(x).scale(p1).add(mixtureComponents[1].gradLogDensity(x).scale(p2))).scale(1 / (p1 + p2));
  }
};


// Banana distribution
var bananaDist = new MultivariateNormal(Float64Array.matrix([[0],[4]]), Float64Array.matrix([[1,0.5],[0.5,1]]));
MCMC.targetNames.push('banana');
MCMC.targets['banana'] = {
  logDensity: function(x) {
    var a = 2, b = 0.2;
    var y = Float64Array.zeros(2,1);
    y[0] = x[0] / a;
    y[1] = x[1] * a + a * b * (x[0] * x[0] + a * a);
    return bananaDist.logDensity(y);
  },
  gradLogDensity: function(x) {
    var a = 2, b = 0.2;
    var y = Float64Array.zeros(2,1);
    y[0] = x[0] / a;
    y[1] = x[1] * a + a * b * (x[0] * x[0] + a * a);
    var grad = bananaDist.gradLogDensity(y);
    var gradx0 = grad[0] / a + grad[1] * a * b * 2 * x[0];
    var gradx1 = grad[1] * a;
    grad[0] = gradx0;
    grad[1] = gradx1;
    return grad;
  }
};
