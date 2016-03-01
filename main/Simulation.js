'use strict';

function Simulation() {
  this.mcmc = {initialized: false, hasAlgorithm: false, hasTarget: false, dim: 2};
  this.delay = 250;
  this.autoplay = true;
};

Simulation.prototype.setAlgorithm = function(algorithmName) {
  console.log('Setting algorithm to ' + algorithmName);
  this.hasAlgorithm = true;
  this.algorithm = algorithmName;
  this.mcmc.initialized = false;
  this.mcmc.description = MCMC.algorithms[algorithmName].description;
  this.mcmc.init = MCMC.algorithms[algorithmName].init;
  this.mcmc.reset = MCMC.algorithms[algorithmName].reset;
  this.mcmc.step = MCMC.algorithms[algorithmName].step;
  this.mcmc.attachUI = MCMC.algorithms[algorithmName].attachUI;
  this.mcmc.about = MCMC.algorithms[algorithmName].about;
  if (this.hasAlgorithm && this.hasTarget) {
    this.visualizer.resize();
    if (this.mcmc.initialized == false)
      this.mcmc.init(this.mcmc);
    this.mcmc.reset(this.mcmc);
    this.mcmc.initialized = true;
  }
};

Simulation.prototype.setTarget = function(targetName) {
  console.log('Setting target to ' + targetName);
  this.hasTarget = true;
  this.target = targetName;
  this.mcmc.logDensity = MCMC.targets[targetName].logDensity;
  this.mcmc.gradLogDensity = MCMC.targets[targetName].gradLogDensity;

  // TODO: actually derive Hessians
  // in the meantime, use finite difference :sadface:
  var grad = this.mcmc.gradLogDensity, N = this.mcmc.dim;
  var h = 1e-8;
  this.mcmc.hessLogDensity = function(x) {
    var hess = zeros(N, N);
    var Delta = eye(N, N).scale(h);
    for (var i = 0; i < N; ++i) {
      for (var j = 0; j < N; ++j) {
        hess[i * N + j]  = (grad(x.add(Delta.col(j)))[i] - grad(x)[i]) / (2 * h)
                         + (grad(x.add(Delta.col(i)))[j] - grad(x)[j]) / (2 * h);
      }
    }
    return hess;
  };

  if (this.mcmc.initialized)
    this.mcmc.reset(this.mcmc);
  if (this.hasAlgorithm && this.hasTarget) {
    this.visualizer.resize();
    if (this.mcmc.initialized == false)
      this.mcmc.init(this.mcmc);
    this.mcmc.reset(this.mcmc);
    this.mcmc.initialized = true;
  }
};

Simulation.prototype.reset = function() {
  this.mcmc.reset(this.mcmc);
  this.visualizer.resize();
};

Simulation.prototype.step = function() {
  if (this.visualizer.queue.length == 0)
    this.mcmc.step(this.mcmc, this.visualizer);
  if (this.visualizer.animateProposal == false) {
    while (this.visualizer.queue.length > 0)
      this.visualizer.dequeue();
  } else {
    this.visualizer.dequeue();
  }
  this.visualizer.render();
};

Simulation.prototype.animate = function() {
  var self = this;
  if (this.autoplay || this.visualizer.tweening)
    this.step();
  if (this.visualizer.tweening) {
    requestAnimationFrame(function() { self.animate(); });
  } else {
    setTimeout(function() { requestAnimationFrame(function() { self.animate(); }); }, self.delay);
  }
};

var viz, sim, gui;

window.onload = function() {
  viz = new Visualizer(document.getElementById('visualizer'));
  sim = new Simulation();
  sim.visualizer = viz;
  viz.simulation = sim;

  var algorithm = MCMC.algorithmNames[0];
  var target = MCMC.targetNames[0];

  if (window.location.hash != '') {
    var hash = window.location.hash.substring(1);
    var tokens = hash.split(',');
    if (MCMC.algorithmNames.indexOf(tokens[0]) > -1) {
      algorithm = tokens[0];
    }
    if (tokens.length > 1 && MCMC.targetNames.indexOf(tokens[1]) > -1) {
      target = tokens[1];
    }
  }

  function updateHash(sim) {
    window.location.hash = '#' + sim.algorithm + ',' + sim.target;
  }

  sim.setAlgorithm(algorithm);
  sim.setTarget(target);

  updateHash(sim);

  sim.mcmc.init(sim.mcmc);
  window.onresize = function() { viz.resize(); };

  gui = new dat.GUI({width: 300});

  var f1 = gui.addFolder('Simulation options');
  f1.add(sim, 'algorithm', MCMC.algorithmNames).name('Algorithm').onChange(function(value) {
    sim.setAlgorithm(value);
    updateHash(sim);
    gui.removeFolder('Algorithm Options');
    var f = gui.addFolder('Algorithm Options');
    sim.mcmc.attachUI(sim.mcmc, f);
    f.add(sim.mcmc, 'about').name('About this algorithm...');
    f.open();
  });
  f1.add(sim, 'target', MCMC.targetNames).name('Target distribution').onChange(function(value) {
    sim.setTarget(value);
    updateHash(sim);
  });
  f1.add(sim, 'autoplay').name('Autoplay');
  f1.add(sim, 'delay', 0, 1000).name('Autoplay delay');
  f1.add(sim, 'step').name('Step');
  f1.add(sim, 'reset').name('Reset');
  f1.open();

  var f2 = gui.addFolder('Visualization Options');
  f2.add(viz, 'animateProposal').name('Animate proposal');
  f2.add(viz, 'showTargetDensity').name('Show target');
  f2.add(viz, 'showSamples').name('Show samples');
  f2.open();

  gui.removeFolder('Algorithm Options');
  var f3 = gui.addFolder('Algorithm Options');
  sim.mcmc.attachUI(sim.mcmc, f3);
  f3.add(sim.mcmc, 'about').name('About...');
  f3.open();

  sim.animate();
};


dat.GUI.prototype.removeFolder = function(name) {
  var folder = this.__folders[name];
  if (!folder) {
    return;
  }
  folder.close();
  this.__ul.removeChild(folder.domElement.parentNode);
  delete this.__folders[name];
  this.onResize();
};