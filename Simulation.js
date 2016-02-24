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
  this.mcmc.init = MCMC.algorithms[algorithmName].init;
  this.mcmc.reset = MCMC.algorithms[algorithmName].reset;
  this.mcmc.step = MCMC.algorithms[algorithmName].step;
  this.mcmc.attachUI = MCMC.algorithms[algorithmName].attachUI;
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
  this.visualizer.dequeue();
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

window.onload = function() {
  var viz = new Visualizer(document.getElementById('visualizer'));
  var sim = new Simulation();
  sim.visualizer = viz;
  viz.simulation = sim;
  sim.setAlgorithm(MCMC.algorithmNames[0]);
  sim.setTarget(MCMC.targetNames[0]);
  sim.mcmc.init(sim.mcmc);

  var gui = new dat.GUI({width: 300});

  var f1 = gui.addFolder('Simulation options');
  f1.add(sim, 'algorithm', MCMC.algorithmNames).name('Algorithm').onChange(function(value) {
    sim.setAlgorithm(value);
    gui.removeFolder('MCMC Options');
    var f = gui.addFolder('MCMC Options');
    sim.mcmc.attachUI(sim.mcmc, f);
    f.open();
  });
  f1.add(sim, 'target', MCMC.targetNames).name('Target distribution').onChange(function(value) {
    sim.setTarget(value);
  });
  f1.add(sim, 'autoplay').name('Autoplay');
  f1.add(sim, 'delay', 16, 1000).name('Autoplay delay (msec)');
  f1.add(sim, 'step').name('Step forward');
  f1.add(sim, 'reset').name('Reset chain');
  f1.open();

  var f2 = gui.addFolder('Visualization Options');
  f2.add(viz, 'animateProposal').name('Animate proposal');
  f2.add(viz, 'showTargetDensity').name('Show target density');
  f2.open();

  gui.removeFolder('MCMC Options');
  var f3 = gui.addFolder('MCMC Options');
  sim.mcmc.attachUI(sim.mcmc, f3);
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