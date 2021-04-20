"use strict";

class Simulation {
  constructor() {
    this.mcmc = {
      initialized: false,
      hasAlgorithm: false,
      hasTarget: false,
      dim: 2,
    };
    this.delay = 250;
    this.tweeningDelay = 0;
    this.autoplay = true;
  }
  setAlgorithm(algorithmName) {
    console.log("Setting algorithm to " + algorithmName);
    this.hasAlgorithm = true;
    this.algorithm = algorithmName;
    this.mcmc.initialized = false;
    this.mcmc.description = MCMC.algorithms[algorithmName].description;
    this.mcmc.init = MCMC.algorithms[algorithmName].init;
    this.mcmc.reset = MCMC.algorithms[algorithmName].reset;
    this.mcmc.step = MCMC.algorithms[algorithmName].step;
    this.mcmc.attachUI = MCMC.algorithms[algorithmName].attachUI;
    this.mcmc.about = MCMC.algorithms[algorithmName].about;
    document.getElementById("info").innerHTML = this.mcmc.description;
    if (this.hasAlgorithm && this.hasTarget) {
      if (this.mcmc.initialized == false) this.mcmc.init(this.mcmc);
      this.mcmc.reset(this.mcmc);
      this.mcmc.initialized = true;
      this.visualizer.resize();
    }
  }
  setTarget(targetName) {
    console.log("Setting target to " + targetName, MCMC.targets[targetName]);
    this.hasTarget = true;
    this.target = targetName;
    this.mcmc.logDensity = MCMC.targets[targetName].logDensity;
    this.mcmc.gradLogDensity = MCMC.targets[targetName].gradLogDensity;

    // update visualizer extents
    const options = { ...MCMC.targets[targetName] };
    this.visualizer.xmin = options.xmin;
    this.visualizer.xmax = options.xmax;
    this.visualizer.resize();

    // TODO: actually derive Hessians
    // in the meantime, use finite difference :sadface:
    var grad = this.mcmc.gradLogDensity,
      N = this.mcmc.dim;
    var h = 1e-8;
    this.mcmc.hessLogDensity = function (x) {
      var hess = zeros(N, N);
      var Delta = eye(N, N).scale(h);
      for (let i = 0; i < N; ++i) {
        for (let j = 0; j < N; ++j) {
          hess[i * N + j] =
            (grad(x.add(Delta.col(j)))[i] - grad(x)[i]) / (2 * h) +
            (grad(x.add(Delta.col(i)))[j] - grad(x)[j]) / (2 * h);
        }
      }
      return hess;
    };

    // update contours
    const xmin = this.visualizer.xmin;
    const xmax = this.visualizer.xmax;
    const ymin = this.visualizer.ymin;
    const ymax = this.visualizer.ymax;

    const nx = 480,
      ny = 256,
      nz = 7;
    this.computeContours(this.mcmc.logDensity, xmin, xmax, ymin, ymax, nx, ny, nz);

    if (this.mcmc.initialized) this.mcmc.reset(this.mcmc);
    if (this.hasAlgorithm && this.hasTarget) {
      if (this.mcmc.initialized == false) this.mcmc.init(this.mcmc);
      this.mcmc.reset(this.mcmc);
      this.mcmc.initialized = true;
      this.visualizer.resize();
    }
  }
  computeContours(logDensity, xmin, xmax, ymin, ymax, nx, ny, nz) {
    // get contours
    var x = linspace(xmin, xmax, nx);
    var y = linspace(ymin, ymax, ny);
    var data = [];
    var point = zeros(2, 1);
    var min = 1e10,
      max = 0;
    for (let i = 0; i < nx; ++i) {
      data.push([]);
      point[0] = x[i];
      for (let j = 0; j < ny; ++j) {
        point[1] = y[j];
        var val = Math.exp(logDensity(point));
        data[i].push(val);
        if (val > max) max = val;
        if (val < min) min = val;
      }
    }
    var z = linspace(min + 0.01 * (max - min), max - 0.02 * (max - min), nz);
    var c = new Conrec();
    c.contour(data, 0, nx - 1, 0, ny - 1, x, y, nz, z);
    var contours = c.contourList();
    this.mcmc.contours = [];
    this.mcmc.contourData = data;
    this.mcmc.contourLevels = z;
    for (let i = 0; i < contours.length; ++i) {
      var contour = [];
      for (let j = 0; j < contours[i].length; ++j) contour.push([contours[i][j].x, contours[i][j].y]);
      this.mcmc.contours.push(contour);
    }

    // numerically integrate to get marginal densities
    this.mcmc.xgrid = x;
    this.mcmc.ygrid = y;
    this.mcmc.marginals = [zeros(nx), zeros(ny)];
    for (let i = 0; i < nx; ++i) {
      for (let j = 0; j < ny; ++j) this.mcmc.marginals[0][i] += data[i][j];
      this.mcmc.marginals[0][i];
    }
    this.mcmc.marginals[0] = this.mcmc.marginals[0].scale(1.0 / this.mcmc.marginals[0].maxCoeff());
    for (let j = 0; j < ny; ++j) {
      for (let i = 0; i < nx; ++i) this.mcmc.marginals[1][j] += data[i][j];
      this.mcmc.marginals[1][j];
    }
    this.mcmc.marginals[1] = this.mcmc.marginals[1].scale(1.0 / this.mcmc.marginals[1].maxCoeff());

    var buffer = document.createElement("canvas");
    buffer.width = nx;
    buffer.height = ny;
    var context = buffer.getContext("2d");
    var image = context.createImageData(nx, ny);
    for (let j = 0; j < ny; ++j) {
      for (let i = 0; i < nx; ++i) {
        var base = 4 * ((ny - 1 - j) * nx + i);
        var value = Math.sqrt((data[i][j] - min) / (max - min)) * 255;
        image.data[base] = 102;
        image.data[base + 1] = 153;
        image.data[base + 2] = 187;
        image.data[base + 3] = value | 0;
      }
    }
    context.putImageData(image, 0, 0);
    this.mcmc.densityCanvas = buffer;
  }
  reset() {
    this.mcmc.reset(this.mcmc);
    this.visualizer.resize();
  }
  step() {
    if (this.visualizer.queue.length == 0) this.mcmc.step(this.mcmc, this.visualizer);
    if (this.visualizer.animateProposal == false) {
      while (this.visualizer.queue.length > 0) this.visualizer.dequeue();
    } else {
      this.visualizer.dequeue();
    }
    this.visualizer.render();
  }
  animate() {
    var self = this;
    if (this.autoplay || this.visualizer.tweening) this.step();
    if (this.visualizer.tweening) {
      setTimeout(function () {
        requestAnimationFrame(function () {
          self.animate();
        });
      }, self.tweeningDelay);
    } else {
      setTimeout(function () {
        requestAnimationFrame(function () {
          self.animate();
        });
      }, self.delay);
    }
  }
}
var viz, sim, gui;

function getUrlVars() {
  var vars = [],
    pair;
  var pairs = window.location.search.substr(1).split("&");
  for (let i = 0; i < pairs.length; i++) {
    pair = pairs[i].split("=");
    vars.push(pair[0]);
    vars[pair[0]] = pair[1] && decodeURIComponent(pair[1].replace(/\+/g, " "));
  }
  return vars;
}

window.onload = function () {
  viz = new Visualizer(
    document.getElementById("plotCanvas"),
    document.getElementById("xHistCanvas"),
    document.getElementById("yHistCanvas")
  );
  sim = new Simulation();
  sim.visualizer = viz;
  viz.simulation = sim;

  var algorithm = MCMC.algorithmNames[0];
  var target = MCMC.targetNames[0];
  var seed = Math.seedrandom();

  function parseBool(value) {
    return value == "true";
  }

  if (window.location.search != "") {
    var queryParams = getUrlVars();

    if ("algorithm" in queryParams && MCMC.algorithmNames.indexOf(queryParams["algorithm"]) > -1) {
      algorithm = queryParams["algorithm"];
    }
    if ("target" in queryParams && MCMC.targetNames.indexOf(queryParams["target"]) > -1) {
      target = queryParams["target"];
    }
    if ("seed" in queryParams) {
      // reseed
      seed = Math.seedrandom(queryParams["seed"]);
      console.log("Setting seed to " + seed);
    }
    let config = [
      ["delay", parseInt, sim, "sim"],
      ["tweeningDelay", parseInt, sim, "sim"],
      ["autoplay", parseBool, sim, "sim"],
      ["animateProposal", parseBool, viz, "viz"],
      ["showSamples", parseBool, viz, "viz"],
      ["showHistograms", parseBool, viz, "viz"],
      ["histBins", parseInt, viz, "viz"],
    ];
    for (let i = 0; i < config.length; i++) {
      let param = config[i][0],
        parse = config[i][1],
        obj = config[i][2],
        objName = config[i][3];
      if (param in queryParams) {
        let value = parse(queryParams[param]);
        console.log("Setting " + objName + "." + param + " to " + value);
        obj[param] = value;
      }
    }
  }

  sim.setAlgorithm(algorithm);
  sim.setTarget(target);

  sim.mcmc.init(sim.mcmc);
  window.onresize = function () {
    viz.resize();
  };

  gui = new dat.GUI({ width: 300 });

  var f1 = gui.addFolder("Simulation options");
  f1.add(sim, "algorithm", MCMC.algorithmNames)
    .name("Algorithm")
    .onChange(function (value) {
      sim.setAlgorithm(value);
      gui.removeFolder("Algorithm Options");
      var f = gui.addFolder("Algorithm Options");
      sim.mcmc.attachUI(sim.mcmc, f);
      // f.add(sim.mcmc, 'about').name('About...');
      f.open();
    });
  f1.add(sim, "target", MCMC.targetNames)
    .name("Target distribution")
    .onChange(function (value) {
      sim.setTarget(value);
    });
  f1.add(sim, "autoplay").name("Autoplay");
  f1.add(sim, "delay", 0, 1000)
    .name("Autoplay delay")
    .onChange(function (value) {
      if (value == 0) {
        viz.animateProposal = false;
      } else {
        viz.animateProposal = true;
      }
    });
  f1.add(sim, "tweeningDelay", 0, 200).name("Tweening delay");
  f1.add(sim, "step").name("Step");
  f1.add(sim, "reset").name("Reset");
  f1.open();

  var f2 = gui.addFolder("Visualization Options");
  f2.add(viz, "animateProposal").name("Animate proposal").listen();
  f2.add(viz, "showTargetDensity").name("Show target");
  f2.add(viz, "showSamples").name("Show samples");
  f2.add(viz, "showHistograms").name("Show histogram");
  f2.add(viz, "histBins", 20, 200)
    .step(1)
    .name("Histogram bins")
    .onChange(function (value) {
      viz.drawHistograms();
      viz.render();
    });
  f2.open();

  gui.removeFolder("Algorithm Options");
  var f3 = gui.addFolder("Algorithm Options");
  sim.mcmc.attachUI(sim.mcmc, f3);
  f3.add(sim.mcmc, "about").name("About this algorithm");
  f3.open();

  sim.animate();
};

dat.GUI.prototype.removeFolder = function (name) {
  var folder = this.__folders[name];
  if (!folder) {
    return;
  }
  folder.close();
  this.__ul.removeChild(folder.domElement.parentNode);
  delete this.__folders[name];
  this.onResize();
};
