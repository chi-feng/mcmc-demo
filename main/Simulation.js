"use strict";

/**
 * Class representing a simulation for MCMC algorithms and target distributions.
 */
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

  /**
   * Set the MCMC algorithm to be used in the simulation.
   * @param {string} algorithmName - Name of the algorithm to set.
   */
  setAlgorithm(algorithmName) {
    console.log(`Setting algorithm to ${algorithmName}`);
    this.hasAlgorithm = true;
    this.algorithm = algorithmName;
    this.mcmc.initialized = false;
    const algorithm = MCMC.algorithms[algorithmName];
    this.mcmc.description = algorithm.description;
    this.mcmc.init = algorithm.init;
    this.mcmc.reset = algorithm.reset;
    this.mcmc.step = algorithm.step;
    this.mcmc.attachUI = algorithm.attachUI;
    this.mcmc.about = algorithm.about;

    document.getElementById("info").innerHTML = this.mcmc.description;

    if (this.hasAlgorithm && this.hasTarget) {
      if (!this.mcmc.initialized) this.mcmc.init(this.mcmc);
      this.mcmc.reset(this.mcmc);
      this.mcmc.initialized = true;
      this.visualizer.resize();
    }
  }

  /**
   * Set the target distribution for the simulation.
   * @param {string} targetName - Name of the target distribution to set.
   */
  setTarget(targetName) {
    console.log(`Setting target to ${targetName}`, MCMC.targets[targetName]);
    this.hasTarget = true;
    this.target = targetName;
    const target = MCMC.targets[targetName];
    this.mcmc.logDensity = target.logDensity;
    this.mcmc.gradLogDensity = target.gradLogDensity;

    // Update visualizer extents
    this.visualizer.xmin = target.xmin;
    this.visualizer.xmax = target.xmax;
    this.visualizer.resize();

    // Use finite differences to approximate Hessian
    const grad = this.mcmc.gradLogDensity;
    const N = this.mcmc.dim;
    const h = 1e-8;
    this.mcmc.hessLogDensity = (x) => {
      const hess = zeros(N, N);
      const Delta = eye(N, N).scale(h);
      for (let i = 0; i < N; ++i) {
        for (let j = 0; j < N; ++j) {
          hess[i * N + j] =
            (grad(x.add(Delta.col(j)))[i] - grad(x)[i]) / (2 * h) +
            (grad(x.add(Delta.col(i)))[j] - grad(x)[j]) / (2 * h);
        }
      }
      return hess;
    };

    // Update contours
    const { xmin, xmax, ymin, ymax } = this.visualizer;
    const nx = 480,
      ny = 256,
      nz = 7;
    this.computeContours(
      this.mcmc.logDensity,
      xmin,
      xmax,
      ymin,
      ymax,
      nx,
      ny,
      nz
    );

    if (this.mcmc.initialized) this.mcmc.reset(this.mcmc);
    if (this.hasAlgorithm && this.hasTarget) {
      if (!this.mcmc.initialized) this.mcmc.init(this.mcmc);
      this.mcmc.reset(this.mcmc);
      this.mcmc.initialized = true;
      this.visualizer.resize();
    }
  }

  /**
   * Compute contours for the log density function.
   * @param {Function} logDensity - Log density function.
   * @param {number} xmin - Minimum x value.
   * @param {number} xmax - Maximum x value.
   * @param {number} ymin - Minimum y value.
   * @param {number} ymax - Maximum y value.
   * @param {number} nx - Number of x grid points.
   * @param {number} ny - Number of y grid points.
   * @param {number} nz - Number of contour levels.
   */
  computeContours(logDensity, xmin, xmax, ymin, ymax, nx, ny, nz) {
    const x = linspace(xmin, xmax, nx);
    const y = linspace(ymin, ymax, ny);
    const data = [];
    const point = zeros(2, 1);
    let min = 1e10,
      max = 0;

    for (let i = 0; i < nx; ++i) {
      data.push([]);
      point[0] = x[i];
      for (let j = 0; j < ny; ++j) {
        point[1] = y[j];
        const val = Math.exp(logDensity(point));
        data[i].push(val);
        if (val > max) max = val;
        if (val < min) min = val;
      }
    }

    const z = linspace(min + 0.01 * (max - min), max - 0.02 * (max - min), nz);
    const c = new Conrec();
    c.contour(data, 0, nx - 1, 0, ny - 1, x, y, nz, z);
    const contours = c.contourList();

    this.mcmc.contours = [];
    this.mcmc.contourData = data;
    this.mcmc.contourLevels = z;
    for (const contour of contours) {
      this.mcmc.contours.push(contour.map((pt) => [pt.x, pt.y]));
    }

    // Numerically integrate to get marginal densities
    this.mcmc.xgrid = x;
    this.mcmc.ygrid = y;
    this.mcmc.marginals = [zeros(nx), zeros(ny)];

    for (let i = 0; i < nx; ++i) {
      for (let j = 0; j < ny; ++j) {
        this.mcmc.marginals[0][i] += data[i][j];
      }
    }
    this.mcmc.marginals[0] = this.mcmc.marginals[0].scale(
      1.0 / this.mcmc.marginals[0].maxCoeff()
    );

    for (let j = 0; j < ny; ++j) {
      for (let i = 0; i < nx; ++i) {
        this.mcmc.marginals[1][j] += data[i][j];
      }
    }
    this.mcmc.marginals[1] = this.mcmc.marginals[1].scale(
      1.0 / this.mcmc.marginals[1].maxCoeff()
    );

    const buffer = document.createElement("canvas");
    buffer.width = nx;
    buffer.height = ny;
    const context = buffer.getContext("2d");
    const image = context.createImageData(nx, ny);

    for (let j = 0; j < ny; ++j) {
      for (let i = 0; i < nx; ++i) {
        const base = 4 * ((ny - 1 - j) * nx + i);
        const value = Math.sqrt((data[i][j] - min) / (max - min)) * 255;
        image.data[base] = 102;
        image.data[base + 1] = 153;
        image.data[base + 2] = 187;
        image.data[base + 3] = value | 0;
      }
    }
    context.putImageData(image, 0, 0);
    this.mcmc.densityCanvas = buffer;
  }

  /**
   * Reset the MCMC simulation.
   */
  reset() {
    this.mcmc.reset(this.mcmc);
    this.visualizer.resize();
  }

  /**
   * Perform a simulation step.
   */
  step() {
    if (this.visualizer.queue.length === 0) {
      this.mcmc.step(this.mcmc, this.visualizer);
    }
    if (!this.visualizer.animateProposal) {
      while (this.visualizer.queue.length > 0) {
        this.visualizer.dequeue();
      }
    } else {
      this.visualizer.dequeue();
    }
    this.visualizer.render();
  }

  /**
   * Animate the simulation.
   */
  animate() {
    const self = this;
    if (this.autoplay || this.visualizer.tweening) {
      this.step();
    }
    setTimeout(
      () => {
        requestAnimationFrame(() => {
          self.animate();
        });
      },
      this.visualizer.tweening ? self.tweeningDelay : self.delay
    );
  }
}

let viz, sim, gui;

/**
 * Get URL query parameters as an object.
 * @returns {Object} URL query parameters.
 */
function getUrlVars() {
  const vars = {};
  const pairs = window.location.search.substr(1).split("&");
  for (const pair of pairs) {
    const [key, value] = pair.split("=");
    vars[key] = value && decodeURIComponent(value.replace(/\+/g, " "));
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

  let algorithm = MCMC.algorithmNames[0];
  let target = MCMC.targetNames[0];
  let seed = Math.seedrandom();

  /**
   * Parse a boolean value from a string.
   * @param {string} value - String to parse.
   * @returns {boolean} Parsed boolean.
   */
  function parseBool(value) {
    return value === "true";
  }

  if (window.location.search !== "") {
    const queryParams = getUrlVars();

    if (
      queryParams.algorithm &&
      MCMC.algorithmNames.includes(queryParams.algorithm)
    ) {
      algorithm = queryParams.algorithm;
    }
    if (queryParams.target && MCMC.targetNames.includes(queryParams.target)) {
      target = queryParams.target;
    }
    if (queryParams.seed) {
      // Reseed
      seed = Math.seedrandom(queryParams.seed);
      console.log(`Setting seed to ${seed}`);
    }

    const config = [
      ["delay", parseInt, sim, "sim"],
      ["tweeningDelay", parseInt, sim, "sim"],
      ["autoplay", parseBool, sim, "sim"],
      ["animateProposal", parseBool, viz, "viz"],
      ["showSamples", parseBool, viz, "viz"],
      ["showHistograms", parseBool, viz, "viz"],
      ["histBins", parseInt, viz, "viz"],
    ];

    for (const [param, parse, obj, objName] of config) {
      if (param in queryParams) {
        const value = parse(queryParams[param]);
        console.log(`Setting ${objName}.${param} to ${value}`);
        obj[param] = value;
      }
    }
  }

  sim.setAlgorithm(algorithm);
  sim.setTarget(target);
  sim.mcmc.init(sim.mcmc);

  window.onresize = () => {
    viz.resize();
  };

  gui = new dat.GUI({ width: 300 });

  const f1 = gui.addFolder("Simulation options");
  f1.add(sim, "algorithm", MCMC.algorithmNames)
    .name("Algorithm")
    .onChange((value) => {
      sim.setAlgorithm(value);
      gui.removeFolder("Algorithm Options");
      const f = gui.addFolder("Algorithm Options");
      sim.mcmc.attachUI(sim.mcmc, f);
      f.open();
    });
  f1.add(sim, "target", MCMC.targetNames)
    .name("Target distribution")
    .onChange((value) => {
      sim.setTarget(value);
    });
  f1.add(sim, "autoplay").name("Autoplay");
  f1.add(sim, "delay", 0, 1000)
    .name("Autoplay delay")
    .onChange((value) => {
      viz.animateProposal = value !== 0;
    });
  f1.add(sim, "tweeningDelay", 0, 200).name("Tweening delay");
  f1.add(sim, "step").name("Step");
  f1.add(sim, "reset").name("Reset");
  f1.open();

  const f2 = gui.addFolder("Visualization Options");
  f2.add(viz, "animateProposal").name("Animate proposal").listen();
  f2.add(viz, "showTargetDensity").name("Show target");
  f2.add(viz, "showSamples").name("Show samples");
  f2.add(viz, "showHistograms").name("Show histogram");
  f2.add(viz, "histBins", 20, 200)
    .step(1)
    .name("Histogram bins")
    .onChange(() => {
      viz.drawHistograms();
      viz.render();
    });
  f2.open();

  gui.removeFolder("Algorithm Options");
  const f3 = gui.addFolder("Algorithm Options");
  sim.mcmc.attachUI(sim.mcmc, f3);
  f3.add(sim.mcmc, "about").name("About this algorithm");
  f3.open();

  sim.animate();
};

/**
 * Remove a folder from dat.GUI.
 * @param {string} name - Name of the folder to remove.
 */
dat.GUI.prototype.removeFolder = function (name) {
  const folder = this.__folders[name];
  if (!folder) {
    return;
  }
  folder.close();
  this.__ul.removeChild(folder.domElement.parentNode);
  delete this.__folders[name];
  this.onResize();
};
