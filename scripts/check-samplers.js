/**
 * Statistical sanity checks for the samplers, run in Node with no browser.
 *
 * Usage: node scripts/check-samplers.js
 *
 * The script concatenates the library and algorithm files, replaces
 * Math.random with a seeded generator, runs each sampler on the standard
 * Gaussian target, and checks the sample mean and variance per coordinate.
 * These are regression detectors with loose tolerances, not proofs of
 * correctness: a sampler whose chain collapses or drifts fails clearly.
 */
const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");

// Seeded RNG (mulberry32) for reproducible runs
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
Math.random = mulberry32(20260829);

// Browser stubs used by the algorithm files
global.window = { open() {}, alert() {} };

const FILES = [
  "lib/linalg.core.js",
  "lib/linalg.opt.js",
  "main/MultivariateNormal.js",
  "main/MCMC.js",
  "algorithms/RandomWalkMH.js",
  "algorithms/AdaptiveMH.js",
  "algorithms/HamiltonianMC.js",
  "algorithms/MALA.js",
  "algorithms/NaiveNUTS.js",
  "algorithms/EfficientNUTS.js",
  "algorithms/DualAveragingHMC.js",
  "algorithms/DualAveragingNUTS.js",
  "algorithms/H2MC.js",
  "algorithms/GibbsSampling.js",
  "algorithms/DE-MCMC-Z.js",
  "algorithms/SVGD.js",
  "algorithms/MCHMC.js",
];

// The driver runs inside the same eval scope as the library files, because
// their top-level const/let declarations do not escape an indirect eval.
function runChecks() {
  /* global MCMC, zeros, eye */

  // Each entry: [algorithm, steps, tolerances]. Tolerances are per-coordinate
  // bounds on |mean| and on the variance interval for the standard Gaussian.
  const CHECKS = [
    ["RandomWalkMH", 20000, { mean: 0.15, varLo: 0.75, varHi: 1.3 }],
    ["AdaptiveMH", 20000, { mean: 0.15, varLo: 0.75, varHi: 1.3 }],
    ["HamiltonianMC", 5000, { mean: 0.15, varLo: 0.75, varHi: 1.3 }],
    ["MALA", 20000, { mean: 0.15, varLo: 0.75, varHi: 1.3 }],
    ["NaiveNUTS", 3000, { mean: 0.15, varLo: 0.75, varHi: 1.3 }],
    ["EfficientNUTS", 3000, { mean: 0.15, varLo: 0.75, varHi: 1.3 }],
    ["DualAveragingHMC", 5000, { mean: 0.15, varLo: 0.75, varHi: 1.3 }],
    ["DualAveragingNUTS", 3000, { mean: 0.15, varLo: 0.75, varHi: 1.3 }],
    ["H2MC", 20000, { mean: 0.15, varLo: 0.75, varHi: 1.3 }],
    ["GibbsSampling", 5000, { mean: 0.15, varLo: 0.75, varHi: 1.3 }],
    ["DE-MCMC-Z", 20000, { mean: 0.15, varLo: 0.75, varHi: 1.3 }],
    // MCHMC carries a step-size-dependent asymptotic bias, so its interval is
    // wider; the pre-fix gradient-flow bug collapsed the variance far below it
    ["MicrocanonicalHamiltonianMC", 8000, { mean: 0.25, varLo: 0.5, varHi: 1.5 }],
    // SVGD particles underdisperse slightly at finite particle counts
    ["SVGD", 1500, { mean: 0.2, varLo: 0.55, varHi: 1.3 }],
  ];

  function makeSelf(name) {
    const target = MCMC.targets["standard"];
    const self = {
      dim: 2,
      logDensity: target.logDensity.bind(target),
      gradLogDensity: target.gradLogDensity.bind(target),
      xmin: target.xmin,
      xmax: target.xmax,
    };
    // Finite-difference Hessian, as Simulation.setTarget builds it
    const grad = self.gradLogDensity;
    const h = 1e-8;
    self.hessLogDensity = (x) => {
      const hess = zeros(2, 2);
      const Delta = eye(2, 2).scale(h);
      for (let i = 0; i < 2; ++i) {
        for (let j = 0; j < 2; ++j) {
          hess[i * 2 + j] =
            (grad(x.add(Delta.col(j)))[i] - grad(x)[i]) / (2 * h) +
            (grad(x.add(Delta.col(i)))[j] - grad(x)[j]) / (2 * h);
        }
      }
      return hess;
    };
    const algorithm = MCMC.algorithms[name];
    // Simulation.setAlgorithm exposes reset on the mcmc object; SVGD.init calls it
    self.reset = algorithm.reset;
    algorithm.init(self);
    algorithm.reset(self);
    return { self, algorithm };
  }

  function moments(points, burn) {
    const kept = points.slice(burn);
    const n = kept.length;
    const mean = [0, 0];
    for (const p of kept) {
      mean[0] += p[0] / n;
      mean[1] += p[1] / n;
    }
    const variance = [0, 0];
    for (const p of kept) {
      variance[0] += Math.pow(p[0] - mean[0], 2) / n;
      variance[1] += Math.pow(p[1] - mean[1], 2) / n;
    }
    return { mean, variance };
  }

  let failures = 0;

  for (const [name, steps, tol] of CHECKS) {
    const { self, algorithm } = makeSelf(name);
    const visualizer = { queue: [] };
    for (let i = 0; i < steps; i++) {
      visualizer.queue.length = 0;
      algorithm.step(self, visualizer);
    }
    const burn = name === "SVGD" ? 0 : Math.floor(self.chain.length / 2);
    const { mean, variance } = moments(self.chain, burn);
    const ok =
      Math.abs(mean[0]) < tol.mean &&
      Math.abs(mean[1]) < tol.mean &&
      variance[0] > tol.varLo &&
      variance[0] < tol.varHi &&
      variance[1] > tol.varLo &&
      variance[1] < tol.varHi;
    if (!ok) failures++;
    console.log(
      `${ok ? "PASS" : "FAIL"} ${name.padEnd(28)} mean=[${mean.map((v) => v.toFixed(3)).join(", ")}] ` +
        `var=[${variance.map((v) => v.toFixed(3)).join(", ")}]`
    );
  }

  // Regression check: raising the SVGD particle count must add exactly the deficit
  {
    const { self, algorithm } = makeSelf("SVGD");
    const visualizer = { queue: [] };
    algorithm.step(self, visualizer);
    self.n = 300;
    visualizer.queue.length = 0;
    algorithm.step(self, visualizer);
    const ok = self.chain.length === 300;
    if (!ok) failures++;
    console.log(`${ok ? "PASS" : "FAIL"} SVGD resize to 300 particles     count=${self.chain.length}`);
  }

  return failures;
}

const source = FILES.map((f) => fs.readFileSync(path.join(root, f), "utf8")).join("\n");
const failures = (0, eval)(`${source}\n(${runChecks.toString()})();`);
process.exit(failures === 0 ? 0 : 1);
