"use strict";

/**
 * Nested Sampling with RadFriends
 *
 * A nested sampling algorithm that uses RadFriends region sampling to
 * efficiently explore the parameter space. RadFriends constructs a region
 * from overlapping spheres centered on live points.
 *
 * Copyright: Johannes Buchner (C) 2013-2019
 * Code recycled from https://github.com/JohannesBuchner/ultranest-js
 * License: AGPL-3.0
 *
 * @see https://arxiv.org/abs/1407.5459
 */

/**
 * Represents a point in the nested sampling algorithm
 */
class NSPoint {
  constructor(L, coords, phys_coords) {
    this.L = L;
    this.coords = coords;
    this.phys_coords = phys_coords;
  }
}

/**
 * Generate a uniform random number in [0, 1)
 */
const randomUniform = () => Math.random();

/**
 * Generate a random sample from a normal distribution
 */
const randomNormal = (mu, stdev) => {
  const proposalDist = new MultivariateNormal(mu, stdev * stdev);
  return proposalDist.getSample();
};

/**
 * Generate a random integer in [imin, imax]
 */
const randomInt = (imin, imax) => {
  return Math.floor(Math.random() * (imax - imin + 1)) + imin;
};

/**
 * Compute log(exp(a) + exp(b)) in a numerically stable way
 */
const logaddexp = (a, b) => {
  if (b > a) {
    return Math.log(1 + Math.exp(a - b)) + b;
  } else {
    return Math.log(1 + Math.exp(b - a)) + a;
  }
};

/**
 * Compute squared Euclidean distance between two coordinate vectors
 */
const computeDistance = (acoords, bcoords) => {
  let distsq = 0;
  const n = acoords.length;
  for (let j = 0; j < n; j++) {
    distsq += (acoords[j] - bcoords[j]) * (acoords[j] - bcoords[j]);
  }
  return distsq;
};

/**
 * Check if squared distance is less than threshold (early termination)
 */
const computeDistanceLt = (acoords, bcoords, maxsqdistance) => {
  let distsq = 0;
  const n = acoords.length;
  for (let j = 0; j < n; j++) {
    distsq += (acoords[j] - bcoords[j]) * (acoords[j] - bcoords[j]);
    if (distsq > maxsqdistance) {
      return false;
    }
  }
  return true;
};

/**
 * Generate a random integer in [min, max) (max exclusive)
 */
const getRandomInt = (min, max) => {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min)) + min;
};

/**
 * Estimate the RadFriends radius using bootstrap resampling
 */
const nearestRdistanceGuess = (ndim, live_points) => {
  const nbootstrap_rounds = 20;
  let maxsqdistance = 0.0;
  const n = live_points.length;

  for (let j = 0; j < nbootstrap_rounds; j++) {
    const selected = [];
    const nonselected = [];

    for (let i = 0; i < n; i++) {
      const k = getRandomInt(0, n);
      if (selected.indexOf(k) === -1) {
        selected.push(k);
      }
    }

    for (let i = 0; i < n; i++) {
      if (selected.indexOf(i) === -1) {
        nonselected.push(i);
      }
    }

    for (let i = 0; i < nonselected.length; i++) {
      const a = nonselected[i];
      let b = selected[0];
      let minsqdistance = computeDistance(live_points[a].coords, live_points[b].coords);

      for (let k = 1; k < selected.length; k++) {
        b = selected[k];
        minsqdistance = Math.min(minsqdistance, computeDistance(live_points[a].coords, live_points[b].coords));
      }

      maxsqdistance = Math.max(minsqdistance, maxsqdistance);
    }
  }

  return maxsqdistance;
};

/**
 * Generate a random unit normal vector
 */
const randomNormalVector = (ndim) => {
  const direction = new MultivariateNormal(zeros(ndim, 1), eye(ndim, ndim)).getSample();
  return direction.scale(1.0 / direction.norm());
};

/**
 * RadFriends region drawer for nested sampling
 */
class RadFriendsDrawer {
  constructor(ndim, transform, likelihood) {
    this.likelihood = likelihood;
    this.transform = transform;
    this.ndim = ndim;
    this.niter = 0;
    this.maxsqdistance = NaN;
    this.phase = 1;
    this.rejected = [];
    this.initRegion();
  }

  initRegion() {
    this.region_low = [];
    this.region_high = [];

    for (let i = 0; i < this.ndim; i++) {
      this.region_low[i] = 0.0;
      this.region_high[i] = 1.0;
    }
  }

  isInside(current, members) {
    for (let i = 0; i < this.ndim; i++) {
      if (current.coords[i] < this.region_low[i]) return false;
      if (current.coords[i] > this.region_high[i]) return false;
    }

    if (!(this.maxsqdistance > 0)) {
      console.log("friends not used because maxsqdistance is " + this.maxsqdistance);
      return true;
    }

    for (let i = 0, n = members.length; i < n; i++) {
      if (computeDistanceLt(members[i].coords, current.coords, this.maxsqdistance)) {
        return true;
      }
    }

    return false;
  }

  countInside(current, members) {
    if (!(this.maxsqdistance > 0)) {
      console.log("friends not used because maxsqdistance is " + this.maxsqdistance);
      return 1;
    }

    let nnearby = 0;
    for (let i = 0; i < members.length; i++) {
      if (computeDistanceLt(members[i].coords, current.coords, this.maxsqdistance)) {
        nnearby += 1;
      }
    }

    return nnearby;
  }

  generateDirect(current, members) {
    let ntotal = 0;
    const n = members.length;

    while (true) {
      for (let j = 0; j < this.ndim; j++) {
        current.coords[j] = randomUniform() * (this.region_high[j] - this.region_low[j]) + this.region_low[j];
        current.phys_coords[j] = current.coords[j];
      }

      ntotal += 1;

      if (n === 0) {
        console.log("generate_direct(): No friends available for checking!");
        return ntotal;
      }

      if (this.isInside(current, members)) return ntotal;
      if (ntotal > 1000) return ntotal;
    }
  }

  generateFromFriends(current, members) {
    let ntotal = 0;
    const n = members.length;

    while (true) {
      ntotal += 1;
      const member = members[randomInt(0, n - 1)];
      const direction = randomNormalVector(this.ndim);
      const radius = Math.sqrt(this.maxsqdistance) * Math.pow(randomUniform(), 1.0 / this.ndim);

      for (let j = 0; j < this.ndim; j++) {
        current.coords[j] = member.coords[j] + direction[j] * radius;
        current.phys_coords[j] = current.coords[j];
      }

      ntotal += 1;

      if (this.isInside(current, members)) {
        const coin = randomUniform();
        const nnearby = this.countInside(current, members);
        if (coin < 1.0 / nnearby) return ntotal;
      }
    }
  }

  next(current, live_points) {
    this.niter += 1;
    const Lmin = current.L;
    const n = live_points.length;

    // Recompute maxsqdistance
    const newmaxsqdistance = nearestRdistanceGuess(this.ndim, live_points);
    if (!(this.maxsqdistance > 0) || newmaxsqdistance < this.maxsqdistance) {
      this.maxsqdistance = newmaxsqdistance;
    }

    for (let j = 0; j < this.ndim; j++) {
      let low = 1;
      let high = 0;

      for (let i = 0; i < n; i++) {
        const p = live_points[i];
        low = Math.min(low, p.coords[j]);
        high = Math.max(high, p.coords[j]);
      }

      this.region_low[j] = Math.max(0, low - Math.sqrt(this.maxsqdistance));
      this.region_high[j] = Math.min(1, high + Math.sqrt(this.maxsqdistance));
    }

    let ntoaccept = 0;

    if (this.phase === 0) {
      while (true) {
        const ntotal = this.generateDirect(current, live_points);
        ntoaccept += 1;
        current.phys_coords = this.transform(current.coords);
        current.L = this.likelihood(current.phys_coords);

        if (current.L >= Lmin) {
          return current;
        } else {
          this.rejected.push(current.phys_coords.copy());
        }

        if (ntotal >= 20) {
          this.phase = 1;
          break;
        }
      }
    }

    while (true) {
      ntoaccept += 1;
      const ntotal = this.generateFromFriends(current, live_points);
      current.phys_coords = this.transform(current.coords);
      current.L = this.likelihood(current.phys_coords);

      if (current.L >= Lmin) {
        return current;
      } else {
        this.rejected.push(current.phys_coords.copy());
      }
    }
  }
}

/**
 * Generate a point uniformly in the unit hypercube
 */
const generateFullspace = (ndim) => {
  const current = new NSPoint(1e300, [], []);
  for (let j = 0; j < ndim; j++) {
    current.coords[j] = randomUniform();
    current.phys_coords[j] = current.coords[j];
  }
  return current;
};

/**
 * Sort live points by likelihood
 */
const sortL = (live_points) => {
  live_points.sort((a, b) => {
    if (a.L < b.L) return -1;
    if (a.L > b.L) return 1;
    return 0;
  });
};

/**
 * Get posterior weights from weighted samples
 */
const getPosteriorWeights = (weighted_samples) => {
  const probs = [];
  let logmax = weighted_samples[0][0] + weighted_samples[0][1].L;

  for (let i = 0; i < weighted_samples.length; i++) {
    logmax = Math.max(logmax, weighted_samples[i][0] + weighted_samples[i][1].L);
  }

  for (let i = 0; i < weighted_samples.length; i++) {
    probs[i] = Math.exp(weighted_samples[i][0] + weighted_samples[i][1].L - logmax);
  }

  return probs;
};

/**
 * Nested sampler main class
 */
class NestedSampler {
  constructor(ndim, drawer, nlive_points, transform, likelihood) {
    this.nlive_points = nlive_points;
    this.transform = transform;
    this.likelihood = likelihood;
    this.ndim = ndim;
    this.Lmax = NaN;
    this.remainderZ = NaN;
    this.ndraws = 0;
    this.drawer = drawer;
    this.live_points = [];
    this.latest_point = NaN;

    this.generateLivePoints();
  }

  generateLivePoints() {
    for (let i = 0; i < this.nlive_points; i++) {
      const current = generateFullspace(this.ndim);
      current.phys_coords = this.transform(current.coords);
      current.L = this.likelihood(current.phys_coords);

      if (i === 0) {
        this.Lmax = current.L;
      } else {
        this.Lmax = Math.max(this.Lmax, current.L);
      }

      this.live_points[i] = current;
      this.latest_point = current;
    }

    sortL(this.live_points);
  }

  next() {
    const i = 0;
    const lowest = this.live_points[i];
    const replacement = new NSPoint(lowest.L, lowest.coords.slice(), lowest.phys_coords.slice());
    const ndraws = this.drawer.next(replacement, this.live_points);

    this.live_points[i] = replacement;
    this.latest_point = replacement;
    sortL(this.live_points);
    this.ndraws += ndraws;

    return lowest;
  }

  integrateRemainder(logwidth, logVolremaining, logZ) {
    const n = this.nlive_points;
    const logV = logwidth;
    const L0 = this.live_points[this.live_points.length - 1].L;
    let Lmax = 0;
    let Lmin = 0;
    let Lmid = 0;

    for (let i = 0; i < n; i++) {
      const Ldiff = Math.exp(this.live_points[i].L - L0);
      if (i > 0) Lmax += Ldiff;
      if (i === n - 1) Lmax += Ldiff;
      if (i < n - 1) Lmin += Ldiff;
      if (i === 0) Lmin += Ldiff;
      Lmid += Ldiff;
    }

    const logZmid = logaddexp(logZ, logV + Math.log(Lmid) + L0);
    const logZup = logaddexp(logZ, logV + Math.log(Lmax) + L0);
    const logZlo = logaddexp(logZ, logV + Math.log(Lmin) + L0);
    const logZerr = Math.max(logZup - logZmid, logZmid - logZlo);

    this.remainderZ = logV + Math.log(Lmid) + L0;
    this.remainderZerr = logZerr;

    const points = [];
    for (let i = 0; i < n; i++) {
      points[i] = [logwidth, this.live_points[i]];
    }

    return points;
  }
}

/**
 * Nested sampling integrator
 */
class NSIntegrator {
  constructor(ndim, transform, likelihood, data_calc, nlive_points, frac_remain, maxiter) {
    this.drawer = new RadFriendsDrawer(ndim, transform, likelihood);
    this.sampler = new NestedSampler(ndim, this.drawer, nlive_points, transform, likelihood);
    this.current = this.sampler.next();

    this.logVolremaining = 0;
    this.logwidth = Math.log(1 - Math.exp(-1.0 / nlive_points));

    this.iter = 0;
    this.weights = [];
    this.results = [];
    this.wi = this.logwidth + this.current.L;
    this.logZ = this.wi;
    this.H = this.current.L - this.logZ;
    this.logZerr = NaN;

    this.nlive_points = nlive_points;
    this.frac_remain = frac_remain;
    this.maxiter = maxiter;

    console.log("integrator[initial]: ln Z = " + this.logZ + " " + this.H + " " + this.wi + " " + this.current.L);
  }

  progress() {
    this.logwidth = Math.log(1 - Math.exp(-1.0 / this.nlive_points)) + this.logVolremaining;
    this.logVolremaining -= 1.0 / this.nlive_points;

    this.weights[this.iter] = [this.logwidth, this.current];

    this.iter += 1;
    this.logZerr = Math.sqrt(this.H / this.nlive_points);

    this.sampler.integrateRemainder(this.logwidth, this.logVolremaining, this.logZ);

    const total_error = this.logZerr + this.sampler.remainderZerr;

    if (Math.exp(this.sampler.remainderZ - this.logZ) < this.frac_remain) {
      console.log("Nested sampling integrator has walked through the most of the posterior and reached convergence.");
      return 0;
    }

    if (this.maxiter > 0 && this.iter > this.maxiter) {
      console.log("Nested sampling integrator has reached the number of iterations limit.");
      return 0;
    }

    this.current = this.sampler.next();
    this.wi = this.logwidth + this.current.L;
    const logZnew = logaddexp(this.logZ, this.wi);
    this.H = Math.exp(this.wi - logZnew) * this.current.L + Math.exp(this.logZ - logZnew) * (this.H + this.logZ) - logZnew;
    this.logZ = logZnew;

    if (this.iter % 50 === 0) {
      console.log(
        "integrator[" +
          this.iter +
          "]: current ln Z = " +
          this.logZ +
          " +- " +
          this.logZerr +
          " +- " +
          this.sampler.remainderZerr
      );
    }

    return 1;
  }

  getResults() {
    const remainder_weights = this.sampler.integrateRemainder(this.logwidth, this.logVolremaining, this.logZ);
    let logZtotal = this.logZ;
    let Htotal = this.H;

    for (let i = 0; i < remainder_weights.length; i++) {
      const Li = remainder_weights[i][1].L;
      const wi = this.logwidth + Li;
      const logZnew = logaddexp(logZtotal, wi);
      Htotal = Math.exp(wi - logZnew) * Li + Math.exp(logZtotal - logZnew) * (Htotal + logZtotal) - logZnew;
      logZtotal = logZnew;
    }

    const logZerrfinal = Math.sqrt(Htotal / this.nlive_points) + this.sampler.remainderZerr;
    const logZfinal = logaddexp(logZtotal, this.sampler.remainderZ);

    return [logZfinal, logZerrfinal, this.weights.concat(remainder_weights)];
  }
}

/**
 * Transform from unit cube to parameter space
 */
const nsTransform = (cube) => {
  const params = zeros(cube.length, 1);
  for (let i = 0; i < params.length; i++) {
    params[i] = cube[i] * 10 - 5;
  }
  return params;
};

/**
 * Nested Sampling with RadFriends Algorithm
 */
MCMC.registerAlgorithm("RadFriends-NS", {
  description: "Nested Sampling with RadFriends",

  about: () => {
    window.open("https://arxiv.org/abs/1407.5459");
  },

  init: (self) => {
    self.live_points = [];
    self.nlive_points = 40;
    self.iter = 0;
    self.wait_iter = 0;
    self.reset(self);
    self.wait_iter = 0;
  },

  reset: (self) => {
    self.iter = 0;
    self.wait_iter = 0;
    self.chain = [];
    self.integrator = new NSIntegrator(self.dim, nsTransform, self.logDensity, null, self.nlive_points, 0.01, 0);
  },

  attachUI: (self, folder) => {
    folder.add(self, "nlive_points", 10, 400).step(1).name("numLivePoints");
    folder.open();
  },

  step: (self, visualizer) => {
    // Point about to be removed
    const lowest = self.integrator.sampler.live_points[0].phys_coords.slice();
    const previous = self.integrator.current.phys_coords.slice();

    const r = self.integrator.progress();

    // Visualize the RadFriends region as overlapping circles
    const x = [];
    const rad = Math.sqrt(self.integrator.drawer.maxsqdistance) * 10;

    for (let i = 0; i < self.integrator.sampler.live_points.length; i++) {
      x.push(self.integrator.sampler.live_points[i].phys_coords.slice());
    }

    visualizer.queue.push({ type: "radfriends-region", x: x, r: rad });

    if (r === 0) {
      // We are done/converged
      if (self.wait_iter === 0) {
        window.alert("Done!");
      }
      self.wait_iter++;
      return;
    }

    console.log("rejected " + self.integrator.drawer.rejected.length + " points");

    visualizer.queue.push({
      type: "ns-dead-point",
      proposal: self.integrator.sampler.latest_point.phys_coords,
      deadpoint: previous,
      rejected: self.integrator.drawer.rejected,
    });

    self.integrator.drawer.rejected = [];

    const results = self.integrator.getResults();
    const weighted_samples = results[2];

    // Return all samples with weights
    const weights = getPosteriorWeights(weighted_samples);
    const samples = [];

    for (let i = 0; i < weighted_samples.length; i++) {
      samples.push(weighted_samples[i][1].phys_coords);
    }

    self.chain = samples;
    self.chain_weights = weights;
  },
});
