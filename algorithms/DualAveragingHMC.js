"use strict";

MCMC.registerAlgorithm("DualAveragingHMC", {
  description: "Hamiltonian Monte Carlo with Dual Averaging",

  about: () => {
    window.open("https://en.wikipedia.org/wiki/Hybrid_Monte_Carlo");
  },

  init: (self) => {
    self.lambda = 1.6;
    self.delta = 0.65;
    self.M_adapt = 100;

    //viz.animateProposal = false;

    self.joint = (theta, r) => {
      return Math.exp(self.logDensity(theta) - r.norm2() / 2);
    };

    self.leapFrog = (theta, r, epsilon) => {
      const r_ = r.add(self.gradLogDensity(theta).scale(epsilon / 2));
      const theta_ = theta.add(r_.scale(epsilon));
      r_.increment(self.gradLogDensity(theta_).scale(epsilon / 2));
      return { theta: theta_, r: r_ };
    };

    self.findReasonableEpsilon = (theta) => {
      let epsilon = 0.1;
      const r = MultivariateNormal.getSample(self.dim);
      let result = self.leapFrog(theta, r, epsilon);
      const a = 2 * (self.joint(result.theta, result.r) / self.joint(theta, r) > 0.5 ? 1 : 0) - 1;
      while (Math.pow(self.joint(result.theta, result.r) / self.joint(theta, r), a) > Math.pow(2.0, -a)) {
        epsilon = Math.pow(2, a) * epsilon;
        result = self.leapFrog(result.theta, result.r, epsilon);
      }
      return Math.max(1e-3, epsilon);
    };
  },

  reset: (self) => {
    self.chain = [MultivariateNormal.getSample(self.dim)];
    self.epsilon = [self.findReasonableEpsilon(self.chain.last())];
    self.mu = Math.log(10 * self.epsilon[0]);
    self.epsilon_bar = [1.0];
    self.H_bar = [1.0];

    self.gamma = 0.2;
    self.t0 = 10;
    self.kappa = 0.75;

    self.accepted = 0;
  },

  attachUI: (self, folder) => {
    folder
      .add(self, "lambda", 0.1, 2)
      .step(0.1)
      .name("&lambda; = &epsilon;L")
      .onChange((value) => {
        sim.reset();
      });
    folder
      .add(self, "delta", 0.1, 1)
      .step(0.05)
      .name("&delta;")
      .onChange((value) => {
        sim.reset();
      });
    folder.open();
  },

  step: (self, visualizer) => {
    const r0 = MultivariateNormal.getSample(self.dim);
    let theta = self.chain.last().copy();
    let r = r0.copy();
    let Lm = Math.max(1, Math.round(self.lambda / self.epsilon.last()));
    if (Lm > 100) {
      console.log("Lm > 100", Lm);
      Lm = 100;
    }
    const trajectory = [theta.copy()];
    for (let i = 0; i < Lm; ++i) {
      const result = self.leapFrog(theta, r, self.epsilon.last());
      theta = result.theta;
      r = result.r;
      trajectory.push(theta.copy());
    }
    const epsilon = ((self.epsilon.last() * 1000) | 0) / 1000;
    visualizer.queue.push({
      type: "proposal",
      proposal: theta,
      trajectory: trajectory,
      initialMomentum: r0,
      epsilon: epsilon,
      alpha: self.delta - self.H_bar.last(),
    });
    const alpha = Math.min(1, self.joint(theta, r) / self.joint(self.chain.last(), r0));
    if (Math.random() < alpha) {
      self.chain.push(theta);
      visualizer.queue.push({ type: "accept", proposal: theta });
      self.accepted++;
    } else {
      self.chain.push(self.chain.last().copy());
      visualizer.queue.push({ type: "reject", proposal: theta });
    }
    const m = self.chain.length;
    if (m <= self.M_adapt) {
      self.H_bar.push((1 - 1 / (m + self.t0)) * self.H_bar.last() + (1 / (m + self.t0)) * (self.delta - alpha));
      const log_epsilon = self.mu - (Math.sqrt(m) / self.gamma) * self.H_bar.last();
      self.epsilon.push(Math.exp(log_epsilon));
      self.epsilon_bar.push(
        Math.exp(
          Math.pow(m, -self.kappa) * log_epsilon + (1 - Math.pow(m, -self.kappa)) * Math.log(self.epsilon_bar.last())
        )
      );
    } else {
      self.epsilon.push(self.epsilon_bar.last());
    }
  },
});
