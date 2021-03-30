"use strict";

MCMC.registerAlgorithm("RandomWalkMH", {
  description: "Random walk Metropolis-Hastings",

  about: () => {
    window.open("https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm");
  },

  init: (self) => {
    self.sigma = 1;
  },

  reset: (self) => {
    self.chain = [MultivariateNormal.getSample(self.dim)];
  },

  attachUI: (self, folder) => {
    folder.add(self, "sigma", 0.05, 2).step(0.05).name("Proposal &sigma;");
    folder.open();
  },

  step: (self, visualizer) => {
    const proposalDist = new MultivariateNormal(self.chain.last(), eye(self.dim).scale(self.sigma * self.sigma));
    const proposal = proposalDist.getSample();
    const logAcceptRatio = self.logDensity(proposal) - self.logDensity(self.chain.last());
    visualizer.queue.push({
      type: "proposal",
      proposal: proposal,
      proposalCov: proposalDist.cov,
    });
    if (Math.random() < Math.exp(logAcceptRatio)) {
      self.chain.push(proposal);
      visualizer.queue.push({ type: "accept", proposal: proposal });
    } else {
      self.chain.push(self.chain.last());
      visualizer.queue.push({ type: "reject", proposal: proposal });
    }
  },
});
