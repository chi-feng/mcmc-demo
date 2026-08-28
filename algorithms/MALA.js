"use strict";

/**
 * Metropolis-adjusted Langevin Algorithm (MALA)
 *
 * Uses gradient information to guide proposals toward high-probability regions.
 * The proposal distribution is a Gaussian centered at the current position
 * plus a drift term proportional to the gradient of the log density.
 *
 * @see http://projecteuclid.org/euclid.bj/1178291835
 */
MCMC.registerAlgorithm("MALA", {
  description: "Metropolis-adjusted Langevin algorithm",

  about: () => {
    window.open("http://projecteuclid.org/euclid.bj/1178291835");
  },

  init: (self) => {
    self.sigma = 0.5;
  },

  reset: (self) => {
    self.chain = [MultivariateNormal.getSample(self.dim)];
  },

  attachUI: (self, folder) => {
    folder.add(self, "sigma", 0.1, 1).step(0.05).name("Proposal &sigma;");
    folder.open();
  },

  step: (self, visualizer) => {
    const gradient = self.gradLogDensity(self.chain.last());
    const Zdist = new MultivariateNormal(zeros(self.dim), eye(self.dim).scale(self.sigma * self.sigma));
    const Z = Zdist.getSample();
    const proposal = self.chain
      .last()
      .add(Z)
      .add(gradient.scale((self.sigma * self.sigma) / 2));

    const logProposalDensity = (x, y) => {
      return (
        -y
          .subtract(x)
          .subtract(self.gradLogDensity(x).scale((self.sigma * self.sigma) / 2))
          .norm2() /
          (2 * self.sigma * self.sigma) -
        (self.dim / 2) * Math.log(2 * Math.PI * self.sigma * self.sigma)
      );
    };

    const logNumerator = self.logDensity(proposal) + logProposalDensity(proposal, self.chain.last());
    const logDenominator = self.logDensity(self.chain.last()) + logProposalDensity(self.chain.last(), proposal);
    const logAcceptRatio = logNumerator - logDenominator;

    visualizer.queue.push({
      type: "proposal",
      proposal: proposal.copy(),
      proposalCov: Zdist.cov.copy(),
      gradient: gradient.scale((self.sigma * self.sigma) / 2),
    });

    if (Math.random() < Math.exp(logAcceptRatio)) {
      self.chain.push(proposal);
      visualizer.queue.push({ type: "accept", proposal: proposal.copy() });
    } else {
      self.chain.push(self.chain.last());
      visualizer.queue.push({ type: "reject", proposal: proposal.copy() });
    }
  },
});
