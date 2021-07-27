## The Markov-chain Monte Carlo Interactive Gallery

*Example*: Hamiltonian Monte Carlo

<a href="https://chi-feng.github.io/mcmc-demo/app.html?algorithm=HamiltonianMC&target=banana" target="_blank"><img src="https://raw.githubusercontent.com/chi-feng/mcmc-demo/master/docs/hmc.gif" width="400" /></a>

Click on an algorithm below to view an interactive demo where you can change algorithm parameters on-the-fly:

### Standard MCMC methods 
*   [Random Walk Metropolis Hastings](https://chi-feng.github.io/mcmc-demo/app.html?algorithm=RandomWalkMH&target=banana)
*   [Adaptive Metropolis Hastings](https://chi-feng.github.io/mcmc-demo/app.html?algorithm=AdaptiveMH&target=banana) [[1]](#ref-1)
*   [Hamiltonian Monte Carlo](https://chi-feng.github.io/mcmc-demo/app.html?algorithm=HamiltonianMC&target=banana) [[2]](#ref-2)
*   [No-U-Turn Sampler](https://chi-feng.github.io/mcmc-demo/app.html?algorithm=NaiveNUTS&target=banana) [[2]](#ref-2)
*   [Metropolis-adjusted Langevin Algorithm (MALA)](https://chi-feng.github.io/mcmc-demo/app.html?algorithm=MALA&target=banana) [[3]](#ref-3)
*   [Hessian-Hamiltonian Monte Carlo (H2MC)](https://chi-feng.github.io/mcmc-demo/app.html?algorithm=H2MC&target=banana) [[4]](#ref-4)
*   [Gibbs Sampling](https://chi-feng.github.io/mcmc-demo/app.html?algorithm=GibbsSampling&target=banana)
*   [DE-MCMC-Z](https://chi-feng.github.io/mcmc-demo/app.html?algorithm=DE-MCMC-Z&target=banana) [[7]](#ref-7)

### Non-Markovian iterative sampling methods
*   [Stein Variational Gradient Descent (SVGD)](https://chi-feng.github.io/mcmc-demo/app.html?algorithm=SVGD&target=banana&delay=0) [[5]](#ref-5)
*   [Nested Sampling with RadFriends (RadFriends-NS)](https://chi-feng.github.io/mcmc-demo/app.html?algorithm=RadFriends-NS&target=banana) [[6]](#ref-6)

### References

[1] H. Haario, E. Saksman, and J. Tamminen, [An adaptive Metropolis algorithm](http://projecteuclid.org/euclid.bj/1080222083) (2001)

[2] M. D. Hoffman, A. Gelman, [The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo](http://arxiv.org/abs/1111.4246) (2011)

[3] G. O. Roberts, R. L. Tweedie, [Exponential Convergence of Langevin Distributions and Their Discrete Approximations](http://www2.stat.duke.edu/~scs/Courses/Stat376/Papers/Langevin/RobertsTweedieBernoulli1996.pdf) (1996)

[4] Li, Tzu-Mao, et al. [Anisotropic Gaussian mutations for metropolis light transport through Hessian-Hamiltonian dynamics](https://people.csail.mit.edu/tzumao/h2mc/) ACM Transactions on Graphics 34.6 (2015): 209.

[5] Q. Liu, et al. [Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm](http://www.cs.dartmouth.edu/~dartml/project.html?p=vgd) Advances in Neural Information Processing Systems. 2016.

[6] J. Buchner [A statistical test for Nested Sampling algorithms](https://arxiv.org/abs/1407.5459) Statistics and Computing. 2014.

[7] Cajo J. F. ter Braak & Jasper A. Vrugt [Differential Evolution Markov Chain with snooker updater and fewer chains](https://link.springer.com/article/10.1007/s11222-008-9104-9) Statistics and Computing. 2008.

### Running locally
Clone or download the repository and open `index.html` in your web browser. All dependencies are included in in `lib/`.

### Adding an algorithm
1. Copy one of the existing algorithms in the `algorithms` directory (a good starting point is `algorithms/HamiltonianMC.js`). 
1. in `app.html` include the your algorithm's javascript file at the bottom of the page. This will add your algorithm to the dropdown menu. 
1. Add any new visualizations to the `Visualizer.prototype.dequeue` function defined in `main/Visualizer.js`. The MCMC simulation adds visualization "events" onto an animation queue. Most common events such as accepting or rejecting a proposal have already been implemented. The renderer composites the contents of three offscreen canvases (densityCanvas, samplesCanvas, and overlayCanvas)
1. Add a link to your algorithm in `README.md` and `index.html`

## See also

Interactive Gaussian process regression demo

https://github.com/chi-feng/gp-demo

<a href="https://github.com/chi-feng/gp-demo"><img src="https://raw.githubusercontent.com/chi-feng/gp-demo/master/screenshot.png" width="400" /></a>
