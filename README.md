## The Markov-chain Monte Carlo Interactive Gallery

*Example*: Hamiltonian Monte Carlo

<img src="https://raw.githubusercontent.com/chi-feng/mcmc-demo/master/docs/hmc.gif" width="400" />

Click on an algorithm below to view an interactive demo where you can change algorithm parameters on-the-fly:

### Standard MCMC methods 
*   [Random Walk Metropolis Hastings](http://chi-feng.github.io/mcmc-demo/app.html?algorithm=RandomWalkMH&target=banana)
*   [Adaptive Metropolis Hastings](http://chi-feng.github.io/mcmc-demo/app.html?algorithm=AdaptiveMH&target=banana) [[1]](#ref-1)
*   [Hamiltonian Monte Carlo](http://chi-feng.github.io/mcmc-demo/app.html?algorithm=HamiltonianMC&target=banana) [[2]](#ref-2)
*   [No-U-Turn Sampler](http://chi-feng.github.io/mcmc-demo/app.html?algorithm=NaiveNUTS&target=banana) [[2]](#ref-2)
*   [Metropolis-adjusted Langevin Algorithm (MALA)](http://chi-feng.github.io/mcmc-demo/app.html?algorithm=MALA&target=banana) [[3]](#ref-3)
*   [Hessian-Hamiltonian Monte Carlo (H2MC)](http://chi-feng.github.io/mcmc-demo/app.html?algorithm=H2MC&target=banana) [[4]](#ref-4)
*   [Gibbs Sampling](http://chi-feng.github.io/mcmc-demo/app.html?algorithm=GibbsSampling&target=banana)

### Non-Markovian iterative sampling methods
*   [Stein Variational Gradient Descent (SVGD)](http://chi-feng.github.io/mcmc-demo/app.html?algorithm=SVGD&target=banana) [[5]](#ref-5)
*   [Nested Sampling with RadFriends (RadFriends-NS)](http://chi-feng.github.io/mcmc-demo/app.html?algorithm=RadFriends-NS&target=banana) [[6]](#ref-6)

### References

[1] H. Haario, E. Saksman, and J. Tamminen, [An adaptive Metropolis algorithm](http://projecteuclid.org/euclid.bj/1080222083) (2001)

[2] M. D. Hoffman, A. Gelman, [The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo](http://arxiv.org/abs/1111.4246) (2011)

[3] G. O. Roberts, R. L. Tweedie, [Exponential Convergence of Langevin Distributions and Their Discrete Approximations](http://www2.stat.duke.edu/~scs/Courses/Stat376/Papers/Langevin/RobertsTweedieBernoulli1996.pdf) (1996)

[4] Li, Tzu-Mao, et al. [Anisotropic Gaussian mutations for metropolis light transport through Hessian-Hamiltonian dynamics](https://people.csail.mit.edu/tzumao/h2mc/) ACM Transactions on Graphics 34.6 (2015): 209.

[5] Q. Liu, et al. [Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm](http://www.cs.dartmouth.edu/~dartml/project.html?p=vgd) Advances in Neural Information Processing Systems. 2016.

[6] J. Buchner [A statistical test for Nested Sampling algorithms](https://arxiv.org/abs/1407.5459) Statistics and Computing. 2014.

### Running locally
Clone or download the repository and open `index.html` in your web browser. All dependencies are included in in `lib/`.

### Adding an algorithm
1. Copy one of the existing algorithms in the `algorithms` directory (a good starting point is `algorithms/HamiltonianMC.js`). 
1. in `app.html` include the your algorithm's javascript file at the bottom of the page. This will add your algorithm to the dropdown menu. 
1. Add any new visualizations to the `Visualizer.prototype.dequeue` function defined in `main/Visualizer.js`. The MCMC simulation adds visualization "events" onto an animation queue. Most common events such as accepting or rejecting a proposal have already been implemented. The renderer composites the contents of three offscreen canvases (densityCanvas, samplesCanvas, and overlayCanvas)
1. Add a link to your algorithm in `README.md` and `index.html`

### A note on linear algebra in Javascript
1. There is a lightweight linear algebra library in `lib/linalg.core.js`
1. It works by "overloading" the built-in Float64Array type by adding the `rows` and `columns` properties and adds many useful linear algebra methods to the object prototype. 
