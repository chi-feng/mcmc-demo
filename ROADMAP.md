# Roadmap

This roadmap records a 2026-08-29 review of the interactive Markov chain Monte Carlo (MCMC) gallery on the `refactoring` branch through local commit 5d2c4cc. A second reviewer (gpt-5.6-sol) cross-checked the repository claims, and the browser-visible blocker claims were verified in a live browser. Each unchecked item assigns work to the maintainer or a contributor.

The maintainer removed engineering effort as a selection constraint for 2026. A direction qualifies only if it corrects a named misconception or makes a named concept tangible and specifies a user-facing improvement. Effort estimates help order dependencies; they do not decide whether a direction belongs in the plan. The sections progress from branch blockers, correctness, and tests through the baseline experience, the platform rebuild, and the directions that use it. The "Explored and rejected" section records the proposals that failed one of the two gates.

## Blockers on the refactoring branch

Every blocker below was fixed on 2026-08-29 in the four-PR sequence that landed the branch (regression fixes 195b005 and 896ded1, gallery integration 1c03d6c and ee669a9, and the review-feedback commit 1a35eac). The list stays for the record; each defect was verified in a live browser before and after the fix.

- [x] Fix the H2MC crash. Commit 5d2c4cc replaced master's plain reassignments with calls to `offsetBuff.copyFrom(...)` and `postInvCovEigenvalues.copyFrom(...)` (`algorithms/H2MC.js:68,88`), but no `copyFrom` method exists in `lib/linalg.core.js` or anywhere else. H2MC throws on its first nontrivial proposal, and the chain never grows past length 1.
- [x] Fix the donut target crash. Commit 05264f8 made the donut's `logDensity` and `gradLogDensity` read `this.radius` and `this.sigma2` (`main/MCMC.js:89`), but `Simulation.setTarget` copies the methods off the target object (`main/Simulation.js:55`) and `computeContours` calls them as bare functions, so `this` is undefined under strict mode. Selecting the donut target throws `Cannot read properties of undefined (reading 'radius')`. Bind the methods to the target in `setTarget`, or reference the target object explicitly inside the methods.
- [x] Add the missing `thumbnails/` directory. `index.html` references eleven thumbnail image files in this directory, but the directory does not exist, and all eleven gallery images render broken. Generate the files with a headless-browser screenshot script so contributors can reproduce them.
- [x] Add `<script src="algorithms/MCHMC.js">` to `app.html`. The gallery links to `?algorithm=MicrocanonicalHamiltonianMC`, but `app.html` does not load the file, so the URL silently selects the first registered algorithm, `HamiltonianMC`. Master has the sibling defect: its gallery link uses the misspelled name `MiMicrocanonicalHamiltonianMC`.
- [x] Add gallery entries for `EfficientNUTS`, `DualAveragingHMC`, and `DualAveragingNUTS`. The gallery listed eleven of the then-fourteen algorithm files that call `MCMC.registerAlgorithm`; after the nested-sampling removal, thirteen algorithms remain and all appear in the gallery.
- [x] Split the silent behavior change into its own commit. The refactor renamed the inner `q0` in `DualAveragingNUTS.buildTree` to `q0_local`, which corrects master's bug of overwriting the `q0` parameter; master computed the dual-averaging acceptance statistic against the wrong reference point. The correction is right, and it should land as a documented fix with a test rather than inside a style-only commit.
- [x] Reconcile the local branch with `origin/refactoring`. On 2026-08-29, the remote ref pointed to the pre-rebase commit 7374073, while the local branch pointed to 5d2c4cc, four commits ahead and one behind. Fetch first, then push the rebased branch with `--force-with-lease` or delete the remote branch before pushing.

## Correctness fixes

All defects below predate the refactor commits, which reproduced them faithfully. The cited papers confirm the intended formulas. Each fix should include a test that fails before the fix.

- [x] Fix the Microcanonical Hamiltonian Monte Carlo (MCHMC) momentum update in `algorithms/MCHMC.js:41`. Equation 16 in the paper by Jakob Robnik, G. Bruno De Luca, Eva Silverstein, and Uroš Seljak (arXiv:2212.08549) adds `2ζu` to a term in the gradient direction `e`. The current code puts the full scalar on `e`, which removes the momentum component tangent to the gradient and makes every updated velocity parallel to `e`, so the sampler follows the negative log-density gradient instead of the intended microcanonical rotation. Line 50 also discards the value returned by `p0.scale(...)`; `scale` returns a copy, so the code never normalizes the initial momentum. Fixing it exposed a third defect: the code negated the log-density gradient where the reference implementation negates the energy gradient, so the velocity pointed downhill and the chain escaped to infinity. All three were fixed together (commit 39d5a44, amended by 1a35eac).
- [x] Copy the momentum at the start of `buildTree` in `algorithms/EfficientNUTS.js` and `algorithms/DualAveragingNUTS.js`. These two No-U-Turn Sampler (NUTS) implementations copy `q` but mutate the caller's `p`, and their base cases return one momentum object as both `p_m` and `p_p`. A second recursive call then mutates the momentum retained for the first subtree, so the internal U-turn check `(q⁺−q⁻)·p⁻` uses the wrong endpoint momentum. The outer loop keeps separate momentum objects for its two global endpoints, so the top-level check does not expose the aliasing. `algorithms/NaiveNUTS.js` copies both `q` and `p` and is unaffected. Test endpoint independence at tree depth 2.
- [x] Add the proposal-density ratio to the Hessian-Hamiltonian Monte Carlo (H2MC) acceptance test in `algorithms/H2MC.js:124`. The proposal Gaussian depends on the current point through the gradient and Hessian, so it is generally asymmetric, but the code uses only the target-density ratio and therefore does not generally preserve the displayed target. Compute `q(x|y)/q(y|x)` from the reverse Gaussian, or label the demo as an uncorrected variant.
- [x] The nested-sampling evidence accounting defects (shell-width off-by-one and a double-counted live-point remainder) were resolved by removing `algorithms/NSRadFriends.js` with the AGPL cleanup; a future clean-room implementation must not reproduce them.
- [x] Fix the Stein variational gradient descent (SVGD) particle-count increase in `algorithms/SVGD.js:67`. The loop bound `self.n - self.chain.length` shrinks as each push grows `chain.length`, so raising the particle count adds only about half the requested particles. Capture the deficit before the loop.
- [x] Align the dual-averaging constants with Hoffman and Gelman's Algorithms 5 and 6 in `algorithms/DualAveragingHMC.js` and `algorithms/DualAveragingNUTS.js`. The paper initializes `H̄₀ = 0` and `γ = 0.05`; both files use 1.0 and 0.2. `findReasonableEpsilon` must also integrate one leapfrog step from the same starting position and momentum after each change to epsilon; the files instead continue from the previous leapfrog result.
- [x] Resolve the mixed licensing. `algorithms/NSRadFriends.js` carries a GNU Affero General Public License 3.0 (AGPL-3.0) notice and says its code was recycled from Johannes Buchner's `ultranest-js`, while the repository-level `LICENSE` declares MIT. Resolved on 2026-08-29 by removing the file and its references (commit 3197703); the repository is MIT-only again. A future nested-sampling demo needs a clean-room implementation or MIT permission from the author.
- [x] Delete or fix the unused broken code. `MCMC.computeAutocorrelation` has no callers, and `MCMC.computeMean` is called only by that unused method; `computeAutocorrelation` allocates a typed array of length `lag` and then writes index `lag`. `Float64Array.prototype.trace` in `lib/linalg.core.js:334` references the undefined variables `A` and `j`. `randomNormal` in `algorithms/NSRadFriends.js:36` passes scalar values where `MultivariateNormal` requires a mean vector and covariance matrix.
- [x] Fix the y-marginal loop bound in `main/Visualizer.js:206`. The loop uses `xgrid.length`, which is 480, while it indexes `ygrid` and `ymarg`, which each have length 256.
- [x] Derive the Gibbs conditional grid in `algorithms/GibbsSampling.js:34` from the current bounds for the coordinate being sampled: `xmin`/`xmax` for x and `ymin`/`ymax` for y. The current grid hardcodes −6 to 6 for both coordinates and would truncate a target with different bounds.
- [ ] Prune the added JSDoc. Commit 5d2c4cc added blocks that narrate names and signatures of internal helpers. Keep the file-level algorithm descriptions, paper links, attribution, and license text; delete the narration.

## Tests and tooling

Most transition code uses numeric arrays and the seeded `Math.random`, but each step also writes events to a visualizer, so tests provide a stub visualizer. Before the platform rebuild, the site continues to load plain scripts; when the project gains a `package.json`, all test and development packages go in `devDependencies`.

- [x] A first statistical layer exists: `scripts/check-samplers.js` runs every sampler on the standard Gaussian with per-check seeds and moment tolerances, and it fails on the pre-fix code (EfficientNUTS variance 1.33, H2MC variance 0.39, MCHMC NaN, SVGD resize 250 of 300). Still open: a real test runner with unit tests for `lib/linalg.core.js` and `main/MultivariateNormal.js` against closed-form values and property tests for leapfrog reversibility.
- [ ] Add a test suite with Vitest or `node:test`. Include unit tests for `lib/linalg.core.js` and `main/MultivariateNormal.js` against closed-form values, property tests for leapfrog reversibility and a detailed-balance sanity check, and seeded statistical tests: run every sampler on the standard Gaussian and check its sample, weighted-sample, or particle mean and covariance within stated tolerances. Include pre-fix cases and tolerances that expose the MCHMC and H2MC defects.
- [ ] Add a continuous integration (CI) workflow that runs the tests and a Playwright browser check. Enumerate every algorithm file, every target, and every gallery link; load each `app.html?algorithm=...&target=...` combination; assert that the requested algorithm and target were selected, that the console has no errors, and that the chain or particle set grows. The selected-algorithm assertion catches the missing MCHMC script tag, and the per-target pass catches the donut crash; a growing default chain alone would miss both.
- [ ] Enable type checking without a build step. The refactor added JSDoc comments; add a `tsconfig.json` with `checkJs` and run `tsc --noEmit` in CI.
- [ ] Replace dat.GUI with lil-gui. The lil-gui project is active, provides larger controls for coarse pointers, and is compatible with most of the dat.GUI API; three.js replaced dat.GUI with lil-gui in r135. This repository cannot treat it as a literal drop-in replacement because `Simulation.js` accesses `gui.__folders` and `gui.__ul`, and `app.html` styles `.dg` internals; port those calls and styles to the documented lil-gui API.
- [ ] Vendor `seedrandom` in `lib/` instead of loading version 2.4.3 from cdnjs. The CDN script is the only external resource required during page load; vendoring lets the demo load offline.
- [ ] Align `findReasonableEpsilon` fully with Algorithm 4 (deferred from the 2026-08-29 review): the heuristic still starts at 0.1 and both dual-averaging files clamp its result, which can deny a sharp target the step size the search found. Removing the clamps changes demo tuning behavior, so it needs visual QA on the funnel before it lands.
- [ ] Give targets per-axis extents (deferred from the same review): Gibbs now reads `xmin`/`xmax` for both axes, which is correct while every target is square; per-axis `ymin`/`ymax` is schema work for the platform rebuild.
- [ ] Pin the Playwright version for `scripts/generate-thumbnails.js` when the project gains its `package.json` and lockfile.

## Baseline user experience

The app uses a full-screen canvas and a fixed-width control panel, with no keyboard handlers, no reduced-motion handling, and no quantitative diagnostics; the viewport also disables zoom with `maximum-scale=1`. These items fix that baseline before the larger directions build on it.

- [ ] Encode acceptance and rejection with more than color. Accepted and rejected moves use the same arrow shape and width, with colors `#4c4` and `#f00`, a distinction that users with red-green color-vision deficiencies can miss. Keep the colors and add a shape distinction, such as solid versus dashed lines or filled versus open arrowheads.
- [ ] Make the app usable on phones: a collapsible control panel, larger touch targets, and a tap gesture that sets the chain's current position.
- [ ] Add a live diagnostics strip with the running acceptance rate, an effective sample size estimate, and samples per second. The implementation must record accept and reject outcomes and elapsed time, since chain positions alone do not carry them. Define equivalent diagnostics for SVGD, whose array represents particles.
- [ ] Add an optional trace-plot panel that shows each coordinate against iteration, revealing mixing problems that a two-dimensional scatter hides, such as slow movement through the lower part of the funnel.
- [ ] After the pure engine extraction, add a comparison mode with two panels that use the same target and seed but different algorithms or parameters, stepped by one parent controller.
- [ ] Add a "copy link to this state" button that serializes the current global and algorithm-specific settings into the existing URL-parameter scheme.
- [ ] Add keyboard controls (Space to step, `r` to reset, `p` to toggle autoplay) and honor `prefers-reduced-motion` by defaulting `animateProposal` to false for users who request reduced motion.
- [ ] Replace `window.alert("Done!")` at nested-sampling convergence in `algorithms/NSRadFriends.js` with an on-canvas banner.

## Baseline pedagogy

The page provides one-line algorithm descriptions, and its "About this algorithm" control opens a paper or reference page. These items connect the animation to the mathematics for the existing thirteen algorithms.

- [ ] Extend the registry with an `explain` field for each algorithm: four to six sentences describing the proposal, the acceptance rule, and the details to watch on screen, with the acceptance rule rendered by a vendored KaTeX build inside a collapsible panel.
- [ ] Show each algorithm's numerical state during animation: the log acceptance ratio for each proposal, the Hamiltonian error along an HMC trajectory, the NUTS tree depth, and a divergence flag for a large energy error. The dual-averaging variants already draw `epsilon` on the canvas; extend that mechanism.
- [ ] Add failure-mode presets: "step size too large" for HMC divergences, "step size too small" for random-walk behavior, and "isolated modes" for random-walk Metropolis-Hastings on the multimodal target, each with one sentence that tells the student what to watch. The presets teach the failure alongside the successful configuration.
- [ ] Add a guided tour driven by a JSON list, where each stop specifies an algorithm, a target, parameters, and a caption, with a Next button so instructors can link a specific sequence.
- [ ] Document iframe embedding for course pages, and add a `?ui=min` parameter that hides the control panel.
- [ ] Add a `CITATION.cff` file and a short contributing guide documenting the one-file algorithm plug-in pattern.

## The platform rebuild

The current architecture couples sampling to rendering. All fourteen algorithm `step` functions receive a visualizer, and each one pushes drawing events into `visualizer.queue`. A caller must therefore supply renderer-like queue state for every sampler instance, and a worker must replace or bypass the visualizer. Parallel chains and synchronized panels are possible in the current code, but each feature would need its own queue adapter. The rebuild below creates one engine boundary for directions 1 through 6, 8, and 9; the course layer can start from the baseline embedding work.

- [ ] Extract a pure, dimension-generic TypeScript engine. Algorithms become generators that consume an injected random number generator and emit typed events such as proposals, acceptances, rejections, trajectories, and divergences. Engine code has no Document Object Model (DOM) or `window` access, so one interface can run on the main thread, in a Web Worker when the page can start one, or in many parallel instances.
- [ ] Port with a replay guarantee. Each ported algorithm must reproduce the corrected JavaScript baseline's chain bit for bit from the same seed. Inject the random number generator through `MultivariateNormal` and every helper that now calls `Math.random`, and land each of the fourteen ports with its equivalence test.
- [ ] Split rendering from the engine. Canvas 2D remains the default renderer, and the ensemble view adds a graphics processing unit (GPU) renderer and compute path. WebGPU, the browser API for GPU graphics and computation, is available by default on supported hardware in [Chrome](https://developer.chrome.com/blog/webgpu-release), [Firefox on Windows and Apple silicon macOS](https://developer.mozilla.org/en-US/docs/Mozilla/Firefox/Experimental_features#webgpu_api), and [Safari 26](https://webkit.org/blog/17333/webkit-features-in-safari-26-0/#webgpu), but coverage still depends on the operating system and hardware. Detect `navigator.gpu` at runtime. On unsupported systems, retain the one-chain and four-chain Canvas 2D views and offer a reduced CPU ensemble only if it meets the interaction budget; do not attempt a WebGPU polyfill.
- [ ] Replace the finite-difference Hessian in `main/Simulation.js` with a small forward-mode automatic-differentiation expression type. Formula-defined targets then obtain gradients and Hessians by evaluating the expression graph, subject to floating-point error, while painted targets continue to use numerical derivatives.
- [ ] Add nightly statistical checks for every compatible sampler-target pair. Use seeded runs, Kolmogorov-Smirnov tests on the marginals, and a maximum mean discrepancy test for the joint distribution against reference samples from high-resolution numerical integration (quadrature). Validate the method and thresholds on targets with analytic reference distributions, and report these tests as regression detectors rather than proofs of correctness. Calibrate cases to expose the MCHMC and H2MC distribution errors; the per-target browser checks in the earlier CI item cover the donut and `copyFrom` crashes.
- [ ] Retain static deployment and local loading. GitHub Pages serves plain static files, and a built, committed `app.html` must continue to work from `file://` with a main-thread fallback. The build emits classic scripts for deployment even if the TypeScript source uses modules. A documented compatibility adapter must also preserve the one-file plain-JavaScript algorithm contribution path by mapping a plug-in to the engine's typed event interface.

## Breakthrough directions

Each direction names the concept it teaches and the change a user will experience.

### 1. Many chains, and diagnostics as instruments

[Vehtari et al. (2021)](https://arxiv.org/abs/1903.08008) recommend rank-normalized split R-hat, rank plots from multiple chains, and bulk and tail effective sample size. R-hat compares variation within and between chains to diagnose convergence. Run four color-coded chains at once. On the multimodal target, students can watch chains settle in different modes while R-hat remains elevated even when each trace appears stationary. The animation connects the diagnostic to the behavior that produces it.

Extend the baseline diagnostics strip with per-chain values, combined R-hat and effective sample size, and a rank-plot drawer. This direction requires the engine rebuild and the diagnostics panels, and it is the first direction to build.

### 2. The typical set, and dimension as a slider

Two-dimensional density plots encourage students to equate the mode with the region that contains most of the probability mass. In high dimensions, the volume of a shell causes probability mass to concentrate away from the mode in a typical set. Add a mode that samples a standard Gaussian in dimension `d`, with a slider from 2 to 1000. Its radius follows a chi distribution and concentrates in a thin shell near √d. Show that radius histogram against the theoretical distribution while two projected coordinates retain the current scatter plot. [Betancourt's 2017 conceptual introduction](https://arxiv.org/abs/1701.02434) provides the syllabus for the typical set and Hamiltonian Monte Carlo (HMC) geometry.

At each dimension, show fixed-parameter and retuned runs for random-walk Metropolis and HMC. A fixed random-walk proposal scale produces falling acceptance as dimension grows; the retuned comparison shows how each method must scale its parameters and how efficiently it traverses the shell. The dimension slider updates the scatter plot, radius histogram, acceptance statistics, and traversal diagnostics. The dimension-generic engine provides the sampler work, and the new implementation work is in these panels.

### 3. The reparameterization morph

Students often meet the non-centered parameterization of a hierarchical funnel only as an algebraic transformation. Add a slider that continuously interpolates an invertible coordinate transform from the centered parameterization to the non-centered parameterization. Include the log absolute Jacobian determinant in the transformed density. The density, contours, current state, and stored samples move together as the funnel becomes an independent Gaussian after the endpoint variables are rescaled. Students can compare one sampler in both coordinate systems and map the samples back to the original variables.

The existing funnel controls gain one slider. This direction is small after targets use automatic-differentiation expressions because the coordinate transform composes with the target expression.

### 4. The ensemble view

A single chain is one realization of a stochastic process. Run about ten thousand independent chains on the GPU and render their states as a changing density. Students can then observe burn-in, step-size effects, and chains trapped in one mode across the population.

Add a view toggle with three settings: one chain, four chains, and ensemble. This direction requires WebGPU compute kernels for random-walk Metropolis, the Metropolis-adjusted Langevin algorithm (MALA), the unadjusted Langevin algorithm (ULA), and fixed-length HMC. It is the only roadmap direction that requires GPU compute, and it follows the feature-detection and fallback policy in the platform section.

### 5. The bridge to diffusion models

[Song and Ermon's 2019 score-based model](https://proceedings.neurips.cc/paper_files/paper/2019/hash/3001ef257407d5a371a96dcd947c7d93-Abstract.html) generates samples with annealed Langevin dynamics. A noise-conditioned network learns the score, which is the gradient of the log density, at a sequence of decreasing noise levels. Train a small two-layer score network in the browser on samples from a two-dimensional Gaussian mixture. Then show annealed Langevin sampling with the learned score beside the same updates with the known score of the noise-perturbed mixture. In a linked control, compare ULA with MALA on the original analytic density at the same step size. The learned-score comparison isolates score estimation error, and the ULA-MALA comparison isolates the Metropolis-Hastings correction.

Add a "learned score" target family and an annealed-Langevin algorithm entry that extends the ULA toggle in "New algorithms." Keep the network implementation framework-free. A prototype must confirm an interactive training time on the supported laptop and phone baseline before the interface commits to in-browser training. The expected implementation effort is moderate.

### 6. From gallery to laboratory

Students can test a sampler on a target from their own exercises. Add two custom-target inputs. The formula input parses a log-density expression into the automatic-differentiation type, which supplies derivatives for gradient- and Hessian-based methods. The painting input creates a smoothed grid interpolant and obtains its derivatives numerically, with a visible warning that resolution and smoothing affect derivative-based samplers. `Simulation.computeContours` already evaluates a supplied `logDensity` on a two-dimensional grid and computes numerical marginals. Reuse those grid and marginal calculations after canvas creation moves into the renderer.

Add a custom-target tab for the formula and painting inputs. Painting requires a small isolated implementation; the formula input depends on the automatic-differentiation work.

### 7. The course layer

The current gallery shows algorithm motion. A course layer will connect those observations to conclusions, and the maintainer chose its shape on 2026-08-29: one flagship scroll-driven essay in the register of ciechanow.ski, at the site root, built from prose, rendered mathematics, and roughly sixty pre-staged figures that each expose a single interaction. The chapters follow one running inference example and cover what Metropolis does, why gradients help, the typical set with direction 2 embedded, diagnostics with direction 1 embedded, sampler failures, the sampler grammar with direction 10 embedded, and the path from Langevin dynamics to diffusion models with direction 5 embedded. The 2017 Hacker News submission of the bare sandbox earned 2 points, which is the evidence that the tool without prose does not carry; the essay is the artifact.

Make the course the site's main entry point. Each figure remains interactive, and each chapter links to the full sandbox. The baseline `?ui=min` mode and URL serialization provide the embedding interface. Most of the work is writing and editing the course.

### 8. Tuning challenges

Students build tuning intuition through repeated decisions and immediate measurements. Add challenge presets with measurable goals: reach a target effective sample size within a step budget on the funnel, diagnose the pathology in an unidentified trace, or find the step size where divergences begin. The diagnostics engine scores each attempt immediately.

Add a challenge drawer with a stated goal, a step budget, and a result. This direction is small after the diagnostics work. Each challenge must teach a workflow skill used outside the demo.

### 9. Classroom mode

In a lecture, each student tunes a sampler on a phone while the instructor's screen aggregates the chains' diagnostics. The class can watch the effective-sample-size ranking respond to different step-size choices.

Add room creation, a student join flow, and an instructor dashboard. Solo users see no classroom controls unless they enter a room. This direction requires the phone work and a small real-time backend with one Cloudflare Durable Object per room. It is the only direction that needs a server, and every non-classroom feature must continue to work without that server.

### 10. The sampler-grammar workbench

Textbooks and demos, this one included, present samplers as a zoo of named methods. Most of the zoo is combinatorial: a sampler is a proposal mechanism (random-walk jump, gradient drift, Hamiltonian flight, ensemble stretch, slice bracket, continuous-time event, learned transport), a correction that repairs the stationary distribution (none, Metropolis-Hastings, importance weight, exact by construction), and an adaptation scheme (none, scale, covariance, step size and mass matrix, learned map). Add a workbench where the reader composes the three slots directly and watches the result run: switching the correction off on MALA produces ULA and visible bias; snapping covariance adaptation onto the random walk produces AdaptiveMH; giving HMC an automatic trajectory length produces NUTS; wrapping any sampler in a learned transport map produces its preconditioned version. The reader leaves knowing the design space rather than thirteen names. I know of no interactive teaching artifact that presents samplers this way.

The workbench needs the engine's typed event interface plus a composition layer that the canonical roster entries themselves are built from, so the grammar is the implementation rather than a facade. The essay's chapters introduce each slot; the workbench is where the reader proves to themselves that the slots compose.

## The gp-demo question

The maintainer also maintains [`gp-demo`](https://github.com/chi-feng/gp-demo), an interactive Gaussian-process regression demo built with vanilla JavaScript and hosted on GitHub Pages. A repository merge now would couple two working sites without changing either user's experience. A shared site can connect them later through two topics. [Murray, Adams, and MacKay (2010)](https://proceedings.mlr.press/v9/murray10a.html) developed elliptical slice sampling for models with multivariate Gaussian priors and demonstrated it on Gaussian-process models. A course chapter could also sample a Gaussian process's hyperparameter posterior with HMC or NUTS.

Keep the repositories separate until the platform exists. If the course layer ships, publish one site that links or mounts both demos and let `gp-demo` adopt the engine and renderer conventions. Reconsider a repository merge only if shared maintenance then requires it.

## The algorithm collection

The collection grows in two moves: first consolidate the redundant variants, then add algorithms by family, where each family teaches one idea and each entry earns its place by showing something the others do not.

### One canonical NUTS

- [ ] Replace the three No-U-Turn Sampler entries and `DualAveragingHMC` with two roster entries. `HamiltonianMC` gains an "adapt step size" toggle and absorbs `DualAveragingHMC`. A single `NUTS` entry implements what Stan ships (Betancourt 2017): multinomial state selection along the trajectory, the generalized U-turn criterion, windowed warmup that adapts the step size and a diagonal mass matrix, and explicit divergence flags. The historical variants stay reachable as a staged "variant" control inside that one entry (Algorithm 2's naive tree, Algorithm 3's slice selection, then multinomial), because the essay teaches the progression and the roster should stop charging the reader three slots for one idea.

### Classical coverage

| Algorithm | Reference | What the visualization teaches |
|---|---|---|
| Slice sampling | Neal 2003 | It shows level sets, stepping out, and shrinkage. |
| Elliptical slice sampling | Murray, Adams, and MacKay 2010 | It shows a tuning-free update and the shrinking ellipse of candidate points. |
| Affine-invariant ensemble | Goodman and Weare 2010; Foreman-Mackey et al. 2013 for `emcee` | It shows a walker cloud whose stretch moves adapt to the target's shape without gradients. |
| Parallel tempering | Swendsen and Wang 1986; Geyer 1991 | It shows a temperature ladder and swaps as a standard way to move between modes that trap local samplers. |
| Unadjusted Langevin algorithm (ULA), as a toggle on MALA | Roberts and Tweedie 1996 | It removes the Metropolis-Hastings correction so students can see the finite-step stationary distribution shift. |
| Barker proposal | Livingstone and Zanella 2022 | It shows gradient-based robustness by sweeping the step size and comparing stability with MALA. |
| Preconditioned Crank-Nicolson (pCN) | Cotter, Roberts, Stuart, and White 2013 | It shows a dimension-robust proposal whose acceptance does not collapse as the dimension slider rises; it is the prerequisite for the function-space entries. |
| Zig-Zag process and Bouncy Particle Sampler | Bierkens, Fearnhead, and Roberts 2019; Bouchard-Côté, Vollmer, and Doucet 2018 | They show continuous-time, nonreversible, piecewise-deterministic paths in two dimensions. |
| Sequential Monte Carlo (SMC) sampler | Del Moral, Doucet, and Jasra 2006 | It shows a weighted particle population annealing from a prior to a posterior, and it would restore the ground nested sampling covered before the AGPL removal. |

### The transport family

Measure transport is the MIT Uncertainty Quantification group's thread (El Moselhy and Marzouk 2012 frames Bayesian inference as finding a map from a reference distribution to the posterior), and it is the collection's largest gap.

| Algorithm | Reference | What the visualization teaches |
|---|---|---|
| Transport-map MCMC | Parno and Marzouk, arXiv:1412.5492 | An adaptive triangular (Knothe-Rosenblatt) map, fit to the chain's own history, straightens the target; the proposal runs in the reference space. Show the map as a deforming grid, with the reference-space chain beside the target-space chain, and watch the banana straighten as the map adapts. |
| Neural transport preconditioning | Hoffman et al. 2019 (NeuTra) | The same idea with a normalizing flow in place of the triangular map; it bridges to the learned-dynamics family. |
| Stein variational Newton | Detommaso, Cui, Marzouk, Scheichl, and Spantini 2018 | Second-order preconditioning of the existing SVGD particles; running both side by side shows what curvature information buys on the ill-conditioned target. |

### The surrogate family

- [ ] Add local approximation MCMC (Conrad, Marzouk, Pillai, and Smith 2016). The sampler treats the log density as expensive, builds a local polynomial surrogate, evaluates the true density only where the error indicator demands refinement, and the visualization shows the surrogate patches and refinement points accumulating along the chain's path. This is the only entry that teaches the cost structure of real scientific inference, where one density evaluation is a simulation. Stage it with an artificial per-evaluation delay so the reader feels the budget.

### The learned-dynamics family

- [ ] Add annealed Langevin sampling with a learned score (Song and Ermon 2019), per breakthrough direction 5.
- [ ] Add flow matching (Lipman et al. 2023), with the training honestly framed: flow matching learns a velocity field from samples, so the demo trains it on the output of a long NUTS run on the same target and then transports fresh reference points along near-straight paths. Show the learned velocity field and contrast its straight probability paths with the diffusion entry's curved ones.
- [ ] Treat density-only flow training as the stretch tier: annealed flow transport (Arbel, Matthews, and Doucet 2021) and flow annealed importance sampling bootstrap (Midgley et al. 2023) sample from an unnormalized density without preexisting samples, at the cost of machinery the essay may not want to carry.

### The function-space tier

- [ ] After the dimension slider exists, add a dimension-independent likelihood-informed (DILI) MCMC demonstration (Cui, Law, and Marzouk 2016) on top of pCN: the likelihood-informed subspace shows the handful of directions the data actually constrains while pCN handles the rest, which is the geometry behind modern high-dimensional inverse problems.

The target changes stand:

- [ ] Add an ill-conditioned Gaussian target with correlation 0.99 and a heavy-tailed Student-t target. The first shows the effects of preconditioning and gradients; the second shows HMC's behavior in heavy tails.
- [ ] Fold the draw-your-own-density mode into direction 6 above.

## Explored and rejected

- Reject a large language model (LLM) tutor that narrates the simulation. The simulator teaches through controlled observation, while a narrator asks students to trust another generated explanation. An LLM also adds an application programming interface (API) dependency to a tool that works offline. Revisit this decision only if the course layer does not give students enough guidance.
- Reject three-dimensional surface rendering of the two-dimensional densities. It uses screen space and interaction controls while making the existing contour-reading task harder.
- Reject a general probabilistic-programming frontend. Stan, PyMC, and NumPyro already provide that interface, and the custom-target laboratory covers this roadmap's pedagogic need.
- Reject a Rust or WebAssembly engine rewrite for speed. The review found no runtime profile showing that the mathematics is a bottleneck at the current scale. WebGPU addresses the planned compute-heavy ensemble view.
- Do not use native ECMAScript modules in the deployed artifact. Browsers apply module security restrictions to `file://` pages, and opening `index.html` directly must continue to work. The TypeScript source can use modules because the build emits classic scripts.
- Reject virtual-reality and augmented-reality modes because the proposals do not identify a concept they would teach.

## Suggested order

1. Fix the branch blockers, then push and merge the `refactoring` branch.
2. Land each correctness fix with its failing-first test, and resolve the licensing split.
3. Add CI with the per-algorithm, per-target browser checks; port the controls to lil-gui; vendor `seedrandom`.
4. Land the baseline user-experience and pedagogy items that do not depend on parallel engine instances for the existing thirteen algorithms.
5. Build the platform: extract the engine with replay-equivalence tests, add automatic-differentiation targets and the statistical checks, and then land the two-panel comparison mode.
6. Build directions 1 through 3: multi-chain diagnostics, the dimension slider, and the reparameterization morph. These directions establish the diagnostic and geometric concepts that later directions reuse.
7. Prototype directions 4 through 8 in order, and require each prototype to demonstrate its pedagogic story and user-facing improvement before full implementation. Consolidate NUTS first, then add algorithms from the collection family by family when a visualization supports one of these directions or an essay chapter.
8. Build classroom mode after the phone and multi-chain work, and revisit shared hosting with `gp-demo` after the platform and course layer exist.
