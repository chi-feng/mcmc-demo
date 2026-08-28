/**
 * Generate the gallery thumbnails in thumbnails/ from app.html.
 *
 * Usage:
 *   npm install playwright && npx playwright install chromium
 *   node scripts/generate-thumbnails.js
 *
 * The script loads app.html for each algorithm with a fixed seed, runs the
 * sampler with animation disabled, hides the UI, and screenshots the canvas.
 */
const path = require("path");
const { chromium } = require("playwright");

// Steps are chosen per algorithm so each thumbnail shows a filled-in chain.
const ALGORITHMS = [
  ["RandomWalkMH", 400],
  ["AdaptiveMH", 400],
  ["HamiltonianMC", 150],
  ["NaiveNUTS", 80],
  ["EfficientNUTS", 80],
  ["DualAveragingHMC", 150],
  ["DualAveragingNUTS", 80],
  ["MALA", 400],
  ["H2MC", 250],
  ["GibbsSampling", 250],
  ["DE-MCMC-Z", 400],
  ["SVGD", 150],
  ["MicrocanonicalHamiltonianMC", 150],
];

(async () => {
  const root = path.resolve(__dirname, "..");
  const browser = await chromium.launch();
  const page = await browser.newPage({
    viewport: { width: 240, height: 240 },
    deviceScaleFactor: 1,
  });

  for (const [algorithm, steps] of ALGORITHMS) {
    const url = `file://${root}/app.html?algorithm=${algorithm}&target=banana&seed=thumbnail`;
    await page.goto(url);
    // Simulation.js declares sim with top-level let, so it is not a window property.
    await page.waitForFunction(() => typeof sim !== "undefined" && sim.mcmc.initialized);
    await page.evaluate((n) => {
      sim.autoplay = false;
      viz.animateProposal = false;
      gui.domElement.style.display = "none";
      document.getElementById("info").style.display = "none";
      for (let i = 0; i < n; i++) sim.step();
      viz.render();
    }, steps);
    const file = path.join(root, "thumbnails", `${algorithm}.png`);
    await page.locator("#plotCanvas").screenshot({ path: file });
    console.log(`wrote ${file}`);
  }

  await browser.close();
})();
