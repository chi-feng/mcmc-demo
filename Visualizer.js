'use strict';

function Visualizer(canvas) {

  this.canvas = canvas;

  this.queue = [];

  this.xmin = -6;
  this.xmax = 6;

  this.showSamples        = true;
  this.showDensityContour = true;
  this.animateProposal    = true;
  this.tweening           = false;
  this.alpha              = 0.5;

  this.densityCanvas = document.createElement('canvas');
  this.samplesCanvas = document.createElement('canvas');
  this.overlayCanvas = document.createElement('canvas');

}

Visualizer.prototype.resize = function() {

  this.canvas.width      = window.innerWidth  * window.devicePixelRatio;
  this.canvas.height     = window.innerHeight * window.devicePixelRatio;
  this.canvas.style.zoom = 1 / window.devicePixelRatio;

  this.ymin = this.xmin * this.canvas.height / this.canvas.width;
  this.ymax = this.xmax * this.canvas.height / this.canvas.width;

  this.scale = this.canvas.width / (this.xmax - this.xmin);
  this.origin = new Float64Array([this.canvas.width / 2, this.canvas.height / 2 + this.canvas.height / (this.ymax - this.ymin) * (this.ymax + this.ymin) / 2]);

  this.densityCanvas.width  = this.canvas.width;
  this.densityCanvas.height = this.canvas.height;
  this.samplesCanvas.width  = this.canvas.width;
  this.samplesCanvas.height = this.canvas.height;
  this.overlayCanvas.width  = this.canvas.width;
  this.overlayCanvas.height = this.canvas.height;

  this.reset();

};

Visualizer.prototype.reset = function() {
  this.queue = [];
  this.densityCanvas.getContext('2d').clearRect(0, 0, this.canvas.width, this.canvas.height);
  this.samplesCanvas.getContext('2d').clearRect(0, 0, this.canvas.width, this.canvas.height);
  this.overlayCanvas.getContext('2d').clearRect(0, 0, this.canvas.width, this.canvas.height);
  this.canvas.getContext('2d').clearRect(0, 0, this.canvas.width, this.canvas.height);
  this.drawDensityContours(this.simulation.mcmc.logDensity);
  this.render();
};

Visualizer.prototype.render = function() {
  var context = this.canvas.getContext('2d');
  context.clearRect(0, 0, this.canvas.width, this.canvas.height);
  if (this.showDensityContour) {
    context.drawImage(this.densityCanvas, 0, 0);
  }
  if (this.showSamples) {
    context.globalCompositeOperation = 'multiply';
    context.drawImage(this.samplesCanvas, 0, 0);
  }
  context.globalCompositeOperation = 'source-over';
  context.drawImage(this.overlayCanvas, 0, 0);
  context.globalCompositeOperation = 'source-over';
};

Visualizer.prototype.transform = function(x) {
  var transformed = new Float64Array(2);
  transformed[0] = x[0] * this.scale + this.origin[0];
  transformed[1] = this.origin[1] - this.scale * x[1];
  return transformed;
};

Visualizer.prototype.drawCircle = function(canvas, options) {
  var context = canvas.getContext('2d');
  context.lineWidth = (options.lw) ? options.lw * window.devicePixelRatio  : 1 * window.devicePixelRatio ;
  context.strokeStyle = (options.color) ? options.color : 'rgb(0,0,0)';
  context.globalAlpha = (options.alpha) ? options.alpha : 1;
  var center = this.transform(options.center);
  context.beginPath();
  if (options.end && options.fill)
    context.moveTo(center[0], center[1]);
  context.arc(center[0], center[1], options.radius * this.scale, (options.start || 0), (options.end || 2 * Math.PI), false);
  if (options.end && options.fill)
    context.closePath();
  if (options.fill) {
    context.fillStyle = options.fill;
    context.fill();
  }
  if (options.lw > 0) {
    context.stroke();
  }
};

Visualizer.prototype.drawPath = function(canvas, options) {
  var context = canvas.getContext('2d');
  context.lineWidth = (options.lw) ? options.lw * window.devicePixelRatio : 1 * window.devicePixelRatio ;
  context.strokeStyle = (options.color) ? options.color : 'rgb(0,0,0)';
  context.globalAlpha = (options.alpha) ? options.alpha : 1;
  var offset = (options.offset) ? options.offset : 0;
  var path = options.path;
  var start = this.transform(path[0]);
  context.beginPath()
  context.moveTo(start[0], start[1]);
  for (var i = 1; i < path.length - offset; ++i) {
    var point = this.transform(path[i]);
    context.lineTo(point[0], point[1]);
  }
  if (options.fill) {
    context.closePath();
    context.fillStyle = options.fill;
    context.fill();
  }
  context.stroke();
};

Visualizer.prototype.drawArrow = function(canvas, options) {
  var context = canvas.getContext('2d');
  context.lineWidth = (options.lw) ? options.lw * window.devicePixelRatio : 1 * window.devicePixelRatio ;
  context.strokeStyle = (options.color) ? options.color : 'rgb(0,0,0)';
  context.globalAlpha = (options.alpha) ? options.alpha : 1;
  var from = this.transform(options.from);
  var to = this.transform(options.to);
  context.beginPath()
  context.moveTo(from[0], from[1]);
  context.lineTo(to[0], to[1]);
  var t = Math.atan2(to[1] - from[1], to[0] - from[0]) + Math.PI;
  var size = 10 * window.devicePixelRatio;
  context.moveTo(to[0] + size * Math.cos(t + Math.PI / 8), to[1] + size * Math.sin(t + Math.PI / 8));
  context.lineTo(to[0], to[1]);
  context.lineTo(to[0] + size * Math.cos(t - Math.PI / 8), to[1] + size * Math.sin(t - Math.PI / 8));
  context.stroke();
};

Visualizer.prototype.drawSample = function(canvas, center) {
  var context = canvas.getContext('2d');
  context.globalCompositeOperation = 'multiply';
  this.drawCircle(canvas, { fill: 'rgb(216,216,216)', center: center, radius: 0.015, lw: 0});
  context.globalCompositeOperation = 'source-over';
}

Visualizer.prototype.dequeue = function() {

  var event = this.queue.shift();

  if (event.type == 'proposal') {

    var context = this.overlayCanvas.getContext('2d');
    context.clearRect(0, 0, this.canvas.width, this.canvas.height);

    /*
    this.drawPath(this.overlayCanvas, { path: this.simulation.mcmc.proposalTrajectory, color: 'rgb(64,64,64)', lw: 1.5 });
    this.drawArrow(this.overlayCanvas, { from: this.simulation.mcmc.proposalTrajectory[this.simulation.mcmc.proposalTrajectory.length-2], to: this.simulation.mcmc.proposalTrajectory[this.simulation.mcmc.proposalTrajectory.length-1], color: 'rgb(64,64,64)', lw: 1.5 });
    this.drawCircle(this.overlayCanvas, { fill: 'rgb(64,64,64)', center: event.proposal, radius: 0.02, lw: 0});
    */
    this.drawArrow(this.overlayCanvas, { from: event.last, to: event.proposal, color: 'rgb(192,192,192)', lw: 2 });

  }

  if (event.type == 'hmc-animation-start') {
    var context = this.overlayCanvas.getContext('2d');
    context.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.drawArrow(this.overlayCanvas, { from: event.last, to: event.last.add(event.initialMomentum), color: 'rgb(128,128,128)', lw: 2, alpha:0.5 });
  }

  if (event.type == 'hmc-animation') {
    this.tweening = true;
    var context = this.overlayCanvas.getContext('2d');
    context.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.drawArrow(this.overlayCanvas, { from: event.last, to: event.last.add(event.initialMomentum), color: 'rgb(128,128,128)', lw: 2, alpha:0.5 });
    this.drawCircle(this.overlayCanvas, { fill: 'rgb(64,64,64)', center: event.proposal, radius: 0.04, lw: 0});
    this.drawPath(this.overlayCanvas, { path: this.simulation.mcmc.proposalTrajectory, offset: this.simulation.mcmc.proposalTrajectory.length - 1 - event.index, color: 'rgb(64,64,64)', lw: 1.5 });
  }

  if (event.type == 'hmc-animation-end') {
    this.tweening = false;
    this.drawPath(this.overlayCanvas, { path: this.simulation.mcmc.proposalTrajectory, color: 'rgb(64,64,64)', lw: 1.5 });
    this.drawArrow(this.overlayCanvas, { from: this.simulation.mcmc.proposalTrajectory[this.simulation.mcmc.proposalTrajectory.length-2], to: this.simulation.mcmc.proposalTrajectory[this.simulation.mcmc.proposalTrajectory.length-1], color: 'rgb(64,64,64)', lw: 1.5 });
  }

  if (event.type == 'accept') {
    this.drawArrow(this.overlayCanvas, { from: event.last, to: event.proposal, color: 'rgb(64,192,64)', lw: 2, alpha:0.5 });
    this.drawSample(this.samplesCanvas, event.proposal);
  }

  if (event.type == 'reject') {
    this.drawArrow(this.overlayCanvas, { from: event.last, to: event.proposal, color: 'rgb(255,64,64)', lw: 2 });
    this.drawSample(this.samplesCanvas, event.last);
  }

};

Visualizer.prototype.drawDensityContours = function(logDensity) {

  var nx = 201, ny = 201, nz = 10;
  var x = Float64Array.linspace(this.xmin - 1, this.xmax + 1, nx);
  var y = Float64Array.linspace(this.ymin - 1, this.ymax + 1, ny);
  var data = [];
  var point = Float64Array.zeros(2,1);

  var min = 1e10, max = 0;
  for (var i = 0; i < nx; ++i) {
    data.push([]);
    point[0] = x[i];
    for (var j = 0; j < ny; ++j) {
      point[1] = y[j];
      var val = Math.exp(logDensity(point));
      data[i].push(val);
      if (val > max) max = val;
      if (val < min) min = val;
    }
  }

  var z = Float64Array.linspace(min + 0.025 * (max - min), max - 0.05 * (max - min), nz);
  var c = new Conrec;
  c.contour(data, 0, nx - 1, 0, ny - 1, x, y, nz, z);
  var contours = c.contourList();

  for (var i = 0; i < contours.length; ++i) {
    var contour = [];
    for (var j = 0; j < contours[i].length; ++j)
      contour.push([contours[i][j].x, contours[i][j].y]);
    this.drawPath(this.densityCanvas, {path:contour, color:'#69b', alpha: 0.8 * this.alpha * (i+1) / contours.length + 0.1});
  }

};
