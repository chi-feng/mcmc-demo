'use strict';

function Visualizer(canvas) {

  this.canvas = canvas;

  this.queue = []; // events for visualization

  this.xmin = -6; // in coordinate-space
  this.xmax =  6; // ymin, ymax set according to canvas aspect ratio

  this.showSamples       = true;
  this.showTargetDensity = true;
  this.animateProposal   = true;
  this.tweening          = false;

  this.arrowSize       = 10;
  this.proposalColor   = '#999';
  this.trajectoryColor = '#333';
  this.acceptColor     = '#4c4';
  this.rejectColor     = '#f00';
  this.nutsColor       = '#09c';

  // offscreen canvases to avoid expensive redraws
  this.densityCanvas = document.createElement('canvas');
  this.samplesCanvas = document.createElement('canvas');
  this.overlayCanvas = document.createElement('canvas');

}

Visualizer.prototype.resize = function() {
  // resize canvas to fit window and scale by devicePixelRatio for HiDPI displays
  this.canvas.width      = document.body.clientWidth  * window.devicePixelRatio;
  this.canvas.height     = document.body.clientHeight * window.devicePixelRatio;
  this.canvas.style.zoom = 1 / window.devicePixelRatio;
  // set ymin, ymax assuming equal aspect ratio
  this.ymin = this.xmin * this.canvas.height / this.canvas.width;
  this.ymax = this.xmax * this.canvas.height / this.canvas.width;
  // scale and origin (location of 0, 0)
  this.scale = this.canvas.width / (this.xmax - this.xmin);
  this.origin = new Float64Array([this.canvas.width / 2, this.canvas.height / 2 + this.canvas.height / (this.ymax - this.ymin) * (this.ymax + this.ymin) / 2]);
  // resize offscreen canvases
  this.densityCanvas.width  = this.canvas.width;
  this.densityCanvas.height = this.canvas.height;
  this.samplesCanvas.width  = this.canvas.width;
  this.samplesCanvas.height = this.canvas.height;
  this.overlayCanvas.width  = this.canvas.width;
  this.overlayCanvas.height = this.canvas.height;

  this.fontSizePx = (12 * window.devicePixelRatio) | 0;
  var context = this.canvas.getContext('2d');
  context.textBaseline = 'top';
  context.font = '' + this.fontSizePx + 'px Arial';
  this.reset();
};

Visualizer.prototype.reset = function() {
  // clear the queue
  this.queue = [ ];
  // stop tweening
  this.tweening = false;
  // clear offscreen and onscreen canvases
  this.densityCanvas.getContext('2d').clearRect(0, 0, this.canvas.width, this.canvas.height);
  this.samplesCanvas.getContext('2d').clearRect(0, 0, this.canvas.width, this.canvas.height);
  this.overlayCanvas.getContext('2d').clearRect(0, 0, this.canvas.width, this.canvas.height);
  this.canvas.getContext('2d').clearRect(0, 0, this.canvas.width, this.canvas.height);
  // redraw density contours
  this.drawDensityContours(this.simulation.mcmc.logDensity);
  this.render();
};

Visualizer.prototype.render = function() {
  var context = this.canvas.getContext('2d');
  // clear onscreen canvas
  context.clearRect(0, 0, this.canvas.width, this.canvas.height);
  // default composition operator
  context.globalCompositeOperation = 'source-over';
  // draw target density canvas
  if (this.showTargetDensity) {
    context.drawImage(this.densityCanvas, 0, 0);
  }
  // draw samples canvas
  if (this.showSamples) {
    context.globalCompositeOperation = 'multiply';
    context.drawImage(this.samplesCanvas, 0, 0);
  }
  // draw overlay canvas
  context.globalCompositeOperation = 'multiply';
  context.drawImage(this.overlayCanvas, 0, 0);

  context.fillText(this.simulation.mcmc.description, 5 * window.devicePixelRatio, 5 * window.devicePixelRatio);

};

// transform world-coordinate to pixel coordinate
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
  var quadratic = (options.quadratic) ? options.quadratic : false;
  context.beginPath()
  context.moveTo(start[0], start[1]);
  if (quadratic){
    for (var i = 1; i < path.length - offset - 1; ++i) {
      var start = this.transform(path[i-1]);
      var mid = this.transform(path[i]);
      var end = this.transform(path[i+1]);
      context.moveTo(start[0], start[1]);
      context.quadraticCurveTo(mid[0], mid[1], end[0], end[1]);
    }
  } else {
    for (var i = 1; i < path.length - offset; ++i) {
      var point = this.transform(path[i]);
      context.lineTo(point[0], point[1]);
    }
  }
  context.stroke();
  if (options.fill) {
    context.fillStyle = options.fill;
    context.fill();
  }
};

Visualizer.prototype.drawArrow = function(canvas, options) {
  var context = canvas.getContext('2d');
  context.lineWidth = (options.lw) ? options.lw * window.devicePixelRatio : 1 * window.devicePixelRatio ;
  context.strokeStyle = (options.color) ? options.color : 'rgb(0,0,0)';
  context.globalAlpha = (options.alpha) ? options.alpha : 1;
  var arrowScale = (options.arrowScale) ? options.arrowScale : 1;
  var from = this.transform(options.from);
  var to = this.transform(options.to);
  context.beginPath()
  context.moveTo(from[0], from[1]);
  context.lineTo(to[0], to[1]);
  var t = Math.atan2(to[1] - from[1], to[0] - from[0]) + Math.PI;
  var size = arrowScale * this.arrowSize * window.devicePixelRatio;
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

// http://stackoverflow.com/questions/17242144/javascript-convert-hsb-hsv-color-to-rgb-accurately
Visualizer.HSVtoRGB = function(h, s, v) {
  var r, g, b, i, f, p, q, t;
  if (arguments.length === 1) {
    s = h.s, v = h.v, h = h.h;
  }
  i = Math.floor(h * 6);
  f = h * 6 - i;
  p = v * (1 - s);
  q = v * (1 - f * s);
  t = v * (1 - (1 - f) * s);
  switch (i % 6) {
    case 0: r = v, g = t, b = p; break;
    case 1: r = q, g = v, b = p; break;
    case 2: r = p, g = v, b = t; break;
    case 3: r = p, g = q, b = v; break;
    case 4: r = t, g = p, b = v; break;
    case 5: r = v, g = p, b = q; break;
  }
  return {
    r: Math.round(r * 255),
    g: Math.round(g * 255),
    b: Math.round(b * 255)
  };
}

Visualizer.prototype.dequeue = function() {

  var event = this.queue.shift();

  var last = this.simulation.mcmc.chain.length > 1 ? this.simulation.mcmc.chain[this.simulation.mcmc.chain.length - 2] : this.simulation.mcmc.chain.last();

  if (event.type == 'proposal') {

    // clear overlay canvas
    var context = this.overlayCanvas.getContext('2d');
    context.clearRect(0, 0, this.canvas.width, this.canvas.height);
    var drawProposalArrow = true;
    var drawProposalCov = true;

    // draw initial momentum vector (for Hamiltonian MC)
    if (event.hasOwnProperty('initialMomentum')) {
      var to = last.add(event.initialMomentum);
      this.drawArrow(this.overlayCanvas, {from: last, to: to, color: this.proposalColor, lw: 1 });
    }
    // draw Hamiltonian MC trajectory or queue animation frames if necessary
    // otherwise, draw arrow from chain.last() to proposal
    if (event.hasOwnProperty('trajectory')) {
      if (this.animateProposal) {
        for (var i = 0; i < event.trajectory.length - 1; ++i)
          this.queue.splice(i, 0, {type: 'trajectory-animation-step', trajectory: event.trajectory, offset: i});
        this.queue.push({type: 'trajectory-animation-end', trajectory: event.trajectory});
      } else {
        this.drawPath(this.overlayCanvas, { path: event.trajectory, color: this.trajectoryColor, lw: 1 });
        for (var i = 0; i < event.trajectory.length - 1; ++i) {
          this.drawCircle(this.overlayCanvas, { fill: this.trajectoryColor, center: event.trajectory[i], radius: 0.015, lw: 0});
        }
        this.drawArrow(this.overlayCanvas, {from: event.trajectory[event.trajectory.length - 2], to: event.trajectory.last(), color: this.trajectoryColor, lw: 1 });
      }
      drawProposalArrow = false;
    }
    // draw NUTS trajectory
    if (event.hasOwnProperty('nuts_trajectory')) {
      drawProposalArrow = false;
      if (this.animateProposal) {
        for (var i = 0; i < event.nuts_trajectory.length; ++i)
          this.queue.splice(i, 0, {type: 'nuts-animation-step', trajectory: event.nuts_trajectory, offset: i});
        this.queue.push({type: 'nuts-animation-end', trajectory: event.nuts_trajectory});
      } else {
        for (var i = 0; i < event.nuts_trajectory.length; ++i) {
          if (event.nuts_trajectory[i].type == 'accept') {
            this.drawPath(this.overlayCanvas, { path: [event.nuts_trajectory[i].from, event.nuts_trajectory[i].to], color: this.nutsColor, lw: 1});
            this.drawCircle(this.overlayCanvas, { fill: this.nutsColor, center: event.nuts_trajectory[i].to, radius: 0.015, lw: 0});
          }
        }
      }
    }
    // draw MALA gradient/proposal offset
    if (event.hasOwnProperty('gradient')) {
      this.drawArrow(this.overlayCanvas, { from: last, to: last.add(event.gradient), color: this.nutsColor, lw: 1 });
      this.drawArrow(this.overlayCanvas, { from: last.add(event.gradient), to: event.proposal, color: this.proposalColor, lw: 1.5});
      if (event.hasOwnProperty('proposalCov')) {
        drawProposalCov = false;
        this.drawProposalContour(this.overlayCanvas, last.add(event.gradient), event.proposalCov);
      }
    }
    // draw proposal covariance
    if (event.hasOwnProperty('proposalCov') && drawProposalCov) {
      var center = event.hasOwnProperty('proposalMean') ? event.proposalMean : last;
      this.drawProposalContour(this.overlayCanvas, center, event.proposalCov);
      if (event.hasOwnProperty('proposalMean')) {
        drawProposalArrow = false;
        this.drawPath(this.overlayCanvas, { path: [last, center], color: this.proposalColor, lw: 1 });
        this.drawArrow(this.overlayCanvas, { from: center, to: event.proposal, color: this.proposalColor, lw: 1 });
      }
    }

    // draw proposal covariance
    if (event.hasOwnProperty('revProposalCov') && drawProposalCov) {
      var center = event.hasOwnProperty('revProposalMean') ? event.revProposalMean : last;
      this.drawProposalContour(this.overlayCanvas, center, event.revProposalCov);
      if (event.hasOwnProperty('revProposalMean')) {
        drawProposalArrow = false;
        this.drawPath(this.overlayCanvas, { path: [event.proposal, event.revProposalMean], color: '#00f', lw: 1 });
        this.drawArrow(this.overlayCanvas, { from: center, to: last, color: '#00f', lw: 1 });
      }
    }

    // draw proposal arrow
    if (drawProposalArrow) {
      this.drawArrow(this.overlayCanvas, {from: last, to: event.proposal, color: this.proposalColor, lw: 1 });
    }
  }

  if (event.type == 'trajectory-animation-step') {
    this.tweening = true; // start skiping delay for calling requestAnimationFrame
    var context = this.overlayCanvas.getContext('2d');
    var path = [event.trajectory[event.offset], event.trajectory[event.offset + 1]];
    this.drawPath(this.overlayCanvas, { path: path, color: this.trajectoryColor, lw: 1});
    // this.drawArrow(this.overlayCanvas, {from: event.trajectory[event.offset], to: event.trajectory[event.offset + 1], color: this.trajectoryColor, lw: 0.5, arrowScale: 0.8, alpha: 0.8 });
    this.drawCircle(this.overlayCanvas, { fill: this.trajectoryColor, center: event.trajectory[event.offset + 1], radius: 0.015, lw: 0});
  }

  if (event.type == 'trajectory-animation-end') {
    this.tweening = false; // stop skipping delay for calling requestAnimationFrame
  }

  if (event.type == 'nuts-animation-step') {
    this.tweening = true; // start skiping delay for calling requestAnimationFrame
    var context = this.overlayCanvas.getContext('2d');
    var type = event.trajectory[event.offset].type;
    if (type == 'accept' || type == 'reject') {
      var path = [event.trajectory[event.offset].from, event.trajectory[event.offset].to];
      var color = (event.trajectory[event.offset].type == 'accept') ? this.nutsColor : '#f00';
      this.drawPath(this.overlayCanvas, { path: path, color: color, lw: (type == 'accept') ? 1 : 0.5});
      this.drawCircle(this.overlayCanvas, { color: color, center: event.trajectory[event.offset].to, radius: 0.015, lw: 0.5});
    } else if (type == 'left' || type == 'right') {
      this.nutsColor = (type == 'right') ? '#09c' : '#66f';
      var path = [event.trajectory[event.offset+1].from, event.trajectory[event.offset+1].to];
      var color = (event.trajectory[event.offset+1].type == 'accept') ? this.nutsColor : '#f00';
      var from = (type == 'left') ? path[1] : path[0];
      var to   = (type == 'left') ? path[0] : path[1];
      // this.drawArrow(this.overlayCanvas, {from: from, to: to, color: color, lw: 1, arrowScale: 0.7});
      this.drawCircle(this.overlayCanvas, { fill: color, center: event.trajectory[event.offset+1].from, radius: 0.025, lw: 0});
    }
  }

  if (event.type == 'nuts-animation-end') {
    this.tweening = false;
  }

  if (event.type == 'accept') {
    this.drawArrow(this.overlayCanvas, { from: last, to: event.proposal, color: this.acceptColor, lw: 2 });
    this.drawSample(this.samplesCanvas, event.proposal);
  }

  if (event.type == 'reject') {
    this.drawArrow(this.overlayCanvas, { from: last, to: event.proposal, color: this.rejectColor, lw: 2 });
    this.drawSample(this.samplesCanvas, last);
  }

};

Visualizer.prototype.drawProposalContour = function(canvas, last, cov) {
  var context = canvas.getContext('2d');
  context.lineWidth = 1 * window.devicePixelRatio ;
  context.globalAlpha = 1;

  // get principle components using eigenvalue decomposition
  var eigs = cov.jacobiRotation({maxIter:100, tolerance: 1e-5});
  for (var i = 0; i < 2; ++i)
    eigs.V.setCol(i, eigs.V.col(i).scale(Math.sqrt(eigs.D[i * 2 + i])));
  var eigs = [eigs.V.col(0), eigs.V.col(1)];

  // get major and minor axes and rotation
  var a = eigs[0].norm();
  var b = eigs[1].norm();
  var angle = Math.atan2(-eigs[0][1], eigs[0][0]);
  var center = this.transform(last);
  context.beginPath();
  context.strokeStyle = '#ddd';
  context.ellipse(center[0], center[1], 2 * a * this.scale, 2 * b * this.scale, angle, 0, 2 * Math.PI, false);
  context.stroke();

  // shaded contour
  // var gradient = context.createRadialGradient(center[0], center[1], 0, center[0], center[1], Math.max(2 * a * this.scale, 2 * b * this.scale));
  // gradient.addColorStop(0.2, "#ccc");
  // gradient.addColorStop(1, "#fff");
  // context.fillStyle = gradient;
  // context.fill();

  context.beginPath();
  context.strokeStyle = '#999';
  context.ellipse(center[0], center[1], a * this.scale, b * this.scale, angle, 0, 2 * Math.PI, false);
  context.stroke();

  // draw principle axes
  // this.drawArrow(canvas, { from: last, to: last.add(eigs[0]), color: 'rgba(192,192,192,' +  this.alpha + ')', lw: 1 });
  // this.drawArrow(canvas, { from: last, to: last.add(eigs[1]), color: 'rgba(192,192,192,' +  this.alpha + ')', lw: 1 });
};

Visualizer.prototype.drawDensityContours = function(logDensity) {

  var nx = 297, ny = 303, nz = 9;
  var x = linspace(this.xmin - 1, this.xmax + 1, nx);
  var y = linspace(this.ymin - 1, this.ymax + 1, ny);
  var data = [];
  var point = zeros(2,1);

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

  var z = linspace(min + 0.01 * (max - min), max - 0.02 * (max - min), nz);
  var c = new Conrec;
  c.contour(data, 0, nx - 1, 0, ny - 1, x, y, nz, z);
  var contours = c.contourList();

  for (var i = 0; i < contours.length; ++i) {
    var contour = [];
    for (var j = 0; j < contours[i].length; ++j)
      contour.push([contours[i][j].x, contours[i][j].y]);
    // this.drawPath(this.densityCanvas, {path:contour, color: 'transparent', fill:'rgba(64,128,192,0.1)'});
    this.drawPath(this.densityCanvas, {path:contour, color:'#69b', alpha: 0.4 * (i+1) / contours.length, lw: 2 });
  }

};
