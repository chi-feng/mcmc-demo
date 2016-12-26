/*
Copyright (c) 2016 Chi Feng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

"use strict";

/**
 * Unconstrained gradient-based optimization
 * @param  {function} f       objective function
 * @param  {function} g       gradient of objective function
 * @param  {object} user_opts
 */
Float64Array.opt = function(f, g, user_opts) {

  // defaults
  var options = {
    method: 'bfgs',
    tolerance: 1e-6,
    step_size: 1.0,
    max_iter: 100,
    warn_max_iter: false,
    line_search_iter: 20,
    line_search_tolerance: 1e-4
  };

  // assign dim
  if (!user_opts.hasOwnProperty('x0'))
    throw 'x0 not set';
  else
    options.dim = user_opts.x0.length;

  // assign user options
  for (var key in user_opts) {
    options[key] = user_opts[key];
  }

  // run optimization
  if (options.method == 'bfgs') {
    return Float64Array.bfgs(f, g, options);
  } else {
    throw 'unrecognized method';
  }

};

Float64Array.bfgs = function(f, grad, options) {
  var n = options.dim;
  var geval = 0;
  // bisection line search in the direction p
  var line_search = function(x, p) {
    var a, a_lo = 0, a_hi = options.step_size;
    for (var k = 0; k < options.line_search_iter; k++) {
      a = (a_hi + a_lo) / 2;
      var h = grad(x.add(p.scale(a))).dot(p); geval++;
      if (h > options.line_search_tolerance) a_hi = a;
      else if (h < -options.line_search_tolerance) a_lo = a;
      else break;
    }
    return a;
  };
  // BFGS
  var B = Float64Array.eye(options.dim, options.dim);
  var x = [options.x0.copy()];
  var p = [ ], a = [ ], g = [grad(x[0])]; geval++;
  var k = 0;
  for (k = 0; k < options.max_iter; k++) {
    if (g[k].dot(g[k]) < options.tolerance) break;
    p.push(B.lu_solve(g[k].negate()));
    a.push(line_search(x[k], p[k]));
    x.push(x[k].add(p[k].scale(a[k])));
    g.push(grad(x[k+1])); geval++;
    var s = p[k].scale(a[k]);
    var y = g[k+1].subtract(g[k]);
    var first = Float64Array.outer(y, y).scale(1.0 / y.dot(s));
    var second = B.multiply(Float64Array.outer(s, s.transpose().multiply(B))).scale(1.0 / s.dot(B.multiply(s)));
    B.increment(first.subtract(second));
  }
  if (k == options.max_iter && options.warn_max_iter)
    console.log('max_iter exceeded, gradient is', g[k]);
  return {x: x[x.length-1], trajectory: x, p: p, a:a, geval: geval};
};
