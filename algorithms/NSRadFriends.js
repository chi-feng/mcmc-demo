'use strict';

/*
Copyright: Johannes Buchner (C) 2013-2017

Nested Sampling with RadFriends   https://arxiv.org/abs/1407.5459

Code recycled from https://github.com/JohannesBuchner/ultranest-js

License: AGPL-3.0

See TODO's at the bottom for open issues

*/


function point(L, coords, phys_coords) {
	this.L = L
	this.coords = coords
	this.phys_coords = phys_coords
}

function random_uniform() {
	return Math.random();
}
function random_normal(mu, stdev) {
    var proposalDist  = new MultivariateNormal(mu, stdev*stdev);
    return proposalDist.getSample();
}
function random_int(imin, imax) {
    return Math.floor(Math.random() * (imax - imin + 1)) + imin;
}
function logaddexp(a, b) {
	if (b > a)
		return Math.log(1 + Math.exp(a - b)) + b
	else
		return Math.log(1 + Math.exp(b - a)) + a
}

function compute_distance(acoords, bcoords) {
	var distsq = 0
	var n = acoords.length
	for(var j = 0; j < n; j++)
		distsq += (acoords[j] - bcoords[j]) * (acoords[j] - bcoords[j])
	return distsq
}


function compute_distance_lt(acoords, bcoords, maxsqdistance) {
	var distsq = 0
	var n = acoords.length
	for(var j = 0; j < n; j++) {
		distsq += (acoords[j] - bcoords[j]) * (acoords[j] - bcoords[j])
		if (distsq > maxsqdistance)
			return false
	}
	return true
}



function nearest_rdistance_guess(ndim, live_points) {
	// Jack-knife implementation
	var maxsqdistance = 0.0
	var n = live_points.length
	for(var i = 0; i < n; i++) {
		// leave ith point out
		var mindistance = 1e300
		var nonmember = live_points[i]
		for (var k = 0; k < n; k++) {
			if (k == i)
				continue;
			var dist = compute_distance(live_points[k].coords, nonmember.coords)
			if (k == 0 || dist < mindistance)
				mindistance = dist
		}
		maxsqdistance = Math.max(mindistance, maxsqdistance)
	}
	// console.log("nearest_rdistance_guess: " + maxsqdistance + " from " + n)
	return maxsqdistance
}

function getRandomInt(min, max) {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min)) + min; //The maximum is exclusive and the minimum is inclusive
}

function nearest_rdistance_guess(ndim, live_points) {
	// boot-strapping implementation
	var nbootstrap_rounds = 20;
	var maxsqdistance = 0.0
	var n = live_points.length
	for(var j = 0; j < nbootstrap_rounds; j++) {
		var selected = [];
		var nonselected = [];
		for(var i = 0; i < n; i++) {
			var k = getRandomInt(0, n)
			if(selected.indexOf(k) == -1)
				selected.push(k)
		}
		for(var i = 0; i < n; i++) {
			if(selected.indexOf(i) == -1)
				nonselected.push(i)
		}
		for(var i = 0; i < nonselected.length; i++) {
			// compute distance to any selected
			var a = nonselected[i];
			var b = selected[0];
			var minsqdistance = compute_distance(live_points[a].coords, live_points[b].coords);
			for(var k = 1; k < selected.length; k++) {
				b = selected[k];
				minsqdistance = Math.min(minsqdistance, compute_distance(live_points[a].coords, live_points[b].coords));
			}
			maxsqdistance = Math.max(minsqdistance, maxsqdistance)
		}
	}
	// console.log("nearest_rdistance_guess: " + maxsqdistance + " from " + n)
	return maxsqdistance
}

function random_normal_vector(ndim) {
        var direction = new MultivariateNormal(zeros(ndim, 1), eye(ndim, ndim)).getSample();
	return direction.scale(1. / direction.norm())
}


function radfriends_drawer(ndim, transform, likelihood) {
	this.likelihood = likelihood,
	this.transform = transform
	this.ndim = ndim
	var niter = 0
	this.niter = niter
	function _init_region() {
		var region_low = []
		var region_high = []
	
		for (var i = 0; i < ndim; i++) {
			region_low[i] = 0.0
			region_high[i] = 1.0
		}
		this.region_low = region_low
		this.region_high = region_high
	}
	this.init_region = _init_region
	var _maxsqdistance = NaN
	this.maxsqdistance = _maxsqdistance
	var phase = 1
	this.phase = phase
	this.init_region()
	this.rejected = []
	
	function _is_inside(current, members) {
		for (var i = 0; i < ndim; i++) {
			if (current.coords[i] < this.region_low[i])
				return false
			if (current.coords[i] > this.region_high[i])
				return false
		}
		if (!(this.maxsqdistance > 0)) {
			console.log("friends not used because maxsqdistance is " + this.maxsqdistance)
			return true;
		}
		for (var i = 0, n = members.length; i < n; i++) {
			if (compute_distance_lt(members[i].coords, current.coords, this.maxsqdistance))
				return true
		}
		return false
	}
	this.is_inside = _is_inside
	function _count_inside(current, members) {
		// assumes it is inside
		/*for (var i = 0; i < ndim; i++) {
			if (current.coords[i] < this.region_low[i])
				return 0
			if (current.coords[i] > this.region_high[i])
				return 0
		}*/
		if (!(this.maxsqdistance > 0)) {
			console.log("friends not used because maxsqdistance is " + this.maxsqdistance)
			return 1;
		}
		var nnearby = 0
		//console.log("neighbors of " + current.coords)
		for (var i = 0; i < members.length; i++) {
			if (compute_distance_lt(members[i].coords, current.coords, this.maxsqdistance)) {
				//console.log("distance " + dist + " (max:" + this.maxsqdistance  + ") to [" + i + "]: " + members[i].coords)
				nnearby += 1;
			}
		}
		return nnearby
	}
	this.count_inside = _count_inside
	function _generate_direct(current, members) {
		var ntotal = 0
		var n = members.length
		while(1) {
			for(var j = 0; j < ndim; j++) {
				current.coords[j] = random_uniform() * (this.region_high[j] - this.region_low[j]) + this.region_low[j]
				current.phys_coords[j] = current.coords[j]
			}
			ntotal += 1
			if (n == 0) {
				console.log("generate_direct(): No friends available for checking!")
				return ntotal
			}
			if (this.is_inside(current, members))
				return ntotal
			if (ntotal > 1000)
				return ntotal
		}
	}
	this.generate_direct = _generate_direct
	
	function _generate_from_friends(current, members) {
		var ntotal = 0
		var n = members.length
		while(1) {
			ntotal += 1
			var member = members[random_int(0, n - 1)]
			var direction = random_normal_vector(ndim)
			var radius = Math.sqrt(this.maxsqdistance) * Math.pow(random_uniform(), 1.0/ndim)
			for(var j = 0; j < ndim; j++) {
				current.coords[j] = member.coords[j] + direction[j] * radius
				current.phys_coords[j] = current.coords[j]
			}
			ntotal += 1
			if (this.is_inside(current, members)) {
				var coin = random_uniform()
				var nnearby = this.count_inside(current, members)
				if (coin < 1.0 / nnearby)
					return ntotal
			}
		}
	}
	this.generate_from_friends = _generate_from_friends
	
	
	function _next(current, live_points) {
		this.niter += 1
		var Lmin = current.L
		var n = live_points.length
		// console.log("drawer: next() iteration " + this.niter + " - " + Lmin)
		//if (!(this.maxsqdistance > 0) || (this.niter % 20 == 1)) {
		if (true) {
			// console.log("drawer: next(): recomputing maxsqdistance")
			var newmaxsqdistance = nearest_rdistance_guess(ndim, live_points)
			if (!(this.maxsqdistance > 0) || newmaxsqdistance < this.maxsqdistance)
				this.maxsqdistance = newmaxsqdistance
			for (var j = 0; j < ndim; j++) {
				var low = 1
				var high = 0
				for (var i = 0; i < n; i++) {
					var p = live_points[i]
					low = Math.min(low, p.coords[j])
					high = Math.max(high, p.coords[j])
				}
				this.region_low[j] = Math.max(0, low - Math.sqrt(this.maxsqdistance))
				this.region_high[j] = Math.min(1, high + Math.sqrt(this.maxsqdistance))
			}
			console.log("drawer: next(): new maxsqdistance: " + this.maxsqdistance)
		}
		var ntoaccept = 0
		if (this.phase == 0) {
			//console.log("drawer: next(): generating from rectangle")
			while (1) {
				var ntotal = this.generate_direct(current, live_points)
				ntoaccept += 1
				current.phys_coords = transform(current.coords)
				current.L = likelihood(current.phys_coords)
				if (current.L >= Lmin) {
					if (this.iter % 100 == 1)
						console.log("drawer: next()[rectangle]: accepted: " + current.L + " after " + ntoaccept + " evals (" + ntotal + ")" )
					return current
				} else {
					this.rejected.push(current.phys_coords.copy())
				}
				if (ntotal >= 20) {
					this.phase = 1
					break
				}
			}
		}
		//console.log("drawer: next(): generating from friends")
		while (1) {
			ntoaccept += 1
			var ntotal = this.generate_from_friends(current, live_points)
			current.phys_coords = transform(current.coords)
			current.L = likelihood(current.phys_coords)
			if (current.L >= Lmin) {
				if (this.iter % 100 == 1)
					console.log("drawer: next()[friends]: accepted: " + current.L + " after " + ntoaccept + " evals (" + ntotal + ")")
				return current
			} else {
				this.rejected.push(current.phys_coords.copy())
			}
		}
	}
	this.next = _next
}


function generate_fullspace(ndim) {
	var current = new point(1e300, [], []) 
	for(var j = 0; j < ndim; j++) {
		current.coords[j] = random_uniform()
		current.phys_coords[j] = current.coords[j]
	}
	return current
}

function sort_L(live_points) {
	// console.log("live points before sort: " + live_points[0].L + " to " + live_points[live_points.length - 1].L)
	live_points.sort(function(a, b) {
		if (a.L < b.L)
			return -1
		if (a.L > b.L)
			return 1
		return 0
	})
	// console.log("live points after  sort: " + live_points[0].L + " to " + live_points[live_points.length - 1].L)
}

function posterior_samples(weighted_samples, nsamples) {
	var probs = []
	var logmax = weighted_samples[0][0] + weighted_samples[0][1].L
	console.log("wsamples: " + weighted_samples[0] + " -> " + weighted_samples[0][1] + " -> " + weighted_samples[0][1].L)
	for (var i = 0; i < weighted_samples.length; i++) {
		logmax = Math.max(logmax, weighted_samples[i][0] + weighted_samples[i][1].L)
	}
	console.log("logmax:" + logmax)
	var sum = 0
	for (var i = 0; i < weighted_samples.length; i++) {
		probs[i] = Math.exp(weighted_samples[i][0] + weighted_samples[i][1].L - logmax)
		sum += probs[i]
	}
	console.log("sum:" + sum)
	var samples = []
	for (var j = 0; j < nsamples; j++) {
		var coin = random_uniform() * sum
		var below = 0
		var i = 0
		while(i < weighted_samples.length) {
			below += probs[i]
			if (coin <= below)
				break
			else
				i += 1
		}
		// console.log("choice:" + i + " of " + weighted_samples.length + " where " + coin + " reached " + below)
		samples[j] = weighted_samples[i][1].phys_coords
	}
	return samples
}


function nested_sampler(ndim, drawer, nlive_points, transform, likelihood) {
	this.nlive_points = nlive_points
	this.transform = transform
	this.likelihood = likelihood
	this.ndim = ndim
	this.Lmax = NaN
	this.remainderZ = NaN
	this.ndraws = 0
	this.drawer = drawer
	this.live_points = []
	this.latest_point = NaN
	function _generate_live_points() {
		console.log("sampler: generating live points ")
		for(var i = 0; i < nlive_points; i++) {
			var Lmin = -1e300
			var current = generate_fullspace(ndim)
			console.log("transforming " + current.coords)
			current.phys_coords = transform(current.coords)
			console.log("became " + current.phys_coords)
			current.L = likelihood(current.phys_coords)
			console.log("with likelihood " + current.L)
			if (i == 0)
				this.Lmax = current.L
			else
				this.Lmax = Math.max(this.Lmax, current.L)
			this.live_points[i] = current
			this.latest_point = current
		}
		sort_L(this.live_points)
		console.log("sampler: generating live points done: " + this.live_points.length)
	}
	this.generate_live_points = _generate_live_points
	this.generate_live_points()
	
	function _next() {
		var i = 0
		var lowest = this.live_points[i]
		//console.log("sampler: next(): need better than " + lowest.L)
		var replacement = new point(lowest.L, lowest.coords.slice(), lowest.phys_coords.slice())
		var ndraws = drawer.next(replacement, this.live_points)
		//console.log("sampler: next(): got " + replacement.L + ", returning " + lowest.L)
		this.live_points[i] = replacement
		this.latest_point = replacement
		sort_L(this.live_points)
		this.ndraws += ndraws
		return lowest
	}
	this.next = _next
	function _integrate_remainder(logwidth, logVolremaining, logZ, points) {
		//console.log("sampler: integrate_remainder()")
		var n = nlive_points
		var logV = logwidth
		var L0 = this.live_points[this.live_points.length - 1].L
		var Lmax = 0
		var Lmin = 0
		var Lmid = 0
		for (var i = 0; i < n; i++) {
			var Ldiff = Math.exp(this.live_points[i].L - L0)
			if (i > 0)
				Lmax += Ldiff
			if (i == n - 1)
				Lmax += Ldiff
			if (i < n - 1)
				Lmin += Ldiff
			if (i == 0)
				Lmin += Ldiff
			Lmid += Ldiff
		}
		var logZmid = logaddexp(logZ, logV + Math.log(Lmid) + L0)
		var logZup  = logaddexp(logZ, logV + Math.log(Lmax) + L0)
		var logZlo  = logaddexp(logZ, logV + Math.log(Lmin) + L0)
		var logZerr = Math.max(logZup - logZmid, logZmid - logZlo)
		this.remainderZ = logV + Math.log(Lmid) + L0
		this.remainderZerr = logZerr
		var points = []
		for (var i = 0; i < n; i++) {
			points[i] = [logwidth, this.live_points[i]]
		}
		return points
	}
	this.integrate_remainder = _integrate_remainder
}

function integrator(ndim, transform, likelihood, data_calc, nlive_points, tolerance, maxiter) {
	var drawer = new radfriends_drawer(ndim, transform, likelihood)
	var sampler = new nested_sampler(ndim, drawer, nlive_points, transform, likelihood)
	this.current = sampler.next()
	this.sampler = sampler
	this.drawer = drawer
	
	this.logVolremaining = 0
	this.logwidth = Math.log(1 - Math.exp(-1.0 / nlive_points))
	
	this.iter = 0
	var weights = []
	this.weights = weights
	this.results = []
	this.wi = this.logwidth + this.current.L
	this.logZ = this.wi
	this.H = this.current.L - this.logZ
	this.logZerr = NaN
	console.log("integrator[initial]: ln Z = " + this.logZ + " " + this.H + " " + this.wi + " " + this.current.L)
	
	function _progress() {
		this.logwidth = Math.log(1 - Math.exp(-1.0 / nlive_points)) + this.logVolremaining
		this.logVolremaining -= 1.0 / nlive_points
		
		weights[this.iter] = [this.logwidth, this.current]
		
		this.iter += 1
		this.logZerr = Math.sqrt(this.H / nlive_points)
		
		sampler.integrate_remainder(this.logwidth, this.logVolremaining, this.logZ)
		
		if (false) {
			var total_error = this.logZerr + sampler.remainderZerr
			if (total_error < tolerance) {
				console.log("integrator: tolerance reached " + total_error + " of " + tolerance)
				return 0
			}
			if (sampler.remainderZerr < this.logZerr / 10.) {
				console.log("integrator: tolerance can not be reached " + sampler.remainderZerr + " vs " + this.logZerr / 10.)
				return 0
			}
			if (maxiter > 0 && this.iter > maxiter) {
				console.log("integrator: max # of iter reached")
				return 0
			}
		}
		this.current = sampler.next()
		this.wi = this.logwidth + this.current.L
		var logZnew = logaddexp(this.logZ, this.wi)
		this.H = Math.exp(this.wi - logZnew) * this.current.L + Math.exp(this.logZ - logZnew) * (this.H + this.logZ) - logZnew
		this.logZ = logZnew
		if (this.iter % 50 == 0)
			console.log("integrator[" + this.iter + "]: current ln Z = " + this.logZ + " +- " + this.logZerr + " +- " + sampler.remainderZerr)
		return 1
	}
	this.progress = _progress
	function _getResults() {
		var remainder_weights = sampler.integrate_remainder(this.logwidth, this.logVolremaining, this.logZ)
		var logZtotal = this.logZ
		var Htotal = this.H
		for(var i = 0; i < remainder_weights.length; i++) {
			var Li = remainder_weights[i][1].L
			var wi = this.logwidth + Li
			var logZnew = logaddexp(logZtotal, wi)
			Htotal = Math.exp(wi - logZnew) * Li + Math.exp(logZtotal - logZnew) * (Htotal + logZtotal) - logZnew
			logZtotal = logZnew
		}

		var logZerrfinal = Math.sqrt(Htotal / nlive_points) + sampler.remainderZerr
		var logZfinal = logaddexp(logZtotal, sampler.remainderZ)

		return [logZfinal, logZerrfinal, weights.concat(remainder_weights)]
	}
	this.getResults = _getResults

}


function transform(cube) {
   var params = zeros(cube.length, 1)
   for(var i = 0; i < params.length; i++) {
     params[i] = cube[i] * 10 - 5;
   }
   return params
}

MCMC.registerAlgorithm('RadFriends-NS', {

  description: 'Nested Sampling with RadFriends',

  about: function() {
    window.open('https://arxiv.org/abs/1407.5459');
  },

  init: function(self) {
    self.live_points = [];
    self.nlive_points = 40; // number of particles
    self.iter = 0;
    self.reset(self);
  },

  reset: function(self) {
    // initialize chain with samples from standard normal
    self.iter = 0;
    self.chain = [];
    self.integrator = new integrator(self.dim, transform, self.logDensity, null, self.nlive_points, 0.1, 0)
  },

  attachUI: function(self, folder) {
    folder.add(self, 'nlive_points', 10, 400).step(1).name('numLivePoints');
    folder.open();
  },

  step: function(self, visualizer) {
    // point about to be removed:
    var lowest = self.integrator.sampler.live_points[0].phys_coords.slice();
    var previous = self.integrator.current.phys_coords.slice()
    
    var r = self.integrator.progress()

    if (r == 0) {
       // TODO:
       // we are done/converged
       // maybe the algorithm should sleep/stop or restart from scratch after a little while?
    }
    
    // visualise:
    //    newest drawn live point: self.integrator.current replaced lowest
    //visualizer.reset()
    console.log("rejected: " + self.integrator.drawer.rejected)

    // visualise:
    //    draw the RadFriends region as overlapping circles
    //    it is simple:
    //    use points from self.integrator.sampler.live_points[i].phys_coords
    //    and as radius sqrt(self.integrator.drawer.maxsqdistance)
    //    this should give a "bubble-bath-like" look of the region where new
    //    points are drawn from
    var x = [];
    var rad = Math.sqrt(self.integrator.drawer.maxsqdistance) * 10;
    for(var i = 0; i < self.integrator.sampler.live_points.length; i++) {
      x.push(self.integrator.sampler.live_points[i].phys_coords.slice());
    }
    console.log("live points:" + x.length + " radius: " +  rad)

    //visualizer.queue.push({type: 'proposal', previous: previous, ns_rejected: self.integrator.drawer.rejected});
    visualizer.queue.push({type: 'radfriends-region', x: x, r: rad});
    visualizer.queue.push({type: 'ns-dead-point', proposal: self.integrator.sampler.latest_point.phys_coords, deadpoint: previous, rejected: self.integrator.drawer.rejected});
    self.integrator.drawer.rejected = []
    
    var results = self.integrator.getResults()
    var weighted_samples = results[2];
    var samples = posterior_samples(weighted_samples, weighted_samples.length);
    
    // TODO: visualise: 
    //    samples are the current equal-weighted approximation of the posterior
    //    update histogram. These are being resampled
    //    if you can handle weighted histograms, use weighted_samples
    self.chain = samples;
  },
});


