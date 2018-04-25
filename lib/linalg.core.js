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
 * Creates a "matrix" from an existing array-like object with optional dimensions
 * @param  {ArrayLike} array existing array, can be nested or flat (row-major)
 * @param  {int}       rows  number of rows
 * @param  {int}       cols  number of columns
 * @return {Float64Array}
 */
Float64Array.matrix = function(array, rows, cols) {

  if (Array.isArray(array[0])) { // flatten nested arrays
    rows = array.length;
    cols = array[0].length;
    var matrix = new Float64Array(rows * cols);
    matrix.rows = rows;
    matrix.cols = cols;
    for (var i = 0; i < rows; ++i)
      for (var j = 0; j < cols; ++j)
        matrix[i*cols+j] = array[i][j];
    return matrix;
  } else {
    var matrix = new Float64Array(array);
    matrix.rows = rows || array.length
    matrix.cols = cols || 1;
    return matrix;
  }
};

/**
 * String representation
 * @param  {int} precision (optional)
 * @return {string}
 */
Float64Array.prototype.toString = function(precision) {
  precision = precision || 4;
  var str = '';
  for (var i = 0; i < this.rows; ++i) {
    str += (i == 0) ? '[[ ' : ' [ ';
    str += this[i * this.cols + 0].toPrecision(precision);
    for (var j = 1; j < this.cols; ++j)
      str += ', ' + this[i * this.cols + j].toPrecision(precision);
    str += (i == this.rows - 1) ? ' ]]' : ' ],\n';
  }
  return str;
};

/**
 * Returns a copy of a "matrix"
 * @return {Float64Array}
 */
Float64Array.prototype.copy = function() {
  var copy = new Float64Array(this);
  copy.rows = this.rows || this.length;
  copy.cols = this.cols || 1;
  return copy;
};

/**
 * Creates a "matrix" with all entries set to zero
 * @param  {int} rows number of rows
 * @param  {int} cols number of columns
 * @return {Float64Array}
 */
Float64Array.zeros = function(rows, cols) {
  cols = cols || 1;
  var matrix = new Float64Array(rows * cols);
  matrix.rows = rows;
  matrix.cols = cols;
  return matrix;
};

/**
 * Creates a column vector with linearly-spaced elements
 * @param  {float} min minimum value (inclusive)
 * @param  {float} max maximum value (inclusive)
 * @param  {int}   n   number of elements
 * @return {Float64Array}
 */
Float64Array.linspace = function(min, max, n) {
  var matrix = new Float64Array(n);
  var dx = (max - min) / (n - 1);
  for (var i = 0; i < n; ++i)
    matrix[i] = i * dx + min;
  return matrix;
};

/**
 * Creates an n x n identity "matrix"
 * @param  {int} n number of rows and columns
 * @return {Float64Array}
 */
Float64Array.eye = function(n) {
  var matrix = new Float64Array(n * n);
  matrix.rows = n;
  matrix.cols = n;
  for (var i = 0; i < n; ++i)
    matrix[i * n + i] = 1;
  return matrix;
};

/**
 * Creates a "matrix" filled with ones
 * @param  {int} rows number of rows
 * @param  {int} cols number of columns
 * @return {Float64Array}
 */
Float64Array.ones = function(rows, cols) {
  cols = cols || 1;
  var matrix = new Float64Array(rows * cols);
  matrix.rows = rows;
  matrix.cols = cols;
  for (var i = 0; i < matrix.length; ++i)
    matrix[i] = 1;
  return matrix;
};

/**
 * Creates a "matrix" filled with constant
 * @param  {float} const constant
 * @param  {int}   rows number of rows
 * @param  {int}   cols number of columns
 * @return {Float64Array}
 */
Float64Array.constant = function(constant, rows, cols) {
  cols = cols || 1;
  var matrix = new Float64Array(rows * cols);
  matrix.rows = rows;
  matrix.cols = cols;
  for (var i = 0; i < matrix.length; ++i)
    matrix[i] = constant;
  return matrix;
};

/**
 * Build a "matrix" where each element is a function applied to element index
 * @param  {function} f    takes ([i, [j]]) as arguments
 * @param  {int}      rows number of rows
 * @param  {int}      cols number of cols
 * @return {Float64Array}
 */
Float64Array.build = function(f, rows, cols) {
  cols = cols || 1;
  var matrix = Float64Array.zeros(rows, cols);
  for (var i = 0; i < rows; ++i)
    for (var j = 0; j < cols; ++j)
      matrix[i*cols+j] = f(i, j);
  return matrix;
};

/**
 * (in place) Each element is replaced by a function applied to the element index
 * @param  {function} f takes ([i, [j]]) as arguments
 * @return {Float64Array}
 */
Float64Array.prototype.rebuild = function(f) {
  if (this.cols == 1)
    for (var i = 0; i < this.rows; ++i)
      this[i] = f(i, i);
  else
    for (var i = 0; i < this.rows; ++i)
      for (var j = 0; j < this.cols; ++j)
        this[i*this.cols+j] = f(i, j);
  return this;
};

/**
 * Matrix transpose (copy)
 * @return {Float64Array}
 */
Float64Array.prototype.transpose = function() {
  var m = this.rows, n = this.cols;
  var transposed = Float64Array.zeros(n, m);
  for (var i = 0; i < m; ++i)
    for (var j = 0; j < n; ++j)
      transposed[j * m + i] = this[i * n + j];
  return transposed;
};

/**
 * Outer product (form matrix from vector tensor product)
 * @param  {Float64Array} u vector (column or row)
 * @param  {Float64Array} v vector (column or row)
 * @return {Float64Array}
 */
Float64Array.outer = function(u, v) {
  var A = Float64Array.zeros(u.length, v.length);
  for (var i = 0; i < u.length; ++i)
    for (var j = 0; j < v.length; ++j)
      A[i * v.length + j] = u[i] * v[j];
  return A;
};

/**
 * cwise map function onto matrix copy
 * @param  {function} f arguments (A[i], i)
 * @return {Float64Array}
 */
Float64Array.prototype.map = function(f) {
  var A = Float64Array.zeros(this.rows, this.cols);
  for (var i = 0; i < this.length; ++i)
    A[i] = f(this[i], i);
  return A;
}

/**
 * Add two matrices and return sum
 * @param  {Float64Array} other
 * @return {Float64Array}
 */
Float64Array.prototype.add = function(other) {
  if (this.cols != other.cols || this.rows != other.rows) throw 'matrix dimension mismatch';
  var sum = Float64Array.zeros(this.rows, this.cols);
  for (var i = 0; i < this.rows; ++i)
    for (var j = 0; j < this.cols; ++j)
      sum[i*this.cols+j] = this[i*this.cols+j] + other[i*this.cols+j];
  return sum;
};

/**
 * Increment matrix (in place)
 * @param  {Float64Array} other
 * @return {Float64Array}
 */
Float64Array.prototype.increment = function(other) {
  if (this.cols != other.cols || this.rows != other.rows) throw 'matrix dimension mismatch';
  for (var i = 0; i < this.rows; ++i)
    for (var j = 0; j < this.cols; ++j)
      this[i*this.cols+j] += other[i*this.cols+j];
  return this;
};

/**
 * Decrement matrix (in place)
 * @param  {Float64Array} other
 * @return {Float64Array}
 */
Float64Array.prototype.decrement = function(other) {
  if (this.cols != other.cols || this.rows != other.rows) throw 'matrix dimension mismatch';
  for (var i = 0; i < this.rows; ++i)
    for (var j = 0; j < this.cols; ++j)
      this[i*this.cols+j] -= other[i*this.cols+j];
  return this;
};

/**
 * Subtract two matrices and return difference
 * @param  {Float64Array} other
 * @return {Float64Array}
 */
Float64Array.prototype.subtract = function(other) {
  if (this.cols != other.cols || this.rows != other.rows) throw 'matrix dimension mismatch';
  var difference = Float64Array.zeros(this.rows, this.cols);
  for (var i = 0; i < this.rows; ++i)
    for (var j = 0; j < this.cols; ++j)
      difference[i*this.cols+j] = this[i*this.cols+j] - other[i*this.cols+j];
  return difference;
};

/**
 * Compute squared euclidian distance
 * @param  {Float64Array} other
 * @return {float}       square euclidian distance
 */
Float64Array.prototype.dist2 = function(other) {
  var d2 = 0;
  for (var i = 0; i < this.length; ++i)
    d2 += Math.pow(this[i] - other[i], 2);
  return d2;
};

/**
 * Compute euclidian distance
 * @param  {Float64Array} other
 * @return {float}       euclidian distance
 */
Float64Array.prototype.dist = function(other) {
  return Math.sqrt(this.dist2(other));
};

/**
 * Multiply by scalar and return copy
 * @param  {Float64Array} scalar
 * @return {Float64Array}
 */
Float64Array.prototype.scale = function(scalar) {
  var scaled = Float64Array.zeros(this.rows, this.cols);
  for (var i = 0; i < this.rows; ++i)
    for (var j = 0; j < this.cols; ++j)
      scaled[i*this.cols+j] = scalar * this[i*this.cols+j];
  return scaled;
};

/**
 * cwise negate matrix copy
 * @return {Float64Array}
 */
Float64Array.prototype.negate = function() {
  var A = Float64Array.zeros(this.rows, this.cols);
  for (var i = 0; i < this.length; ++i)
    A[i] = -this[i];
  return A;
};

/**
 * Trace of a "matrix," i.e. sum along diagonal
 * @return {float}
 */
Float64Array.prototype.trace = function() {
  var trace = 0;
  for (var i = 0; i < Math.min(this.rows, this.cols); ++i)
    trace += A[i*this.cols+j];
  return trace;
};

/**
 * Element-wise 2-norm (Frobenius-norm for matrices)
 * @return {float}
 */
Float64Array.prototype.norm = function() {
  var norm = 0;
  for (var i = 0; i < this.length; ++i)
    norm += this[i] * this[i];
  return Math.sqrt(norm);
};

/**
 * Element-wise squared norm
 * @return {float}
 */
Float64Array.prototype.norm2 = function() {
  var norm = 0;
  for (var i = 0; i < this.length; ++i)
    norm += this[i] * this[i];
  return norm;
};

/**
 * Sum of all elements
 * @return {float}
 */
Float64Array.prototype.sum = function() {
  var sum = 0;
  for (var i = 0; i < this.length; ++i)
    sum += this[i];
  return sum;
};

/**
 * Get diagonal as a column vector
 * @return {Float64Array}
 */
Float64Array.prototype.diagonal = function() {
  var diagonal = Float64Array.zeros(Math.min(this.rows, this.cols));
  for (var i = 0; i < diagonal.length; ++i)
    diagonal[i] = this[i*this.cols+i];
  return diagonal;
};

/**
 * Create diagonal matrix from a vector
 * @return {Float64Array} a diagonal matrix
 */
Float64Array.prototype.asDiagonal = function() {
  var D = Float64Array.zeros(this.length, this.length);
  for (var i = 0; i < this.length; ++i) {
    D[i * this.length + i] = this[i];
  }
  return D;
};

/**
 * Get row i as a row vector
 * @param  {int} i row index
 * @return {Float64Array}
 */
Float64Array.prototype.row = function(i) {
  var row = Float64Array.zeros(1, this.cols);
  for (var j = 0; j < this.cols; ++j)
    row[j] = this[i*this.cols+j];
  return row;
};

/**
 * Get column j as as column vector
 * @param  {int} j column index
 * @return {Float64Array}
 */
Float64Array.prototype.col = function(j) {
  var col = Float64Array.zeros(this.cols, 1);
  for (var i = 0; i < this.rows; ++i)
    col[i] = this[i*this.cols+j];
  return col;
};

Float64Array.prototype.setRow = function(i, row) {
  for (var j = 0; j < this.cols; ++i)
    this[i*this.cols + j] = row[i];
  return this;
};

Float64Array.prototype.setCol = function(j, col) {
  for (var i = 0; i < this.rows; ++i)
    this[i*this.cols + j] = col[i];
  return this;
};

/**
 * Swap rows i and k
 * @param  {int} i row index
 * @param  {int} k row index
 * @return {Float64Array} (for chaining)
 */
Float64Array.prototype.swap_rows = function(i, k) {
  for (var j = 0; j < this.cols; ++j) {
    var tmp = this[i*this.cols+j];
    this[i*this.cols+j] = this[k*this.cols+j];
    this[k*this.cols+j] = tmp;
  }
  return this;
};

/**
 * Computes determinant using upper triangulation
 * @return {float} NaN if uninvertible
 */
Float64Array.prototype.det = function() {
  if (this.rows != this.cols) throw 'det() requires square matrix';
  if (this.rows == 2 && this.cols == 2) {
    return this[0] * this[3] - this[1] * this[2];
  }
  // upper triangularize, then return product of diagonal
  var U = this.copy();
  for (var i = 0; i < n; ++i) {
    var max = 0;
    for (var row = i; row < n; ++row)
      if (Math.abs(U[row*n+i]) > Math.abs(U[max*n+i]))
        max = row;
    if (max > 0)
      U.swap_rows(i, max);
    if (U[i*n+i] == 0) return NaN;
    for (var row = i + 1; row < n; ++row) {
      var r = U[row*n+i] / U[i*n+i];
      if (r == 0) continue;
      for (var col = i; col < n; ++col);
        U[row*n+col] -= U[i*n+col] * r;
    }
  }
  var det = 1;
  for (var i = 0; i < n; ++i)
    det *= U[i*n+i];
  return det;
};

/**
 * Generalized dot product (sum of element-wise multiplication)
 * @param  {Float64Array} other another "matrix" of same size
 * @return {float}
 */
Float64Array.prototype.dot = function(other) {
  var prod = 0;
  for (var i = 0; i < this.length; ++i)
    prod += this[i] * other[i];
  return prod;
};

/**
 * Matrix multiplication (naive implementation)
 * @param  {Float64Array} other
 * @return {Float64Array}
 */
Float64Array.prototype.multiply = function(other) {
  var A = this, B = other;
  if (A.cols != B.rows) throw 'multiply() dimension mismatch';
  var n = A.rows, l = A.cols, m = B.cols;
  var C = Float64Array.zeros(n, m)
  // vector-vector product
  if (m == 1 && n == 1) {
    C[0] = A.dot(B);
    return C;
  }
  // matrix-vector product
  if (m == 1) {
    for (var i = 0; i < n; ++i)
      for (var j = 0; j < l; ++j)
        C[i] += A[i*l+j] * B[j];
    return C;
  }
  // vector-matrix product
  if (n == 1) {
    for (var j = 0; j < m; ++j) {
      for (var k = 0; k < l; ++k)
        C[j] += A[k] * B[k * m + j];
    }
    return C;
  }
  // matrix-matrix product
  for (var i = 0; i < n; ++i) {
    for (var j = 0; j < m; ++j) {
      var cij = 0;
      for (var k = 0; k < l; ++k)
        cij += A[i * l + k] * B[k * m + j];
      C[i * m + j] = cij;
    }
  }
  return C;
};

/**
 * Computes PA = LU decomposition
 * @return {object} {L, U, P}
 */
Float64Array.prototype.lu = function() {
  if (this.rows != this.cols) throw 'lu() requires square matrix';
  var n = this.rows;
  var L = Float64Array.zeros(n, n);
  var U = Float64Array.zeros(n, n);
  var P = Float64Array.eye(n, n);
  for (var j = 0; j < n; ++j) {
    var max = j;
    for (var i = j; i < n; ++i)
      if (Math.abs(this[i*n+j]) > Math.abs(this[max*n+j]))
        max = i;
    if (j != max)
      P.swap_rows(j, max);
  }
  var PA = P.multiply(this);
  for (var j = 0; j < n; ++j) {
    L[j*n+j] = 1;
    for (var i = 0; i < j+1; ++i) {
      var s = 0;
      for (var k = 0; k < i; ++k)
        s += U[k*n+j] * L[i*n+k]
      U[i*n+j] = PA[i*n+j] - s
    }
    for (var i = j; i < n; ++i) {
      var s = 0;
      for (var k = 0; k < i; ++k)
        s += U[k*n+j] * L[i*n+k]
      L[i*n+j] = (PA[i*n+j] - s) / U[j*n+j];
    }
  }
  return {L:L, U:U, P:P};
};

/**
 * Cholesky A = LL^T decomposition (in-place)
 * @return {[type]} [description]
 */
Float64Array.prototype.chol_inplace = function() {
  if (this.rows != this.cols) throw 'chol_inplace() requires square matrix';
  var A = this;
  var m = A.rows, n = A.cols;
  var i, j, k, s = 0.0;
  for (i = 0; i < n; ++i) {
    for (j = 0; j < (i + 1); ++j) {
      s = 0.0;
      for (k = 0; k < j; ++k)
        s += A[i * n + k] * A[j * n + k];
      if (i != j) A[j * n + i] = 0;
      if (i == j && A[i * n + i] - s < 0) throw "chol_inplace() matrix not positive definite";
      A[i * n + j] = (i == j) ? Math.sqrt(A[i * n + i] - s) : ((A[i * n + j] - s) / A[j * n + j]);
    }
  }
  return A;
};

/**
 * Cholesky A = LL^T decomposition (returns copy)
 * @return {Float64Array}
 */
Float64Array.prototype.chol = function() {
  return this.copy().chol_inplace();
};

/**
 * Solves Lx = b using foward substitution, updates b
 * @param  {Float64Array} b rhs
 * @return {Float64Array}
 */
Float64Array.prototype.fsolve_inplace = function(b) {
  var L = this;
  var m = L.rows, n = L.cols;
  for (var i = 0; i < n; ++i) {
    var s = 0.0
    for (var j = 0; j < i; ++j)
      s += L[i * n + j] * b[j];
    b[i] = (b[i] - s) / L[i * n + i];
  }
  return b;
};

/**
 * Solves Lx = b using foward substitution
 * @param  {Float64Array} b rhs
 * @return {Float64Array}
 */
Float64Array.prototype.fsolve = function(b) {
  return this.fsolve_inplace(b.copy());
};

/**
 * Solves Ux = b using backward substitution, updates b
 * @param  {Float64Array} b rhs
 * @param  {object} options {transpose: false}
 * @return {Float64Array}
 */
Float64Array.prototype.bsolve_inplace = function(b, options) {
  var U = this;
  var m = U.rows, n = U.cols;
  options = options || {};
  var transpose = options.hasOwnProperty('transpose') ? options.transpose : false;
  for (var i = n - 1; i >= 0; --i) {
    var s = 0.0;
    for (var j = i + 1; j < n; ++j)
      s += (transpose ? U[j * n + i] : U[i * n + j]) * b[j];
    b[i] = (b[i] - s) / U[i * n + i];
  }
  return b;
};

/**
 * Solves Ux = b using backward substitution
 * @param  {Float64Array} b rhs
 * @param  {object} options {transpose: false}
 * @return {Float64Array}
 */
Float64Array.prototype.bsolve = function(b, options) {
  return this.bsolve_inplace(b.copy(), options);
};

/**
 * Solve Ax = b using PA = LU decomposition
 * @param  {Float64Array} b rhs
 * @return {Float64Array} x
 */
Float64Array.prototype.lu_solve = function(b) {
  var res = this.lu(), P = res.P, L = res.L, U = res.U;
  return U.bsolve(L.fsolve(P.multiply(b)));
};

/**
 * Computes the matrix inverse using PA = LU decomposition
 * @return {Float64Array} A^-1
 */
Float64Array.prototype.lu_inverse = function() {
  var res = this.lu(), P = res.P, L = res.L, U = res.U;
  var inverse = Float64Array.zeros(this.rows, this.cols);
  var eye = Float64Array.eye(this.rows, this.cols);
  for (var j = 0; j < this.cols; ++j) {
    inverse.setCol(j, U.bsolve(L.fsolve(P.multiply(eye.col(j)))));
  }
  return inverse;
};

/**
 * Solve Ax = b using A = LL^T decomposition
 * @param  {Float64Array} b rhs
 * @return {Float64Array} x
 */
Float64Array.prototype.llt_solve = function(b) {
  var L = this.chol();
  return L.bsolve(L.fsolve(b), {transpose: true});
};

/**
 * Computes the matrix inverse using LL^T decomposition
 * @return {Float64Array} A^-1
 */
Float64Array.prototype.llt_inverse = function() {
  var L = this.chol();
  var inverse = Float64Array.zeros(this.rows, this.cols);
  var eye = Float64Array.eye(this.rows, this.cols);
  for (var j = 0; j < this.cols; ++j) {
    inverse.setCol(j, L.bsolve(L.fsolve(eye.col(j)), {transpose: true}));
  }
  return inverse;
};

/**
 * Solve Ax = b using A = LL^T decomposition (in-place)
 * @param  {Float64Array} b rhs
 * @return {Float64Array} x
 */
Float64Array.prototype.llt_solve_inplace = function(b) {
  var L = this.chol_inplace();
  return L.bsolve_inplace(L.fsolve_inplace(b), {transpose: true});
};


/**
 * Computes diagonal matrix D of eigenvalues and matrix V whose columns are the corresponding
 * right eigenvectors so that AV = VD
 * @param  {object} options tolerance and maxIter
 * @return {object}         V:V D:D
 */
Float64Array.prototype.jacobiRotation = function(options) {

  if (this.cols != this.rows) throw 'matrix must be square';

  if (arguments.length < 1)
    options = {};

  var maxIter = options.maxIter || 100;
  var tolerance = options.tolerance || 1e-5;

  var n = this.rows;
  var D = this.copy();
  var V = Float64Array.eye(n, n);

  var iter, maxOffDiag, p, q;
  for (iter = 0; iter < maxIter; ++iter) {

    // find max off diagonal term at (p, q)
    maxOffDiag = 0;
    for (var i = 0; i < n - 1; ++i) {
      for (var j = i + 1; j < n; ++j) {
        if (Math.abs(D[i * n + j]) > maxOffDiag) {
          maxOffDiag = Math.abs(D[i * n + j]);
          p = i; q = j;
        }
      }
    }

    if (maxOffDiag < tolerance)
      break;

    // Rotates matrix D through theta in pq-plane to set D[p][q] = 0
    // Rotation stored in matrix V whose columns are eigenvectors of D
    // d = cot 2 * theta, t = tan theta, c = cos theta, s = sin theta
    var d = (D[p * n + p] - D[q * n + q]) / (2.0 * D[p * n + q]);
    var t = Math.sign(d) / (Math.abs(d) + Math.sqrt(d * d + 1));
    var c = 1.0 / Math.sqrt(t * t + 1);
    var s = t * c;
    D[p * n + p] += t * D[p * n + q];
    D[q * n + q] -= t * D[p * n + q];
    D[p * n + q] =      D[q * n + p] = 0.0;
    for (var k = 0; k < n; k++) {  // Transform D
      if (k != p && k != q) {
        var akp =  c * D[k * n + p] + s * D[k * n + q];
        var akq = -s * D[k * n + p] + c * D[k * n + q];
        D[k * n + p] = akp;
        D[p * n + k] = akp;
        D[k * n + q] = akq;
        D[q * n + k] = akq;
      }
    }
    for (var k = 0; k < n; k++) {  // Store V
      var rkp =  c * V[k * n + p] + s * V[k * n + q];
      var rkq = -s * V[k * n + p] + c * V[k * n + q];
      V[k * n + p] = rkp;
      V[k * n + q] = rkq;
    }
  }

  if (iter == maxIter) {
    console.log('Hit maxIter: ', maxOffDiag, ' > ', tolerance);
  }

  return {V:V, D:D, eigenvalues: D.diagonal(), eigenvectors: V};

};

Float64Array.prototype.maxCoeff = function() {
  var max = this[0];
  for (var i = 0; i < this.length; ++i) {
    if (this[i] > max)
      max = this[i];
  }
  return max;
};

Float64Array.prototype.cwiseProduct = function(other) {
  var A = this.copy();
  for (var i = 0; i < this.length; ++i) {
    A[i] = this[i] * other[i];
  }
  return A;
};

Float64Array.prototype.cwiseQuotient = function(other) {
  var A = this.copy();
  for (var i = 0; i < this.length; ++i) {
    A[i] = this[i] / other[i];
  }
  return A;
};

Float64Array.prototype.cwiseInverse = function() {
  var A = this.copy();
  for (var i = 0; i < this.length; ++i) {
    A[i] = 1.0 / this[i];
  }
  return A;
};

Float64Array.prototype.cwiseSqrt = function() {
  var A = this.copy();
  for (var i = 0; i < this.length; ++i) {
    A[i] = Math.sqrt(this[i]);
  }
  return A;
};

// get sub-block of matrix
Float64Array.prototype.getBlock = function(top, left, rows, cols) {
	var B = new Float64Array(rows*cols);
	B.rows = rows;
	B.cols = cols;
	for (var i = 0; i < rows; i++) {
		for (var j = 0; j < cols; j++) {
			B[i*B.cols+j] = this[(i+top)*this.cols + (j+left)];
		}
	}
	return B;
}

// set sub-block of matrix to another matrix
Float64Array.prototype.setBlock = function(top, left, A) {
	for (var i = 0; i < A.rows; i++) {
		for (var j = 0; j < A.cols; j++) {
			this[(i+top)*this.cols + (j+left)] = A[i*A.cols+j];
		}
	}
};

// QR decomposition using Householder reflections.
Float64Array.prototype.qr = function() {
	var make_householder = function(a) {
		var v = a.scale(1 / (a[0] + Math.sign(a[0]) * a.norm()));
		v[0] = 1;
		var H = Float64Array.eye(a.length);
		H.decrement(Float64Array.outer(v, v).scale(2 / v.dot(v)));
		return H;
	};
	var A = this.copy();
	var m = A.rows;
	var n = A.cols;
	var Q = Float64Array.eye(m);
	var upper = n - ((m == n) ? 1 : 0);
	for (var i = 0; i < upper; i++) {
		var a = A.getBlock(i, i, m - i, 1);
		var H = Float64Array.eye(m);
		H.setBlock(i, i, make_householder(a));
		Q = Q.multiply(H);
		A = H.multiply(A);
	}
	return {Q:Q, R:A};
};

var zeros = Float64Array.zeros;
var eye = Float64Array.eye;
var linspace = Float64Array.linspace;
var matrix = Float64Array.matrix;

