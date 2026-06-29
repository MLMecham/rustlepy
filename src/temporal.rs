//! Temporal features (Layer 2): delta via Savitzky-Golay derivative.
//!
//! Matches `librosa.feature.delta` = `scipy.signal.savgol_filter(data, width,
//! polyorder=order, deriv=order, axis=-1, mode="interp")`. For `mode="interp"`
//! the interior is the SG convolution and the edges fit a polynomial to the
//! first/last `width` samples — both are the same operation: fit a degree-`p`
//! polynomial by least squares and evaluate its `d`-th derivative.

use ndarray::{Array2, ArrayView2};

/// Solve a small dense system `a x = b` (Gaussian elimination, partial pivot).
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for col in 0..n {
        let mut piv = col;
        for r in (col + 1)..n {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        a.swap(col, piv);
        b.swap(col, piv);
        let d = a[col][col];
        for r in (col + 1)..n {
            let f = a[r][col] / d;
            for cc in col..n {
                a[r][cc] -= f * a[col][cc];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let mut s = b[r];
        for cc in (r + 1)..n {
            s -= a[r][cc] * x[cc];
        }
        x[r] = s / a[r][r];
    }
    x
}

/// Least-squares polynomial of degree `polyorder` fit to `(k, window[k])` for
/// `k = 0..window.len()`, with its `deriv`-th derivative evaluated at `t`.
/// (delta = 1.0, so no spacing rescale — matches librosa's call.)
fn local_poly_deriv(window: &[f64], t: f64, polyorder: usize, deriv: usize) -> f64 {
    let w = window.len();
    let ncoef = polyorder + 1;

    // Normal equations (V^T V) c = V^T y, V[k, j] = k^j. The matrix depends only
    // on (w, polyorder) — could be hoisted/factored once; kept inline for clarity.
    let mut moments = vec![0.0_f64; 2 * polyorder + 1];
    let mut rhs = vec![0.0_f64; ncoef];
    for k in 0..w {
        let kf = k as f64;
        let mut p = 1.0;
        for m in 0..moments.len() {
            moments[m] += p;
            if m < ncoef {
                rhs[m] += p * window[k];
            }
            p *= kf;
        }
    }
    let mat: Vec<Vec<f64>> = (0..ncoef)
        .map(|a| (0..ncoef).map(|b| moments[a + b]).collect())
        .collect();
    let c = solve(mat, rhs);

    // d^deriv/dt^deriv [ sum_j c[j] t^j ] = sum_{j>=deriv} c[j]*(j!/(j-deriv)!)*t^(j-deriv)
    let mut val = 0.0;
    for j in deriv..ncoef {
        let mut coef = c[j];
        for q in 0..deriv {
            coef *= (j - q) as f64;
        }
        val += coef * t.powi((j - deriv) as i32);
    }
    val
}

fn delta_row(row: &[f64], width: usize, order: usize) -> Vec<f64> {
    let n = row.len();
    let halflen = width / 2;
    let mut out = vec![0.0_f64; n];
    for i in 0..n {
        let (start, eval_pos) = if i < halflen {
            (0usize, i) // left edge: fit first `width`, eval at i
        } else if i >= n - halflen {
            (n - width, i - (n - width)) // right edge: fit last `width`
        } else {
            (i - halflen, halflen) // interior: centered window, eval at center
        };
        out[i] = local_poly_deriv(&row[start..start + width], eval_pos as f64, order, order);
    }
    out
}

/// Delta features along axis 1 (each row independently), matching
/// `librosa.feature.delta(data, width, order, axis=-1, mode="interp")`.
pub fn delta(data: ArrayView2<f64>, width: usize, order: usize) -> Array2<f64> {
    let (n_rows, n_cols) = data.dim();
    let mut out = Array2::<f64>::zeros((n_rows, n_cols));
    for r in 0..n_rows {
        let row = data.row(r).to_vec();
        let d = delta_row(&row, width, order);
        for c in 0..n_cols {
            out[[r, c]] = d[c];
        }
    }
    out
}
