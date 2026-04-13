/// Linear algebra kernels for dense matrices.
///
/// Uses `nalgebra::DMatrix<f64>` for storage.
/// SVD uses LAPACK dgesdd (Apple Accelerate) for large matrices,
/// nalgebra for small ones (≤8×8) where call overhead dominates.
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;

extern crate lapack_src;

// ---------------------------------------------------------------------------
// SVD helpers
// ---------------------------------------------------------------------------

/// Full (thin) SVD: A = U diag(S) Vt.
/// Uses LAPACK dgesdd (Apple Accelerate) for large matrices,
/// nalgebra for small (≤8×8) where call overhead dominates.
/// Returns (U, S_vec, Vt).
pub fn svd_full(a: &DMatrix<f64>) -> (DMatrix<f64>, DVector<f64>, DMatrix<f64>) {
    // For small matrices, nalgebra avoids LAPACK call/copy overhead
    if a.nrows() <= 8 && a.ncols() <= 8 {
        let svd = a.clone().svd(true, true);
        let u = svd.u.expect("SVD U");
        let vt = svd.v_t.expect("SVD Vt");
        return (u, svd.singular_values, vt);
    }

    // LAPACK path for larger matrices
    let m = a.nrows() as i32;
    let n = a.ncols() as i32;
    let k = m.min(n) as usize;

    // LAPACK dgesdd expects column-major (Fortran order).
    // nalgebra stores column-major, so we can use the data directly.
    let mut a_data: Vec<f64> = Vec::with_capacity((m * n) as usize);
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            a_data.push(a[(i, j)]);
        }
    }

    let mut s = vec![0.0f64; k];
    let mut u_data = vec![0.0f64; (m * m.min(n)) as usize];
    let mut vt_data = vec![0.0f64; (m.min(n) * n) as usize];
    let mut work = vec![0.0f64; 1];
    let mut iwork = vec![0i32; 8 * k];
    let mut info = 0i32;
    let ldu = m;
    let ldvt = m.min(n);

    // Query optimal workspace
    unsafe {
        lapack::dgesdd(
            b'S', m, n, &mut a_data, m,
            &mut s, &mut u_data, ldu, &mut vt_data, ldvt,
            &mut work, -1, &mut iwork, &mut info,
        );
    }
    let lwork = work[0] as i32;
    work.resize(lwork as usize, 0.0);

    // Compute SVD
    unsafe {
        lapack::dgesdd(
            b'S', m, n, &mut a_data, m,
            &mut s, &mut u_data, ldu, &mut vt_data, ldvt,
            &mut work, lwork, &mut iwork, &mut info,
        );
    }
    assert!(info == 0, "LAPACK dgesdd failed with info={info}");

    // Convert column-major LAPACK output back to nalgebra DMatrix
    let u = DMatrix::from_fn(m as usize, k, |i, j| u_data[j * m as usize + i]);
    let s_vec = DVector::from_vec(s);
    let vt = DMatrix::from_fn(k, n as usize, |i, j| vt_data[j * k + i]);

    (u, s_vec, vt)
}

/// Truncated SVD keeping only the first `k` components.
pub fn svd_truncated(
    a: &DMatrix<f64>,
    k: usize,
) -> (DMatrix<f64>, DVector<f64>, DMatrix<f64>) {
    let (u, s, vt) = svd_full(a);
    let k = k.min(s.len());
    let u_trunc = u.columns(0, k).into_owned();
    let s_trunc = s.rows(0, k).into_owned();
    let vt_trunc = vt.rows(0, k).into_owned();
    (u_trunc, s_trunc, vt_trunc)
}

/// SVD dispatching: if 0 < k < min(nrows, ncols) use truncated, else full.
pub fn svd_dispatch(
    a: &DMatrix<f64>,
    k: usize,
) -> (DMatrix<f64>, DVector<f64>, DMatrix<f64>) {
    if k > 0 && k < a.nrows().min(a.ncols()) {
        svd_truncated(a, k)
    } else {
        let (u, s, vt) = svd_full(a);
        if k > 0 {
            let k = k.min(s.len());
            (
                u.columns(0, k).into_owned(),
                s.rows(0, k).into_owned(),
                vt.rows(0, k).into_owned(),
            )
        } else {
            (u, s, vt)
        }
    }
}

// ---------------------------------------------------------------------------
// QR helpers
// ---------------------------------------------------------------------------

/// QR decomposition: returns Q only (economy/thin).
/// Uses fixed-size for small matrices.
pub fn qr_q(a: &DMatrix<f64>) -> DMatrix<f64> {
    a.clone().qr().q()  // nalgebra QR already optimizes well
}

/// QR decomposition: returns (Q, R).
pub fn qr_full(a: &DMatrix<f64>) -> (DMatrix<f64>, DMatrix<f64>) {
    let qr = a.clone().qr();
    (qr.q(), qr.r())
}

// ---------------------------------------------------------------------------
// Matrix inverse
// ---------------------------------------------------------------------------

/// Compute the inverse of a square matrix.
/// Uses analytic formula for 2x2, LU for larger.
/// Falls back to pseudo-inverse if singular.
pub fn mat_inv(a: &DMatrix<f64>) -> DMatrix<f64> {
    let n = a.nrows();
    if n == 2 && a.ncols() == 2 {
        let det = a[(0, 0)] * a[(1, 1)] - a[(0, 1)] * a[(1, 0)];
        if det.abs() < 1e-30 {
            return mat_pinv(a);
        }
        let inv_det = 1.0 / det;
        DMatrix::from_row_slice(2, 2, &[
            a[(1, 1)] * inv_det, -a[(0, 1)] * inv_det,
            -a[(1, 0)] * inv_det, a[(0, 0)] * inv_det,
        ])
    } else {
        a.clone()
            .try_inverse()
            .unwrap_or_else(|| mat_pinv(a))
    }
}

/// Compute the pseudo-inverse of a matrix.
pub fn mat_pinv(a: &DMatrix<f64>) -> DMatrix<f64> {
    let svd = a.clone().svd(true, true);
    svd.pseudo_inverse(1e-15).expect("pseudo-inverse failed")
}

// ---------------------------------------------------------------------------
// BLAS-backed matrix multiply for large matrices
// ---------------------------------------------------------------------------

extern "C" {
    fn cblas_dgemm(
        order: i32, transa: i32, transb: i32,
        m: i32, n: i32, k: i32,
        alpha: f64,
        a: *const f64, lda: i32,
        b: *const f64, ldb: i32,
        beta: f64,
        c: *mut f64, ldc: i32,
    );
}

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_COL_MAJOR: i32 = 102;
const CBLAS_NO_TRANS: i32 = 111;
const CBLAS_TRANS: i32 = 112;

/// Matrix multiply C = A * B using BLAS dgemm (column-major, zero-copy).
/// Falls back to nalgebra for small matrices.
pub fn matmul(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();
    assert_eq!(k, b.nrows(), "matmul dimension mismatch");

    if m <= 8 && n <= 8 && k <= 8 {
        return a * b;
    }

    // nalgebra stores column-major — use Fortran-order BLAS directly (zero copy!)
    let mut c_data = vec![0.0f64; m * n];
    unsafe {
        cblas_dgemm(
            CBLAS_COL_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
            m as i32, n as i32, k as i32,
            1.0,
            a.as_slice().as_ptr(), m as i32,  // nalgebra col-major = Fortran lda=m
            b.as_slice().as_ptr(), k as i32,
            0.0,
            c_data.as_mut_ptr(), m as i32,
        );
    }
    // c_data is column-major, which is what DMatrix expects
    DMatrix::from_vec(m, n, c_data)
}

/// C = A^T * B using BLAS (column-major, zero-copy).
pub fn matmul_at_b(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    let m = a.ncols();
    let k = a.nrows();
    let n = b.ncols();
    assert_eq!(k, b.nrows());

    if m <= 8 && n <= 8 && k <= 8 {
        return a.transpose() * b;
    }

    let mut c_data = vec![0.0f64; m * n];
    unsafe {
        cblas_dgemm(
            CBLAS_COL_MAJOR, CBLAS_TRANS, CBLAS_NO_TRANS,
            m as i32, n as i32, k as i32,
            1.0,
            a.as_slice().as_ptr(), k as i32,  // A is k×m in memory, lda=k
            b.as_slice().as_ptr(), k as i32,
            0.0,
            c_data.as_mut_ptr(), m as i32,
        );
    }
    DMatrix::from_vec(m, n, c_data)
}

// ---------------------------------------------------------------------------
// Complex eigendecomposition
// ---------------------------------------------------------------------------

/// Eigendecomposition of a real square matrix.
/// Returns (eigenvalues, eigenvectors) where eigenvalues are complex and
/// eigenvectors are complex column vectors.
///
/// Eigenvalues are sorted by descending magnitude.
pub fn eig_complex(
    a: &DMatrix<f64>,
) -> (Vec<Complex64>, DMatrix<Complex64>) {
    let n = a.nrows();
    assert_eq!(n, a.ncols(), "eig requires a square matrix");

    // Use nalgebra's Schur decomposition to get eigenvalues
    let schur = a.clone().schur();
    let (q, t) = schur.unpack();

    // Extract eigenvalues from quasi-triangular T
    let mut eigenvalues: Vec<Complex64> = Vec::with_capacity(n);
    let mut eigenvectors_complex: Vec<Vec<Complex64>> = Vec::new();

    let mut i = 0;
    while i < n {
        if i + 1 < n && t[(i + 1, i)].abs() > 1e-15 {
            // 2x2 block: complex conjugate pair
            let a11 = t[(i, i)];
            let a12 = t[(i, i + 1)];
            let a21 = t[(i + 1, i)];
            let a22 = t[(i + 1, i + 1)];
            let tr = a11 + a22;
            let det = a11 * a22 - a12 * a21;
            let disc = tr * tr - 4.0 * det;
            if disc < 0.0 {
                let re = tr / 2.0;
                let im = (-disc).sqrt() / 2.0;
                eigenvalues.push(Complex64::new(re, im));
                eigenvalues.push(Complex64::new(re, -im));
            } else {
                let sqrt_disc = disc.sqrt();
                eigenvalues.push(Complex64::new((tr + sqrt_disc) / 2.0, 0.0));
                eigenvalues.push(Complex64::new((tr - sqrt_disc) / 2.0, 0.0));
            }
            i += 2;
        } else {
            eigenvalues.push(Complex64::new(t[(i, i)], 0.0));
            i += 1;
        }
    }

    // Compute eigenvectors from Schur vectors
    // For real eigenvalues: eigenvector = Q * e_i from T
    // For complex pairs: reconstruct from 2x2 blocks
    let mut j = 0;
    while j < n {
        if j + 1 < n && t[(j + 1, j)].abs() > 1e-15 {
            // Complex conjugate pair
            // Solve (T - lambda*I) * v = 0 using back-substitution on
            // the quasi-triangular matrix
            let lambda = eigenvalues[j];

            // Solve for eigenvector of T
            let v1 = solve_schur_eigenvector(&t, lambda);
            let v2 = v1.iter().map(|c| c.conj()).collect::<Vec<_>>();

            // Transform back: eigvec = Q * v
            let mut ev1 = vec![Complex64::new(0.0, 0.0); n];
            let mut ev2 = vec![Complex64::new(0.0, 0.0); n];
            for row in 0..n {
                for col in 0..n {
                    ev1[row] += Complex64::new(q[(row, col)], 0.0) * v1[col];
                    ev2[row] += Complex64::new(q[(row, col)], 0.0) * v2[col];
                }
            }
            eigenvectors_complex.push(ev1);
            eigenvectors_complex.push(ev2);
            j += 2;
        } else {
            // Real eigenvalue - extract Schur vector column
            let mut ev = vec![Complex64::new(0.0, 0.0); n];

            // Solve (T - lambda*I)v = 0 for the Schur form
            let lambda = eigenvalues[j];
            let v = solve_schur_eigenvector(&t, lambda);

            for row in 0..n {
                for col in 0..n {
                    ev[row] += Complex64::new(q[(row, col)], 0.0) * v[col];
                }
            }
            eigenvectors_complex.push(ev);
            j += 1;
        }
    }

    // Sort by descending magnitude
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        eigenvalues[b]
            .norm()
            .partial_cmp(&eigenvalues[a].norm())
            .unwrap()
    });

    let sorted_evals: Vec<Complex64> = indices.iter().map(|&i| eigenvalues[i]).collect();
    let mut phi = DMatrix::from_element(n, n, Complex64::new(0.0, 0.0));
    for (col_out, &idx) in indices.iter().enumerate() {
        for row in 0..n {
            phi[(row, col_out)] = eigenvectors_complex[idx][row];
        }
    }

    // Normalize eigenvectors
    for col in 0..n {
        let norm: f64 = (0..n)
            .map(|row| phi[(row, col)].norm_sqr())
            .sum::<f64>()
            .sqrt();
        if norm > 1e-15 {
            for row in 0..n {
                phi[(row, col)] /= norm;
            }
        }
    }

    (sorted_evals, phi)
}

/// Solve for eigenvector of a quasi-upper-triangular (Schur form) matrix.
fn solve_schur_eigenvector(
    t: &DMatrix<f64>,
    lambda: Complex64,
) -> Vec<Complex64> {
    let n = t.nrows();
    let mut v = vec![Complex64::new(0.0, 0.0); n];

    // Back-substitution on (T - lambda * I)v = 0
    // Start from the eigenvalue's position and work backwards
    // Find the diagonal entry closest to lambda
    let mut start = n - 1;
    let mut min_dist = f64::MAX;
    let mut k = 0;
    while k < n {
        if k + 1 < n && t[(k + 1, k)].abs() > 1e-15 {
            // 2x2 block
            let a11 = t[(k, k)];
            let a22 = t[(k + 1, k + 1)];
            let tr = a11 + a22;
            let re = tr / 2.0;
            let dist = (Complex64::new(re, 0.0) - lambda).norm();
            if dist < min_dist {
                min_dist = dist;
                start = k;
            }
            k += 2;
        } else {
            let dist = (Complex64::new(t[(k, k)], 0.0) - lambda).norm();
            if dist < min_dist {
                min_dist = dist;
                start = k;
            }
            k += 1;
        }
    }

    // Set initial component
    if start + 1 < n && t[(start + 1, start)].abs() > 1e-15 {
        // 2x2 block - solve the small eigenvalue problem
        let a11 = t[(start, start)];
        let a12 = t[(start, start + 1)];
        // v[start] and v[start+1] from (T_block - lambda*I) * [v1; v2] = 0
        // (a11 - lambda)*v1 + a12*v2 = 0
        let lam_minus_a11 = lambda - Complex64::new(a11, 0.0);
        if lam_minus_a11.norm() > 1e-15 {
            v[start + 1] = Complex64::new(1.0, 0.0);
            v[start] = -Complex64::new(a12, 0.0) / lam_minus_a11;
        } else {
            v[start] = Complex64::new(1.0, 0.0);
            v[start + 1] = Complex64::new(0.0, 0.0);
        }
    } else {
        v[start] = Complex64::new(1.0, 0.0);
    }

    // Back-substitution for rows before `start`
    if start > 0 {
        for i in (0..start).rev() {
            let mut sum = Complex64::new(0.0, 0.0);
            for j in (i + 1)..n {
                sum += Complex64::new(t[(i, j)], 0.0) * v[j];
            }
            let diag = Complex64::new(t[(i, i)], 0.0) - lambda;
            if diag.norm() > 1e-15 {
                v[i] = -sum / diag;
            }
        }
    }

    v
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Check if two matrices are element-wise close.
pub fn allclose(a: &DMatrix<f64>, b: &DMatrix<f64>, rtol: f64) -> bool {
    if a.nrows() != b.nrows() || a.ncols() != b.ncols() {
        return false;
    }
    let atol = 1e-8;
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            let diff = (a[(i, j)] - b[(i, j)]).abs();
            let tol = atol + rtol * b[(i, j)].abs();
            if diff > tol {
                return false;
            }
        }
    }
    true
}

/// Build a block matrix from 4 sub-matrices: [[a, b], [c, d]].
pub fn block2x2(
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
    c: &DMatrix<f64>,
    d: &DMatrix<f64>,
) -> DMatrix<f64> {
    let rows_top = a.nrows();
    let rows_bot = c.nrows();
    let cols_left = a.ncols();
    let cols_right = b.ncols();
    let mut result = DMatrix::zeros(rows_top + rows_bot, cols_left + cols_right);
    result
        .view_mut((0, 0), (rows_top, cols_left))
        .copy_from(a);
    result
        .view_mut((0, cols_left), (rows_top, cols_right))
        .copy_from(b);
    result
        .view_mut((rows_top, 0), (rows_bot, cols_left))
        .copy_from(c);
    result
        .view_mut((rows_top, cols_left), (rows_bot, cols_right))
        .copy_from(d);
    result
}

/// Symmetrize a matrix: (A + A^T) / 2. In-place, no allocation.
pub fn symmetrize(a: &mut DMatrix<f64>) {
    let n = a.nrows();
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = (a[(i, j)] + a[(j, i)]) * 0.5;
            a[(i, j)] = avg;
            a[(j, i)] = avg;
        }
    }
}

/// Create a diagonal matrix from a vector.
pub fn diag(v: &DVector<f64>) -> DMatrix<f64> {
    DMatrix::from_diagonal(v)
}

/// Append columns: [a | b].
pub fn append_cols(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    if a.ncols() == 0 {
        return b.clone();
    }
    if b.ncols() == 0 {
        return a.clone();
    }
    let mut result = DMatrix::zeros(a.nrows(), a.ncols() + b.ncols());
    result.view_mut((0, 0), (a.nrows(), a.ncols())).copy_from(a);
    result
        .view_mut((0, a.ncols()), (b.nrows(), b.ncols()))
        .copy_from(b);
    result
}

/// Stack rows: [a; b].
pub fn stack_rows(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    if a.nrows() == 0 {
        return b.clone();
    }
    if b.nrows() == 0 {
        return a.clone();
    }
    let mut result = DMatrix::zeros(a.nrows() + b.nrows(), a.ncols());
    result.view_mut((0, 0), (a.nrows(), a.ncols())).copy_from(a);
    result
        .view_mut((a.nrows(), 0), (b.nrows(), b.ncols()))
        .copy_from(b);
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svd_roundtrip() {
        let a = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let (u, s, vt) = svd_full(&a);
        let reconstructed = &u * DMatrix::from_diagonal(&s) * &vt;
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                assert!(
                    (a[(i, j)] - reconstructed[(i, j)]).abs() < 1e-10,
                    "SVD roundtrip failed at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_svd_truncated() {
        let a = DMatrix::from_row_slice(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let (u, s, vt) = svd_truncated(&a, 2);
        assert_eq!(u.ncols(), 2);
        assert_eq!(s.len(), 2);
        assert_eq!(vt.nrows(), 2);
    }

    #[test]
    fn test_qr_orthogonality() {
        let a = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let q = qr_q(&a);
        let qtq = q.transpose() * &q;
        let eye = DMatrix::identity(q.ncols(), q.ncols());
        assert!(allclose(&qtq, &eye, 1e-10));
    }

    #[test]
    fn test_mat_inv() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let a_inv = mat_inv(&a);
        let eye = &a * &a_inv;
        let expected = DMatrix::identity(2, 2);
        assert!(allclose(&eye, &expected, 1e-10));
    }

    #[test]
    fn test_eig_complex_real_eigenvalues() {
        // Diagonal matrix with known eigenvalues
        let a = DMatrix::from_row_slice(2, 2, &[3.0, 0.0, 0.0, 1.0]);
        let (evals, _) = eig_complex(&a);
        assert_eq!(evals.len(), 2);
        // Sorted by descending magnitude
        assert!((evals[0].re - 3.0).abs() < 1e-10);
        assert!((evals[1].re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_eig_complex_complex_eigenvalues() {
        // Rotation matrix: eigenvalues should be e^{+/- i*pi/4}
        let angle = std::f64::consts::FRAC_PI_4;
        let c = angle.cos();
        let s = angle.sin();
        let a = DMatrix::from_row_slice(2, 2, &[c, -s, s, c]);
        let (evals, _) = eig_complex(&a);
        assert_eq!(evals.len(), 2);
        // Both have magnitude 1
        assert!((evals[0].norm() - 1.0).abs() < 1e-10);
        assert!((evals[1].norm() - 1.0).abs() < 1e-10);
        // Imaginary parts should be +/- sin(pi/4)
        let im_max = evals[0].im.abs().max(evals[1].im.abs());
        assert!((im_max - s).abs() < 1e-10);
    }

    #[test]
    fn test_block2x2() {
        let a = DMatrix::from_row_slice(1, 1, &[1.0]);
        let b = DMatrix::from_row_slice(1, 1, &[2.0]);
        let c = DMatrix::from_row_slice(1, 1, &[3.0]);
        let d = DMatrix::from_row_slice(1, 1, &[4.0]);
        let block = block2x2(&a, &b, &c, &d);
        assert_eq!(block[(0, 0)], 1.0);
        assert_eq!(block[(0, 1)], 2.0);
        assert_eq!(block[(1, 0)], 3.0);
        assert_eq!(block[(1, 1)], 4.0);
    }

    #[test]
    fn test_allclose() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let b = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert!(allclose(&a, &b, 1e-10));
        let c = DMatrix::from_row_slice(2, 2, &[1.1, 2.0, 3.0, 4.0]);
        assert!(!allclose(&a, &c, 1e-10));
    }
}
