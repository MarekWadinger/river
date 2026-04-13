/// Online Dynamic Mode Decomposition (DMD).
///
/// Port of `vendor/river/river/decomposition/odmd.py` (OnlineDMD).
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::linalg;
use crate::linalg::append_cols;
use crate::online_svd::OnlineSvdZhang;

#[derive(Clone, Copy)]
pub enum SvdModify {
    Update,
    Revert,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct OnlineDmd {
    pub r: usize,
    pub w: f64,
    pub initialize: usize,
    pub exponential_weighting: bool,
    pub eig_rtol: Option<f64>,

    // State
    pub m: usize,
    pub n_seen: usize,
    pub a: DMatrix<f64>,
    pub p: DMatrix<f64>,

    // SVD for truncation (r < m)
    pub svd: Option<OnlineSvdZhang>,

    // Init buffers
    pub x_init: Option<DMatrix<f64>>,
    pub y_init: Option<DMatrix<f64>>,

    // For unsupervised mode
    pub x_last: Option<Vec<f64>>,
    pub x_first: Option<Vec<f64>>,

    // Cached eigendecomposition
    pub a_last: DMatrix<f64>,
    pub a_allclose_cached: bool,

    // Track if first update has been done
    pub inited: bool,
    pub p_inited: bool,
}

impl OnlineDmd {
    pub fn new(
        r: usize,
        w: f64,
        initialize: usize,
        exponential_weighting: bool,
        eig_rtol: Option<f64>,
    ) -> Self {
        let svd = if r != 0 {
            Some(OnlineSvdZhang::new(r, 1e-12, false))
        } else {
            None
        };

        Self {
            r,
            w,
            initialize,
            exponential_weighting,
            eig_rtol,
            m: 0,
            n_seen: 0,
            a: DMatrix::zeros(0, 0),
            p: DMatrix::zeros(0, 0),
            svd,
            x_init: None,
            y_init: None,
            x_last: None,
            x_first: None,
            a_last: DMatrix::zeros(0, 0),
            a_allclose_cached: false,
            inited: false,
            p_inited: false,
        }
    }

    fn init_update(&mut self) {
        if self.r == 0 {
            self.r = self.m;
        }
        if self.initialize > 0 && self.initialize < self.r {
            self.initialize = self.r;
        }
        self.a = DMatrix::identity(self.r, self.r);
        self.a_last = self.a.clone();
        self.x_init = Some(DMatrix::zeros(self.initialize, self.m));
        self.y_init = Some(DMatrix::zeros(self.initialize, self.m));
        self.inited = true;
    }

    /// Truncate using SVD: project x, y into reduced space, rotate A and P.
    pub fn truncate_w_svd(
        &mut self,
        x: &DMatrix<f64>,
        y: &DMatrix<f64>,
        modify: SvdModify,
    ) -> (DMatrix<f64>, DMatrix<f64>) {
        // Compute U_new^T @ U_prev without cloning the 80×2 matrices.
        // We save U_prev^T (r × m, small-ish) before SVD modifies U.
        let u_prev_t = self
            .svd
            .as_ref()
            .expect("SVD must be initialized")
            .u
            .transpose(); // r × m

        {
            let svd = self.svd.as_mut().expect("SVD must be initialized");
            match modify {
                SvdModify::Update => svd.update_mat(x),
                SvdModify::Revert => {
                    let row = x.row(0);
                    let data: Vec<f64> = row.iter().cloned().collect();
                    svd.revert(&data, -1);
                }
            }
        }

        let svd_ref = self.svd.as_ref().expect("SVD must be initialized");
        // uu = U_new^T @ U_prev (r×r) — use BLAS for the m×r matmuls
        let uu = linalg::matmul_at_b(&svd_ref.u, &u_prev_t.transpose()); // r × r

        let x_proj = linalg::matmul(x, &svd_ref.u); // n × r
        let p_a = self.a.nrows();
        let y_cols = y.ncols().min(svd_ref.u.nrows());
        let u_sub = svd_ref.u.rows(0, y_cols).columns(0, p_a).clone_owned();
        let y_sub = y.columns(0, y_cols).clone_owned();
        let y_proj = linalg::matmul(&y_sub, &u_sub);

        // Rotate A: A = UU @ A @ UU^T
        let uu_t = uu.transpose();
        if self.a.nrows() == self.a.ncols() {
            let tmp = linalg::matmul(&uu, &self.a);
            self.a = linalg::matmul(&tmp, &uu_t);
        } else {
            let p = self.a.nrows();
            let uu_p = uu.view((0, 0), (p, p)).clone_owned();
            let tmp = linalg::matmul(&uu_p, &self.a);
            self.a = linalg::matmul(&tmp, &uu_t);
        }

        // Rotate P: P = inv(UU @ inv(P) @ UU^T) / w
        let p_inv = linalg::mat_inv(&self.p);
        let tmp = linalg::matmul(&uu, &p_inv);
        let rotated = linalg::matmul(&tmp, &uu_t);
        self.p = linalg::mat_inv(&rotated) / self.w;

        (x_proj, y_proj)
    }

    /// Core A, P update with new data.
    pub fn update_a_p(
        &mut self,
        x: &DMatrix<f64>,
        y: &DMatrix<f64>,
        w_mat: &DMatrix<f64>,
    ) {
        let xt = x.transpose();
        let ax = linalg::matmul(&self.a, &xt);
        let px = linalg::matmul(&self.p, &xt);
        let pxt = px.transpose();
        let gamma_input = w_mat + linalg::matmul(x, &px);
        let gamma = linalg::mat_inv(&gamma_input);

        // A += (Y^T - A@X^T) @ Gamma @ (P@X^T)^T
        let residual = y.transpose() - &ax;
        let tmp = linalg::matmul(&residual, &gamma);
        self.a += linalg::matmul(&tmp, &pxt);

        // P = (P - PX @ Gamma @ PX^T) / w
        let tmp2 = linalg::matmul(&px, &gamma);
        self.p = (&self.p - linalg::matmul(&tmp2, &pxt)) / self.w;

        // Symmetrize P
        linalg::symmetrize(&mut self.p);

        // Check A_allclose
        if !self.a_allclose() {
            self.a_last = self.a.clone();
        }
    }

    /// Update with a single (x, y) pair. y=None means unsupervised.
    pub fn update(&mut self, x: &[f64], y: Option<&[f64]>) {
        let m = x.len();

        // Handle unsupervised mode
        let (x_use, y_use) = if let Some(y_data) = y {
            (x.to_vec(), y_data.to_vec())
        } else {
            if self.x_last.is_none() {
                self.x_last = Some(x.to_vec());
                return;
            }
            let prev = self.x_last.take().unwrap();
            self.x_last = Some(x.to_vec());
            (prev, x.to_vec())
        };

        if self.n_seen == 0 {
            self.m = m;
            self.init_update();
        }

        // Initialize with buffer
        if self.initialize > 0 && self.n_seen < self.initialize {
            if let Some(ref mut x_buf) = self.x_init {
                for (j, &val) in x_use.iter().enumerate() {
                    x_buf[(self.n_seen, j)] = val;
                }
            }
            if let Some(ref mut y_buf) = self.y_init {
                for (j, &val) in y_use.iter().enumerate() {
                    y_buf[(self.n_seen, j)] = val;
                }
            }
            if self.n_seen == self.initialize - 1 {
                let x_buf = self.x_init.take().unwrap();
                let y_buf = self.y_init.take().unwrap();
                self.learn_many(&x_buf, Some(&y_buf));
                self.n_seen -= x_buf.nrows();
            }
        } else {
            // Incremental update
            if self.n_seen == 0 && !self.p_inited {
                let epsilon = 1e-15;
                let alpha = 1.0 / epsilon;
                self.p = DMatrix::identity(self.r, self.r) * alpha;
                self.p_inited = true;
            }

            let x_mat = DMatrix::from_row_slice(1, x_use.len(), &x_use);
            let y_mat = DMatrix::from_row_slice(1, y_use.len(), &y_use);

            let (x_proj, y_proj) = if self.r < self.m {
                self.truncate_w_svd(&x_mat, &y_mat, SvdModify::Update)
            } else {
                (x_mat, y_mat)
            };

            let w = DMatrix::from_element(1, 1, 1.0);
            self.update_a_p(&x_proj, &y_proj, &w);
        }

        self.n_seen += 1;
    }

    /// Revert a (x, y) pair.
    pub fn revert(&mut self, x: &[f64], y: Option<&[f64]>) {
        let (x_use, y_use) = if let Some(y_data) = y {
            (x.to_vec(), y_data.to_vec())
        } else {
            if self.x_first.is_none() {
                self.x_first = Some(x.to_vec());
                return;
            }
            let prev = self.x_first.take().unwrap();
            self.x_first = Some(x.to_vec());
            (prev, x.to_vec())
        };

        let x_mat = DMatrix::from_row_slice(1, x_use.len(), &x_use);
        let y_mat = DMatrix::from_row_slice(1, y_use.len(), &y_use);

        let (x_proj, y_proj) = if self.r < self.m {
            self.truncate_w_svd(&x_mat, &y_mat, SvdModify::Revert)
        } else {
            (x_mat, y_mat)
        };

        let weight = if self.exponential_weighting {
            1.0 / -(self.w.powi(self.n_seen as i32))
        } else {
            -1.0
        };

        let w = DMatrix::from_element(1, 1, weight);
        self.update_a_p(&x_proj, &y_proj, &w);

        self.n_seen -= 1;
    }

    /// Batch initialization.
    pub fn learn_many(&mut self, x: &DMatrix<f64>, y: Option<&DMatrix<f64>>) {
        let (x_data, y_data) = if let Some(y_mat) = y {
            (x.clone(), y_mat.clone())
        } else {
            // Unsupervised: y = x shifted by 1
            let n = x.nrows();
            if n < 2 {
                return;
            }
            let x_part = x.rows(0, n - 1).clone_owned();
            let y_part = x.rows(1, n - 1).clone_owned();
            (x_part, y_part)
        };

        let n = x_data.nrows();
        self.m = x_data.ncols();

        if self.r == 0 {
            self.r = self.m;
        }

        // Exponential weighting
        let weights: Vec<f64> = if self.exponential_weighting {
            (0..n)
                .map(|i| self.w.sqrt().powi((n - 1 - i) as i32))
                .collect()
        } else {
            vec![1.0; n]
        };

        let mut xq = x_data.clone();
        let mut yq = y_data.clone();
        for i in 0..n {
            for j in 0..xq.ncols() {
                xq[(i, j)] *= weights[i];
            }
            for j in 0..yq.ncols() {
                yq[(i, j)] *= weights[i];
            }
        }

        self.n_seen += n;

        if !self.p_inited {
            // First initialization
            if self.r < self.m {
                // Learn SVD and extract needed data
                {
                    let svd = self.svd.as_mut().expect("SVD required for r < m");
                    svd.learn_many(&xq);
                }
                // Now borrow immutably to extract data
                let svd = self.svd.as_ref().expect("SVD required for r < m");
                let u = svd.u.clone();
                let s = svd.s.clone();
                let vt = svd.vt.clone();

                let m_feat = yq.ncols();
                let l = self.m - m_feat;

                let uu = if l != 0 {
                    let mut aug = DMatrix::zeros(self.m, self.r);
                    let copy_rows = m_feat.min(u.nrows());
                    aug.view_mut((0, 0), (copy_rows, self.r))
                        .copy_from(&u.rows(0, copy_rows));
                    for i in 0..l.min(self.r) {
                        aug[(m_feat + i, i)] = 1.0;
                    }
                    u.transpose() * &aug
                } else {
                    DMatrix::identity(self.r, self.r)
                };

                // A = U^T[:, :y_cols] @ Y^T @ V^T.T @ diag(1/S) @ UU
                let s_inv =
                    DVector::from_iterator(s.len(), s.iter().map(|&si| 1.0 / si));
                let ut_y = u.transpose().columns(0, yq.ncols()).into_owned()
                    * yq.transpose()
                    * vt.transpose()
                    * DMatrix::from_diagonal(&s_inv);
                self.a = ut_y * &uu;

                // P = inv(U^T @ X^T @ X @ U) / w
                let xx = xq.transpose() * &xq;
                let utxxu = u.transpose() * &xx * &u;
                self.p = linalg::mat_inv(&utxxu) / self.w;
            } else {
                // Exact DMD
                let xqt = xq.transpose();
                let xqt_pinv = linalg::mat_pinv(&xqt);
                self.a = yq.transpose() * &xqt_pinv;

                let xx = &xqt * xq;
                self.p = linalg::mat_inv(&xx) / self.w;
            }

            self.a_last = self.a.clone();
            self.p_inited = true;
            self.inited = true;
            self.initialize = 0;
        } else {
            // Already initialized: batch update
            self._update_many(&xq, &yq);
        }
    }

    /// Batch incremental update (after initialization).
    fn _update_many(&mut self, x: &DMatrix<f64>, y: &DMatrix<f64>) {
        let p_count = x.nrows();
        let weights: Vec<f64> = if self.exponential_weighting {
            (0..p_count)
                .map(|i| self.w.sqrt().powi((p_count - 1 - i) as i32))
                .collect()
        } else {
            vec![1.0; p_count]
        };
        let c_inv = DMatrix::from_diagonal(&DVector::from_iterator(
            p_count,
            weights.iter().map(|w| 1.0 / w),
        ));

        let (x_proj, y_proj) = if self.r < self.m {
            self.truncate_w_svd(x, y, SvdModify::Update)
        } else {
            (x.clone(), y.clone())
        };

        self.update_a_p(&x_proj, &y_proj, &c_inv);
    }

    /// Eigendecomposition of A.
    pub fn eig(&self) -> (Vec<Complex64>, DMatrix<Complex64>) {
        linalg::eig_complex(&self.a)
    }

    /// Compute DMD modes.
    pub fn modes(&self) -> DMatrix<Complex64> {
        let (_, phi) = self.eig();
        if self.r < self.m {
            if let Some(ref svd) = self.svd {
                let s_inv = DVector::from_iterator(
                    svd.s.len(),
                    svd.s.iter().map(|&si| Complex64::new(1.0 / si, 0.0)),
                );
                let u_complex = svd.u.map(|v| Complex64::new(v, 0.0));
                let s_inv_diag = DMatrix::from_diagonal(&s_inv);
                &u_complex * &s_inv_diag * &phi
            } else {
                phi
            }
        } else {
            phi
        }
    }

    /// Check if A has converged.
    pub fn a_allclose(&self) -> bool {
        if let Some(rtol) = self.eig_rtol {
            if self.a.nrows() == 0 || self.a_last.nrows() == 0 {
                return false;
            }
            let a_abs = self.a.map(|v| v.abs());
            let rows = a_abs.nrows().min(self.a_last.nrows());
            let cols = a_abs.ncols().min(self.a_last.ncols());
            let a_last_sub: DMatrix<f64> = self
                .a_last
                .view((0, 0), (rows, cols))
                .clone_owned()
                .map(|v| v.abs());
            let a_sub: DMatrix<f64> = a_abs
                .view((0, 0), (rows, cols))
                .clone_owned();
            linalg::allclose(&a_last_sub, &a_sub, rtol)
        } else {
            false
        }
    }

    /// Transform: x @ modes
    pub fn transform(&self, x: &[f64]) -> Vec<Complex64> {
        let modes = self.modes();
        let n = modes.ncols();
        let mut result = vec![Complex64::new(0.0, 0.0); n];
        for j in 0..n {
            for i in 0..x.len().min(modes.nrows()) {
                result[j] += Complex64::new(x[i], 0.0) * modes[(i, j)];
            }
        }
        result
    }

    /// Get the full A matrix in original space.
    pub fn a_full(&self) -> DMatrix<f64> {
        if self.r < self.m {
            if let Some(ref svd) = self.svd {
                &svd.u * &self.a * svd.u.transpose()
            } else {
                self.a.clone()
            }
        } else {
            self.a.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_online_dmd_basic() {
        let n = 50;
        let freq = 2.0;
        let dt = 0.1;
        let tspan: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();
        let w1: Vec<f64> = tspan.iter().map(|&t| (PI * freq * t).cos()).collect();
        let w2: Vec<f64> = tspan.iter().map(|&t| -(PI * freq * t).sin()).collect();

        let mut dmd = OnlineDmd::new(2, 0.1, 0, false, None);
        for i in 0..n - 1 {
            let x = vec![w1[i], w2[i]];
            let y = vec![w1[i + 1], w2[i + 1]];
            dmd.update(&x, Some(&y));
        }

        assert!(dmd.n_seen > 0);
        let (evals, _) = dmd.eig();
        assert_eq!(evals.len(), 2);
    }

    #[test]
    fn test_online_dmd_unsupervised() {
        let mut dmd = OnlineDmd::new(0, 1.0, 0, false, None);
        for i in 0..20 {
            let x = vec![(i as f64 * 0.1).cos(), (i as f64 * 0.1).sin()];
            dmd.update(&x, None);
        }
        // First sample is stored, so n_seen = 19
        assert_eq!(dmd.n_seen, 19);
    }

    #[test]
    fn test_online_dmd_learn_many() {
        let n = 20;
        let m = 3;
        let mut data = DMatrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                data[(i, j)] = ((i * m + j) as f64).sin();
            }
        }
        let x = data.rows(0, n - 1).clone_owned();
        let y = data.rows(1, n - 1).clone_owned();

        let mut dmd = OnlineDmd::new(2, 1.0, 0, false, None);
        dmd.learn_many(&x, Some(&y));
        assert_eq!(dmd.n_seen, n - 1);
        assert_eq!(dmd.a.nrows(), 2);
    }
}
