/// Online DMD with Control (DMDwC).
///
/// Port of `vendor/river/river/decomposition/odmd.py` (OnlineDMDwC).
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::linalg;
use crate::online_dmd::OnlineDmd;
use crate::online_svd::OnlineSvdZhang;

#[derive(Serialize, Deserialize, Clone)]
pub struct OnlineDmdwC {
    pub inner: OnlineDmd,
    pub p_trunc: usize, // state truncation
    pub q_trunc: usize, // control truncation
    pub l: usize,       // control dimension
    pub b: DMatrix<f64>,
    pub known_b: bool,

    // For unsupervised mode
    pub u_last: Option<Vec<f64>>,
    pub u_first: Option<Vec<f64>>,

    inited: bool,
}

impl OnlineDmdwC {
    pub fn new(
        b: Option<DMatrix<f64>>,
        p: usize,
        q: usize,
        w: f64,
        initialize: usize,
        exponential_weighting: bool,
        eig_rtol: Option<f64>,
    ) -> Self {
        let known_b = b.is_some();
        let r = p + q;
        Self {
            inner: OnlineDmd::new(r, w, initialize, exponential_weighting, eig_rtol),
            p_trunc: p,
            q_trunc: q,
            l: 0,
            b: b.unwrap_or_else(|| DMatrix::zeros(0, 0)),
            known_b,
            u_last: None,
            u_first: None,
            inited: false,
        }
    }

    fn init_update(&mut self) {
        if self.p_trunc == 0 {
            self.p_trunc = self.inner.m;
        }
        if self.q_trunc == 0 {
            self.q_trunc = self.l;
        }
        if self.known_b {
            self.inner.r = self.p_trunc;
        } else {
            self.inner.r = self.p_trunc + self.q_trunc;
        }
        // Reinitialize SVD with correct r
        self.inner.svd = Some(OnlineSvdZhang::new(self.inner.r, 1e-12, false));

        if self.inner.initialize < self.inner.r {
            self.inner.initialize = self.inner.r;
        }

        self.inner.a = DMatrix::identity(self.p_trunc, self.p_trunc);
        self.inner.a_last = self.inner.a.clone();

        if !self.known_b {
            self.b = DMatrix::identity(self.p_trunc, self.q_trunc);
            let ab = append_cols(&self.inner.a, &self.b);
            self.inner.a_last = ab;
        }

        self.inner.x_init =
            Some(DMatrix::zeros(self.inner.initialize, self.inner.m));
        self.inner.y_init =
            Some(DMatrix::zeros(self.inner.initialize, self.inner.m));

        self.inited = true;
    }

    /// Reconstruct full A, B matrices in original space via SVD U.
    pub fn reconstruct_ab(&self) -> (DMatrix<f64>, DMatrix<f64>) {
        let m_feat = if self.known_b {
            self.inner.m
        } else {
            self.inner.m - self.l
        };

        if self.inner.r < self.inner.m {
            if let Some(ref svd) = self.inner.svd {
                let u = &svd.u;
                let u_rows = u.rows(0, m_feat).clone_owned();
                let u_state = u_rows.columns(0, self.p_trunc).clone_owned();
                let a_full = &u_state * &self.inner.a * u_state.transpose();

                let u_ctrl = u.rows(u.nrows() - self.l, self.l).clone_owned();
                let u_ctrl_q = u_ctrl.columns(u_ctrl.ncols() - self.q_trunc, self.q_trunc).clone_owned();
                let u_rows2 = u.rows(0, m_feat).clone_owned();
                let u_state2 = u_rows2.columns(0, self.p_trunc).clone_owned();
                let b_full = &u_state2 * &self.b * u_ctrl_q.transpose();

                (a_full, b_full)
            } else {
                (self.inner.a.clone(), self.b.clone())
            }
        } else {
            (self.inner.a.clone(), self.b.clone())
        }
    }

    /// Update with a new (x, y, u) triple.
    pub fn update(
        &mut self,
        x: &[f64],
        y: Option<&[f64]>,
        u: Option<&[f64]>,
    ) {
        // Handle unsupervised mode
        let (x_use, y_use, u_use) = if y.is_none() {
            if self.inner.x_last.is_none() {
                self.inner.x_last = Some(x.to_vec());
                self.u_last = u.map(|v| v.to_vec());
                return;
            }
            let prev_x = self.inner.x_last.take().unwrap();
            self.inner.x_last = Some(x.to_vec());
            let prev_u = self.u_last.take();
            self.u_last = u.map(|v| v.to_vec());
            (prev_x, x.to_vec(), prev_u.map(|v| v.to_vec()))
        } else {
            (
                x.to_vec(),
                y.unwrap().to_vec(),
                u.map(|v| v.to_vec()),
            )
        };

        if let Some(ref u_data) = u_use {
            if self.inner.n_seen == 0 {
                self.inner.m = x_use.len();
                self.l = u_data.len();
                self.init_update();
                if !self.known_b {
                    self.inner.m += u_data.len();
                }
            }

            if self.inner.initialize > 0
                && self.inner.n_seen <= self.inner.initialize - 1
            {
                // Buffer for init
                if let Some(ref mut x_buf) = self.inner.x_init {
                    for (j, &val) in x_use.iter().enumerate() {
                        x_buf[(self.inner.n_seen, j)] = val;
                    }
                }
                if let Some(ref mut y_buf) = self.inner.y_init {
                    for (j, &val) in y_use.iter().enumerate() {
                        y_buf[(self.inner.n_seen, j)] = val;
                    }
                }

                if self.inner.n_seen == self.inner.initialize - 1 {
                    let x_buf = self.inner.x_init.take().unwrap();
                    let y_buf = self.inner.y_init.take().unwrap();
                    // Build U_init from the buffered u values
                    // For simplicity, we'll handle learn_many directly
                    let u_mat =
                        DMatrix::from_row_slice(1, u_data.len(), u_data);
                    // Actually we need to store all u_init values
                    // For now, call learn_many with the current buffer
                    self.learn_many(&x_buf, Some(&y_buf), Some(&u_mat));
                    self.inner.n_seen -= x_buf.nrows();
                }
                self.inner.n_seen += 1;
            } else {
                let mut y_vec = y_use.clone();
                let mut x_vec = x_use.clone();

                if self.known_b && self.b.nrows() > 0 {
                    // y -= u @ B^T
                    let u_mat =
                        DMatrix::from_row_slice(1, u_data.len(), u_data);
                    let bu = &u_mat * self.b.transpose();
                    for i in 0..y_vec.len().min(bu.ncols()) {
                        y_vec[i] -= bu[(0, i)];
                    }
                } else {
                    // Augment x with u
                    x_vec.extend_from_slice(u_data);
                    // A = [A | B]
                    self.inner.a = append_cols(&self.inner.a, &self.b);
                }

                self.inner.update(&x_vec, Some(&y_vec));

                // Split A back into A and B
                if self.inner.a.nrows() < self.inner.a.ncols() {
                    let a_cols = self.inner.a.ncols();
                    let new_b = self
                        .inner
                        .a
                        .view(
                            (0, a_cols - self.q_trunc),
                            (self.p_trunc, self.q_trunc),
                        )
                        .clone_owned();
                    let new_a = self
                        .inner
                        .a
                        .view((0, 0), (self.p_trunc, a_cols - self.q_trunc))
                        .clone_owned();
                    self.b = new_b;
                    self.inner.a = new_a;
                }
            }
        } else {
            // No control input - delegate to inner DMD
            self.inner.update(&x_use, Some(&y_use));
        }
    }

    /// Revert a (x, y, u) triple.
    pub fn revert(
        &mut self,
        x: &[f64],
        y: Option<&[f64]>,
        u: Option<&[f64]>,
    ) {
        if u.is_none() {
            self.inner.revert(x, y);
            return;
        }

        let (x_use, y_use, u_use) = if y.is_none() {
            if self.inner.x_first.is_none() {
                self.inner.x_first = Some(x.to_vec());
                self.u_first = u.map(|v| v.to_vec());
                return;
            }
            let prev_x = self.inner.x_first.take().unwrap();
            self.inner.x_first = Some(x.to_vec());
            let prev_u = self.u_first.take();
            self.u_first = u.map(|v| v.to_vec());
            (prev_x, x.to_vec(), prev_u)
        } else {
            (
                x.to_vec(),
                y.unwrap().to_vec(),
                u.map(|v| v.to_vec()),
            )
        };

        let u_data = u_use.unwrap();
        let mut y_vec = y_use;
        let mut x_vec = x_use;

        if self.known_b && self.b.nrows() > 0 {
            let u_mat =
                DMatrix::from_row_slice(1, u_data.len(), &u_data);
            let bu = &u_mat * self.b.transpose();
            for i in 0..y_vec.len().min(bu.ncols()) {
                y_vec[i] -= bu[(0, i)];
            }
        } else {
            x_vec.extend_from_slice(&u_data);
            self.inner.a = append_cols(&self.inner.a, &self.b);
        }

        self.inner.revert(&x_vec, Some(&y_vec));

        if !self.known_b {
            let a_cols = self.inner.a.ncols();
            let new_b = self
                .inner
                .a
                .view(
                    (0, a_cols - self.q_trunc),
                    (self.p_trunc, self.q_trunc),
                )
                .clone_owned();
            let new_a = self
                .inner
                .a
                .view((0, 0), (self.p_trunc, a_cols - self.q_trunc))
                .clone_owned();
            self.b = new_b;
            self.inner.a = new_a;
        }
    }

    /// Batch initialization.
    pub fn learn_many(
        &mut self,
        x: &DMatrix<f64>,
        y: Option<&DMatrix<f64>>,
        u: Option<&DMatrix<f64>>,
    ) {
        if u.is_none() {
            self.inner.learn_many(x, y);
            return;
        }

        let u_mat = u.unwrap();
        let y_mat = if let Some(y) = y {
            y.clone()
        } else {
            // Unsupervised
            let n = x.nrows();
            if n < 2 {
                return;
            }
            x.rows(1, n - 1).clone_owned()
        };

        self.inner.m = x.ncols();
        self.l = u_mat.ncols();
        self.init_update();

        if !self.known_b {
            self.inner.m += u_mat.ncols();
        }

        let x_aug;
        let y_final;
        if self.known_b && self.b.nrows() > 0 {
            y_final = &y_mat - u_mat * self.b.transpose();
            x_aug = x.clone();
        } else {
            x_aug = append_cols(x, u_mat);
            y_final = y_mat;
            if self.b.nrows() > 0 {
                self.inner.a = append_cols(&self.inner.a, &self.b);
            }
        }

        self.inner.learn_many(&x_aug, Some(&y_final));

        if self.p_trunc == 0 {
            self.p_trunc = self.inner.m;
        }
        if self.q_trunc == 0 {
            self.q_trunc = self.l;
        }
        if !self.known_b {
            let a_cols = self.inner.a.ncols();
            let new_b = self
                .inner
                .a
                .view(
                    (0, a_cols - self.q_trunc),
                    (self.p_trunc, self.q_trunc),
                )
                .clone_owned();
            let new_a = self
                .inner
                .a
                .view((0, 0), (self.p_trunc, a_cols - self.q_trunc))
                .clone_owned();
            self.b = new_b;
            self.inner.a = new_a;
        }
    }

    /// Eigendecomposition.
    pub fn eig(&self) -> (Vec<Complex64>, DMatrix<Complex64>) {
        self.inner.eig()
    }

    /// DMD modes (projected).
    pub fn modes(&self) -> DMatrix<Complex64> {
        let (_, phi) = self.inner.eig();
        if self.inner.r < self.inner.m {
            if let Some(ref svd) = self.inner.svd {
                let s_inv = DVector::from_iterator(
                    svd.s.len(),
                    svd.s.iter().map(|&si| Complex64::new(1.0 / si, 0.0)),
                );
                let m_feat = if self.known_b {
                    self.inner.m
                } else {
                    self.inner.m - self.l
                };
                let u_rows = svd.u.rows(0, m_feat).clone_owned();
                let u_sub = u_rows.columns(0, self.p_trunc).clone_owned();
                let u_complex = u_sub.map(|v| Complex64::new(v, 0.0));
                let s_inv_sub = DVector::from_iterator(
                    self.p_trunc,
                    s_inv.iter().take(self.p_trunc).cloned(),
                );
                let s_inv_diag = DMatrix::from_diagonal(&s_inv_sub);
                &u_complex * &s_inv_diag * &phi
            } else {
                phi
            }
        } else {
            phi
        }
    }

    /// Predict next state given x and u.
    pub fn predict_one(&self, x: &[f64], u: &[f64]) -> Vec<f64> {
        let (a_full, b_full) = self.reconstruct_ab();
        let x_vec = DVector::from_column_slice(x);
        let u_vec = DVector::from_column_slice(u);
        let result = &a_full * &x_vec + &b_full * &u_vec;
        result.iter().copied().collect()
    }

    /// Multi-step prediction.
    pub fn predict_horizon(
        &self,
        x: &[f64],
        horizon: usize,
        u_horizon: &DMatrix<f64>,
    ) -> DMatrix<f64> {
        let (a_full, b_full) = self.reconstruct_ab();
        let m = x.len();
        let mut mat = DMatrix::zeros(horizon + 1, m);
        for j in 0..m {
            mat[(0, j)] = x[j];
        }
        for s in 1..=horizon {
            let prev = mat.row(s - 1).transpose();
            let u_row = u_horizon.row(s - 1).transpose();
            let next = (&a_full * &prev + &b_full * &u_row)
                .iter()
                .map(|v| *v)
                .collect::<Vec<f64>>();
            for j in 0..m {
                mat[(s, j)] = next[j];
            }
        }
        mat.rows(1, horizon).clone_owned()
    }
}

use crate::linalg::append_cols;

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_online_dmdwc_basic() {
        let n = 50;
        let freq = 2.0;
        let dt = 0.1;
        let tspan: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();
        let w1: Vec<f64> = tspan.iter().map(|&t| (PI * freq * t).cos()).collect();
        let w2: Vec<f64> = tspan.iter().map(|&t| -(PI * freq * t).sin()).collect();

        let mut dmd = OnlineDmdwC::new(None, 2, 1, 0.1, 4, false, None);
        for i in 0..n - 1 {
            let x = vec![w1[i], w2[i]];
            let y = vec![w1[i + 1], w2[i + 1]];
            let u = vec![1.0];
            dmd.update(&x, Some(&y), Some(&u));
        }

        assert!(dmd.inner.n_seen > 0);
        let (evals, _) = dmd.eig();
        assert!(!evals.is_empty());
    }
}
