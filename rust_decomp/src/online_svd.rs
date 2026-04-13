/// Online SVD using the Zhang (2022) algorithm.
///
/// Port of `vendor/river/river/decomposition/osvd.py` (OnlineSVDZhang).
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::linalg;
use crate::linalg::{append_cols, stack_rows};

#[derive(Serialize, Deserialize, Clone)]
pub struct OnlineSvdZhang {
    pub n_components: usize,
    pub tol: f64,
    pub rank_updates: bool,

    // State
    pub u: DMatrix<f64>,  // m x r
    pub s: DVector<f64>,  // r
    pub vt: DMatrix<f64>, // r x n

    pub v_buff: DMatrix<f64>, // r x n_incr
    pub u0: DMatrix<f64>,     // r x r (or r+c x r+c after rank increase)
    pub q_u: usize,
    pub q_r: usize,
    pub n_seen: usize,
    pub m: usize, // feature dimension

    // W is identity in current implementation
    initialized: bool,
}

impl OnlineSvdZhang {
    pub fn new(n_components: usize, tol: f64, rank_updates: bool) -> Self {
        Self {
            n_components,
            tol,
            rank_updates,
            u: DMatrix::zeros(0, 0),
            s: DVector::zeros(0),
            vt: DMatrix::zeros(0, 0),
            v_buff: DMatrix::zeros(0, 0),
            u0: DMatrix::zeros(0, 0),
            q_u: 0,
            q_r: 0,
            n_seen: 0,
            m: 0,
            initialized: false,
        }
    }

    /// Construct from existing state (equivalent to _from_state).
    pub fn from_state(
        u: DMatrix<f64>,
        s: DVector<f64>,
        vt: DMatrix<f64>,
        rank_updates: bool,
    ) -> Self {
        let n_components = s.len();
        let m = u.nrows();
        Self {
            n_components,
            tol: 1e-12,
            rank_updates,
            v_buff: DMatrix::zeros(n_components, 0),
            u0: DMatrix::identity(n_components, n_components),
            q_u: 0,
            q_r: 0,
            n_seen: vt.ncols(),
            m,
            u,
            s,
            vt,
            initialized: true,
        }
    }

    /// Initialize on first data pass.
    fn init_first_pass(&mut self, nfeatures: usize) {
        self.m = nfeatures;
        if self.n_components == 0 {
            self.n_components = nfeatures;
        }
        self.v_buff = DMatrix::zeros(self.n_components, 0);
        self.u0 = DMatrix::identity(self.n_components, self.n_components);
        self.initialized = true;
    }

    /// Batch initialization via full SVD.
    pub fn learn_many(&mut self, x: &DMatrix<f64>) {
        // x is (n_samples x m)
        let m = x.ncols();
        if !self.initialized {
            self.init_first_pass(m);
        }

        if self.s.len() > 0 {
            // Already initialized: do iterative updates in chunks
            let chunk_size = m;
            let nrows = x.nrows();
            let mut start = 0;
            while start < nrows {
                let end = (start + chunk_size).min(nrows);
                let chunk = x.rows(start, end - start).clone_owned();
                self.update_mat(&chunk);
                start = end;
            }
        } else {
            // First init: full SVD of X^T
            let xt = x.transpose(); // m x n
            let (u, s, vt) = linalg::svd_dispatch(&xt, self.n_components);
            self.u = u;
            self.s = s;
            self.vt = vt;
            self.n_seen = x.nrows();
            self.n_components = self.s.len();
            self.v_buff = DMatrix::zeros(self.n_components, 0);
            self.u0 = DMatrix::identity(self.n_components, self.n_components);
        }
    }

    /// Update with a single sample (1D slice).
    pub fn update(&mut self, x: &[f64]) {
        let nf = x.len();
        let mat = DMatrix::from_row_slice(1, nf, x);
        self.update_mat(&mat);
    }

    /// Update with a matrix of samples (n_samples x m).
    pub fn update_mat(&mut self, x: &DMatrix<f64>) {
        let c = x.nrows();
        let nf = x.ncols();

        if !self.initialized {
            self.init_first_pass(nf);
        }

        // If SVD not yet computed, we can't do incremental updates.
        // Accumulate and do batch init.
        if self.s.len() == 0 {
            // Need batch init first
            self.learn_many(x);
            return;
        }

        let r = self.n_components;
        let a = x.transpose(); // m x c

        // Step 1: M = U^T @ A (W = I) — BLAS for large matrices
        let m_mat = linalg::matmul_at_b(&self.u, &a); // r x c

        // Step 2: P = A - U @ M
        let um = linalg::matmul(&self.u, &m_mat); // m x c
        let p = &a - &um; // m x c

        // PtP = P^T @ P (W = I)
        let ptp = linalg::matmul_at_b(&p, &p); // c x c
        let ptp_cond = ptp.iter().any(|&v| v < 0.0);

        let ra: DMatrix<f64>;
        let po: DMatrix<f64>;
        let po_from_qr: bool;

        if ptp_cond {
            // QR path for numerical stability
            let q = linalg::qr_q(&p);
            let pot = q.transpose();
            ra = &pot * &p;
            po = q;
            po_from_qr = true;
        } else {
            // Sqrt path
            ra = ptp.map(|v| v.sqrt());
            po = DMatrix::zeros(0, 0); // placeholder, computed later if needed
            po_from_qr = false;
        }

        // Step 2: Check tolerance
        if ra.iter().all(|&v| v.abs() < self.tol) {
            // Buffer the update
            self.q_u += c;
            self.v_buff = append_cols(&self.v_buff, &m_mat);
        } else {
            let mut _s = self.s.clone();
            let mut _v = self.vt.transpose(); // n x r
            let mut m_curr = m_mat.clone();

            // Step 7-9: If we have buffered updates, process them first
            if self.q_u > 0 {
                // Construct Y = [diag(S), V_buff]
                let diag_s = DMatrix::from_diagonal(&_s);
                let y = append_cols(&diag_s, &self.v_buff); // r x (r + n_incr)

                // SVD of Y
                let (uy, sy, vyt) = linalg::svd_full(&y);
                let vy = vyt.transpose();

                // Update U0, S, V
                self.u0 = linalg::matmul(&self.u0, &uy);
                _s = sy;
                let v1_block = vy.rows(0, r).clone_owned();
                let v1 = v1_block.columns(0, vy.ncols() - 1).clone_owned();
                let v2_block = vy.rows(r, 1).clone_owned();
                let v2_row = v2_block.columns(0, vy.ncols() - 1).clone_owned();
                // _V = vstack(_V @ V1, V2)
                let new_v_top = linalg::matmul(&_v, &v1);
                _v = stack_rows(&new_v_top, &v2_row);

                // Update M with UY^T
                m_curr = linalg::matmul_at_b(&uy, &m_curr);
            }

            // Step 13: Normalize e
            let po_final: DMatrix<f64>;
            if !po_from_qr {
                let ra_inv = linalg::mat_inv(&ra);
                let po_unnorm = &p * &ra_inv; // m x c
                // Step 14: Reorthogonalize
                let u_col0 = self.u.column(0);
                let pot_wu0 = po_unnorm.transpose() * &u_col0; // c x 1
                if pot_wu0.iter().any(|v| v.abs() > self.tol) {
                    // Reorthogonalize
                    let ut_po = linalg::matmul_at_b(&self.u, &po_unnorm);
                    let mut po_reorth = &po_unnorm - linalg::matmul(&self.u, &ut_po);
                    po_final = linalg::qr_q(&mut po_reorth);
                } else {
                    po_final = po_unnorm;
                }
            } else {
                // Step 14: Reorthogonalize if needed
                let pot = po.transpose();
                let u_col0 = self.u.column(0);
                let check = &pot * &u_col0;
                if check.iter().any(|v| v.abs() > self.tol) {
                    let ut_po2 = linalg::matmul_at_b(&self.u, &po);
                    let mut po_reorth = &po - linalg::matmul(&self.u, &ut_po2);
                    po_final = linalg::qr_q(&mut po_reorth);
                } else {
                    po_final = po;
                }
            }

            // Step 17: Construct Y = [[diag(S), M], [zeros, Ra]]
            let diag_s = DMatrix::from_diagonal(&_s);
            let zeros_cr = DMatrix::zeros(c, r);
            let y = linalg::block2x2(&diag_s, &m_curr, &zeros_cr, &ra);

            // Full SVD of Y
            let (uy, sy, vyt) = linalg::svd_full(&y);
            let vy = vyt.transpose();

            // Step 20: Update U0
            let u0_rows = self.u0.nrows();
            let u0_cols = self.u0.ncols();
            let u0_expanded = linalg::block2x2(
                &self.u0,
                &DMatrix::zeros(u0_rows, c),
                &DMatrix::zeros(c, u0_cols),
                &DMatrix::identity(c, c),
            );
            self.u0 = linalg::matmul(&u0_expanded, &uy);

            // _Ue = hstack(_U, Po)
            let ue = append_cols(&self.u, &po_final);

            if self.rank_updates && r < sy.len() && sy[r] > self.tol {
                // Rank increasing update
                let new_u = linalg::matmul(&ue, &self.u0);
                let new_s = sy;

                let v1 = vy.rows(0, r).clone_owned();
                let v2_row = vy.rows(r, 1).clone_owned();
                let new_v = stack_rows(&linalg::matmul(&_v, &v1), &v2_row);

                self.u = new_u;
                self.s = new_s;
                self.vt = new_v.transpose();
                self.n_components = self.s.len();
                self.u0 = DMatrix::identity(self.n_components, self.n_components);
            } else {
                // Non rank-increasing update
                let u0_sub = self.u0.columns(0, r).clone_owned();
                let new_u = linalg::matmul(&ue, &u0_sub);
                let new_s = DVector::from_iterator(r, sy.iter().take(r).cloned());

                let v_1pad = vy.ncols().saturating_sub(_v.ncols());
                let v_expanded = linalg::block2x2(
                    &_v,
                    &DMatrix::zeros(_v.nrows(), v_1pad),
                    &DMatrix::zeros(c, _v.ncols()),
                    &DMatrix::identity(c, v_1pad),
                );
                let vy_sub = vy.columns(0, r).clone_owned();
                let new_v = linalg::matmul(&v_expanded, &vy_sub);

                self.u = new_u;
                self.s = new_s;
                self.vt = new_v.transpose();
                self.u0 = DMatrix::identity(r, r);
            }

            // Alg. 11: Catch up buffered updates
            if self.q_u > 0 && self.v_buff.ncols() > 0 {
                let r = self.s.len();
                let diag_s = DMatrix::from_diagonal(&self.s);
                let y = append_cols(&diag_s, &self.v_buff);
                let (uy, sy, vyt) = linalg::svd_full(&y);
                let vy = vyt.transpose();
                self.u = linalg::matmul(&self.u, &uy);
                let v1 = vy.rows(0, r).clone_owned();
                let v2 = vy
                    .rows(r, (self.q_u + c - 1).min(vy.nrows() - r))
                    .clone_owned();
                let v_old = self.vt.transpose();
                let new_v = stack_rows(&linalg::matmul(&v_old, &v1), &v2);
                self.s = sy;
                self.vt = new_v.transpose();
            }

            self.n_components = self.s.len();
            self.v_buff = DMatrix::zeros(self.n_components, 0);
            self.q_u = 0;
        }

        self.n_seen += c;
    }

    /// Revert / downdate a sample.
    pub fn revert(&mut self, _x: &[f64], idx: i64) {
        let c: usize = 1;
        let nc = self.vt.ncols();
        let r = self.n_components;
        let _v = self.vt.transpose(); // n x r

        // Get N (the V column(s) corresponding to the sample being reverted)
        // N is r x c (columns of Vt)
        let n_col = if idx >= 0 {
            self.vt.columns(idx as usize, c).clone_owned()
        } else if idx == -1 {
            self.vt.columns(nc - c, c).clone_owned()
        } else {
            let start = ((nc as i64) - (c as i64) + idx + 1) as usize;
            self.vt.columns(start, c).clone_owned()
        };

        // B = zeros(nc, c); B[-c:] = I_c
        let mut b = DMatrix::zeros(nc, c);
        for i in 0..c {
            b[(nc - c + i, i)] = 1.0;
        }

        // Q = B - V @ N
        let q = &b - linalg::matmul(&_v, &n_col); // nc x c

        let qtq = q.transpose() * &q;
        let qtq_cond = qtq.iter().any(|&v| v < 0.0);

        let qot: DMatrix<f64>;
        let ra: DMatrix<f64>;
        if qtq_cond {
            let q_factor = linalg::qr_q(&q);
            qot = q_factor.transpose();
            ra = &qot * &q;
        } else {
            ra = qtq.map(|v| v.sqrt());
            qot = DMatrix::zeros(0, 0); // computed later
        }

        if ra.iter().all(|v| v.abs() < self.tol) {
            self.q_r += c;
        } else {
            let mut effective_c = c;
            let effective_n: DMatrix<f64>;
            let effective_qot: DMatrix<f64>;

            if self.q_r > 0 {
                effective_c += self.q_r;
                let mut b2 = DMatrix::zeros(nc, effective_c);
                for i in 0..effective_c {
                    b2[(nc - effective_c + i, i)] = 1.0;
                }
                let n2 = if idx >= 0 {
                    self.vt
                        .columns(idx as usize, effective_c)
                        .clone_owned()
                } else if idx == -1 {
                    self.vt
                        .columns(nc - effective_c, effective_c)
                        .clone_owned()
                } else {
                    let start =
                        ((nc as i64) - (effective_c as i64) + idx + 1) as usize;
                    self.vt.columns(start, effective_c).clone_owned()
                };
                let q2 = &b2 - linalg::matmul(&_v, &n2);
                effective_qot = linalg::qr_q(&q2).transpose();
                effective_n = n2;
            } else {
                effective_n = n_col;
                if qtq_cond {
                    effective_qot = qot;
                } else {
                    effective_qot = linalg::qr_q(&q).transpose();
                }
            }

            // S_ = pad(diag(S), (0, c), (0, c))
            let s_diag = DMatrix::from_diagonal(&self.s);
            let mut s_padded = DMatrix::zeros(r + effective_c, r + effective_c);
            s_padded
                .view_mut((0, 0), (r, r))
                .copy_from(&s_diag);

            // NtN = N^T @ N
            let ntn = effective_n.transpose() * &effective_n;
            // norm_n = sqrt(1 - NtN), NaN -> 0
            let norm_n = ntn.map(|v| {
                let val = 1.0 - v;
                if val < 0.0 {
                    0.0
                } else {
                    val.sqrt()
                }
            });

            // K = S_ @ (I - [N; 0] @ [N; norm_n]^T)
            let mut n_top =
                DMatrix::zeros(r + effective_c, effective_c);
            n_top
                .view_mut((0, 0), (r, effective_c))
                .copy_from(&effective_n);
            let mut n_bot =
                DMatrix::zeros(r + effective_c, effective_c);
            n_bot
                .view_mut((0, 0), (r, effective_c))
                .copy_from(&effective_n);
            n_bot
                .view_mut((r, 0), (effective_c, effective_c))
                .copy_from(&norm_n);

            let eye =
                DMatrix::identity(r + effective_c, r + effective_c);
            let nn_t = linalg::matmul(&n_top, &n_bot.transpose());
            let k = linalg::matmul(&s_padded, &(&eye - &nn_t));

            // SVD of K
            let (u_, s_, vt_) = linalg::svd_full(&k);

            let new_r = if self.rank_updates && s_.len() > 0 && s_[s_.len() - 1] <= self.tol {
                self.n_components - 1
            } else {
                self.n_components
            };

            // U_ = old_U @ U_[:n_components, :n_components]
            let u_block = u_
                .view((0, 0), (self.n_components, new_r))
                .clone_owned();
            let new_u = linalg::matmul(&self.u, &u_block);

            let new_s =
                DVector::from_iterator(new_r, s_.iter().take(new_r).cloned());

            // Vt_ = Vt_[:new_r, :] @ [Vt; Qot][:, :-c]
            let vt_block = vt_.rows(0, new_r).clone_owned();
            let vt_qot = stack_rows(&self.vt, &effective_qot);
            let vt_trimmed = vt_qot.columns(0, nc - effective_c).clone_owned();
            let new_vt = linalg::matmul(&vt_block, &vt_trimmed);

            self.n_components = new_r;
            self.q_r = 0;
            self.u = new_u;
            self.s = new_s;
            self.vt = new_vt;
        }

        self.n_seen -= c;
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_online_svd_learn_many() {
        let data = DMatrix::from_row_slice(
            6,
            3,
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
            ],
        );
        let mut svd = OnlineSvdZhang::new(2, 1e-12, false);
        svd.learn_many(&data);
        assert_eq!(svd.u.nrows(), 3);
        assert_eq!(svd.u.ncols(), 2);
        assert_eq!(svd.s.len(), 2);
        assert_eq!(svd.vt.nrows(), 2);
        assert_eq!(svd.vt.ncols(), 6);
    }

    #[test]
    fn test_online_svd_update() {
        // Initialize with batch
        let data = DMatrix::from_row_slice(
            4,
            3,
            &[
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                1.0,
            ],
        );
        let mut svd = OnlineSvdZhang::new(2, 1e-12, false);
        svd.learn_many(&data);
        assert_eq!(svd.n_seen, 4);

        // Update with one more sample
        svd.update(&[0.5, 0.5, 0.5]);
        assert_eq!(svd.n_seen, 5);
    }

    #[test]
    fn test_online_svd_revert() {
        let data = DMatrix::from_row_slice(
            4,
            3,
            &[
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                1.0,
            ],
        );
        let mut svd = OnlineSvdZhang::new(2, 1e-12, false);
        svd.learn_many(&data);

        svd.update(&[0.5, 0.5, 0.5]);
        assert_eq!(svd.n_seen, 5);

        svd.revert(&[0.5, 0.5, 0.5], -1);
        assert_eq!(svd.n_seen, 4);
    }
}
