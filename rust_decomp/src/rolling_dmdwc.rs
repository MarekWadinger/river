/// RustRollingDMDwC — fused Rolling + OnlineDMDwC in a single #[pyclass].
use std::collections::VecDeque;

use bincode::{deserialize, serialize};
use numpy::PyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::{Deserialize, Serialize};

use crate::online_dmdwc::OnlineDmdwC;
use crate::rolling_dmd::{dmatrix_to_numpy, extract_2d_array, extract_f64_vec};

#[derive(Serialize, Deserialize)]
#[pyclass(module = "river.decomposition._rust_decomp")]
pub struct RustRollingDMDwC {
    dmd: OnlineDmdwC,
    window: VecDeque<(Vec<f64>, Option<Vec<f64>>, Option<Vec<f64>>)>,
    window_size: usize,
    // Constructor args for pickle
    init_p: usize,
    init_q: usize,
    init_w: f64,
    init_initialize: usize,
    init_exponential_weighting: bool,
    init_eig_rtol: Option<f64>,
    init_seed: Option<u64>,
}

#[pymethods]
impl RustRollingDMDwC {
    #[new]
    #[pyo3(signature = (p=0, q=0, w=1.0, window_size=301, initialize=1, exponential_weighting=false, eig_rtol=None, seed=None))]
    fn new(
        p: usize,
        q: usize,
        w: f64,
        window_size: usize,
        initialize: usize,
        exponential_weighting: bool,
        eig_rtol: Option<f64>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if !(0.0 < w && w <= 1.0) {
            return Err(PyValueError::new_err("w must be in (0, 1]"));
        }
        Ok(Self {
            dmd: OnlineDmdwC::new(
                None,
                p,
                q,
                w,
                initialize,
                exponential_weighting,
                eig_rtol,
            ),
            window: VecDeque::with_capacity(window_size),
            window_size,
            init_p: p,
            init_q: q,
            init_w: w,
            init_initialize: initialize,
            init_exponential_weighting: exponential_weighting,
            init_eig_rtol: eig_rtol,
            init_seed: seed,
        })
    }

    #[pyo3(signature = (x, y=None, u=None))]
    fn update(
        &mut self,
        py: Python<'_>,
        x: Bound<'_, pyo3::PyAny>,
        y: Option<Bound<'_, pyo3::PyAny>>,
        u: Option<Bound<'_, pyo3::PyAny>>,
    ) -> PyResult<()> {
        let x_vec = extract_f64_vec(py, &x)?;
        let y_vec = y
            .as_ref()
            .map(|obj| extract_f64_vec(py, obj))
            .transpose()?;
        let u_vec = u
            .as_ref()
            .map(|obj| extract_f64_vec(py, obj))
            .transpose()?;

        // If window is full, revert the oldest
        if self.window.len() >= self.window_size {
            if let Some((old_x, old_y, old_u)) = self.window.pop_front() {
                self.dmd
                    .revert(&old_x, old_y.as_deref(), old_u.as_deref());
            }
        }

        self.dmd
            .update(&x_vec, y_vec.as_deref(), u_vec.as_deref());

        self.window.push_back((x_vec, y_vec, u_vec));

        Ok(())
    }

    #[pyo3(signature = (x, y=None, u=None))]
    fn learn_many(
        &mut self,
        x: Bound<'_, pyo3::PyAny>,
        y: Option<Bound<'_, pyo3::PyAny>>,
        u: Option<Bound<'_, pyo3::PyAny>>,
    ) -> PyResult<()> {
        let x_mat = extract_2d_array(&x)?;
        let y_mat = y.as_ref().map(extract_2d_array).transpose()?;
        let u_mat = u.as_ref().map(extract_2d_array).transpose()?;
        self.dmd
            .learn_many(&x_mat, y_mat.as_ref(), u_mat.as_ref());
        Ok(())
    }

    // ---- Properties ----

    #[getter]
    fn eig<'py>(&self, py: Python<'py>) -> PyResult<(PyObject, PyObject)> {
        let (evals, evecs) = self.dmd.eig();

        let evals_data: Vec<num_complex::Complex64> = evals.clone();
        let evals_arr = numpy::PyArray1::from_vec(py, evals_data);

        let nrows = evecs.nrows();
        let ncols = evecs.ncols();
        let mut evecs_data = Vec::with_capacity(nrows * ncols);
        for i in 0..nrows {
            for j in 0..ncols {
                evecs_data.push(evecs[(i, j)]);
            }
        }
        let evecs_arr = numpy::PyArray2::from_vec2(
            py,
            &evecs_data.chunks(ncols).map(|c| c.to_vec()).collect::<Vec<_>>(),
        )
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;

        Ok((evals_arr.into_any().unbind(), evecs_arr.into_any().unbind()))
    }

    #[getter]
    fn modes<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let m = self.dmd.modes();
        let nrows = m.nrows();
        let ncols = m.ncols();
        let mut data = Vec::with_capacity(nrows * ncols);
        for i in 0..nrows {
            for j in 0..ncols {
                data.push(m[(i, j)]);
            }
        }
        let arr = numpy::PyArray2::from_vec2(
            py,
            &data.chunks(ncols).map(|c| c.to_vec()).collect::<Vec<_>>(),
        )
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        Ok(arr.into_any().unbind())
    }

    #[getter]
    fn a<'py>(&self, py: Python<'py>) -> PyObject {
        dmatrix_to_numpy(py, &self.dmd.inner.a).into_any().unbind()
    }

    #[getter]
    fn b<'py>(&self, py: Python<'py>) -> PyObject {
        dmatrix_to_numpy(py, &self.dmd.b).into_any().unbind()
    }

    #[getter]
    fn a_allclose(&self) -> bool {
        self.dmd.inner.a_allclose()
    }

    #[getter]
    fn n_seen(&self) -> usize {
        self.dmd.inner.n_seen
    }

    #[getter]
    fn r(&self) -> usize {
        self.dmd.inner.r
    }

    #[getter]
    fn m(&self) -> usize {
        self.dmd.inner.m
    }

    #[getter]
    fn get_window_size(&self) -> usize {
        self.window_size
    }

    // ---- Predict ----

    fn predict_one<'py>(
        &self,
        py: Python<'py>,
        x: Bound<'_, pyo3::PyAny>,
        u: Bound<'_, pyo3::PyAny>,
    ) -> PyResult<PyObject> {
        let x_vec = extract_f64_vec(py, &x)?;
        let u_vec = extract_f64_vec(py, &u)?;
        let result = self.dmd.predict_one(&x_vec, &u_vec);
        let arr = numpy::PyArray1::from_vec(py, result);
        Ok(arr.into_any().unbind())
    }

    fn predict_horizon<'py>(
        &self,
        py: Python<'py>,
        x: Bound<'_, pyo3::PyAny>,
        horizon: usize,
        u: Bound<'_, pyo3::PyAny>,
    ) -> PyResult<PyObject> {
        let x_vec = extract_f64_vec(py, &x)?;
        let u_mat = extract_2d_array(&u)?;
        let result = self.dmd.predict_horizon(&x_vec, horizon, &u_mat);
        Ok(dmatrix_to_numpy(py, &result).into_any().unbind())
    }

    // ---- Transform ----

    fn transform_one<'py>(
        &self,
        py: Python<'py>,
        x: Bound<'_, pyo3::PyAny>,
    ) -> PyResult<PyObject> {
        let x_vec = extract_f64_vec(py, &x)?;
        let modes = self.dmd.modes();
        let n = modes.ncols();
        let mut data: Vec<num_complex::Complex64> = Vec::with_capacity(n);
        for j in 0..n {
            let mut sum = num_complex::Complex64::new(0.0, 0.0);
            for i in 0..x_vec.len().min(modes.nrows()) {
                sum += num_complex::Complex64::new(x_vec[i], 0.0) * modes[(i, j)];
            }
            data.push(sum);
        }
        let arr = numpy::PyArray1::from_vec(py, data);
        Ok(arr.into_any().unbind())
    }

    // ---- Pickle ----

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let data = serialize(self)
            .map_err(|e| PyValueError::new_err(format!("Serialization failed: {e}")))?;
        Ok(PyBytes::new(py, &data))
    }

    fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        *self = deserialize(state.as_bytes())
            .map_err(|e| PyValueError::new_err(format!("Deserialization failed: {e}")))?;
        Ok(())
    }

    fn __getnewargs__(
        &self,
    ) -> PyResult<(usize, usize, f64, usize, usize, bool, Option<f64>, Option<u64>)> {
        Ok((
            self.init_p,
            self.init_q,
            self.init_w,
            self.window_size,
            self.init_initialize,
            self.init_exponential_weighting,
            self.init_eig_rtol,
            self.init_seed,
        ))
    }
}
