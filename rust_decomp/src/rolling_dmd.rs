/// RustRollingDMD — fused Rolling + OnlineDMD in a single #[pyclass].
use std::collections::VecDeque;

use bincode::{deserialize, serialize};
use nalgebra::DMatrix;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use serde::{Deserialize, Serialize};

use crate::online_dmd::OnlineDmd;

/// Extract a Vec<f64> from a Python object that might be a dict or numpy ndarray.
pub fn extract_f64_vec(py: Python<'_>, obj: &Bound<'_, pyo3::PyAny>) -> PyResult<Vec<f64>> {
    // Fast path: dict
    if let Ok(dict) = obj.downcast::<PyDict>() {
        let mut vals = Vec::with_capacity(dict.len());
        for item in dict.values() {
            vals.push(item.extract::<f64>()?);
        }
        return Ok(vals);
    }
    // Fast path: numpy 1D array — zero-copy read
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<f64>>() {
        return Ok(arr.as_slice().map_err(|e| PyValueError::new_err(format!("{e}")))?.to_vec());
    }
    // Fast path: numpy 2D array (1, m) — zero-copy read
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<f64>>() {
        let slice = arr.as_slice().map_err(|e| PyValueError::new_err(format!("{e}")))?;
        return Ok(slice.to_vec());
    }
    // Fallback: try as flat list/tuple
    if let Ok(list) = obj.extract::<Vec<f64>>() {
        return Ok(list);
    }
    Err(PyValueError::new_err("Expected dict, numpy array, or list of floats"))
}

/// Extract a 2D numpy array -> DMatrix.
/// numpy is row-major (C-order); nalgebra is column-major. Uses
/// `DMatrix::from_row_slice` for a single bulk copy + transpose, avoiding
/// the per-element ndarray indexing that dominated the previous impl.
pub fn extract_2d_array(obj: &Bound<'_, pyo3::PyAny>) -> PyResult<DMatrix<f64>> {
    let arr: PyReadonlyArray2<f64> = obj.extract()?;
    let view = arr.as_array();
    let nrows = view.nrows();
    let ncols = view.ncols();
    if let Some(slice) = view.as_slice() {
        // Contiguous C-order: bulk copy with transpose handled by nalgebra.
        Ok(DMatrix::from_row_slice(nrows, ncols, slice))
    } else {
        // Non-contiguous (e.g., a transposed/sliced view): collect row-major.
        let data: Vec<f64> = view.iter().copied().collect();
        Ok(DMatrix::from_row_slice(nrows, ncols, &data))
    }
}

/// Convert DMatrix to numpy 2D array.
/// nalgebra column-major → numpy row-major: write directly into a freshly
/// allocated PyArray2 buffer, transposing during the single-pass copy.
pub fn dmatrix_to_numpy<'py>(py: Python<'py>, mat: &DMatrix<f64>) -> Bound<'py, PyArray2<f64>> {
    let nrows = mat.nrows();
    let ncols = mat.ncols();
    // SAFETY: PyArray2::new returns an uninitialized array; we fill every
    // element below before any Python code can observe it.
    let arr = unsafe { PyArray2::<f64>::new(py, [nrows, ncols], false) };
    let col_slice = mat.as_slice(); // column-major
    unsafe {
        let dst = arr.as_slice_mut().unwrap();
        for i in 0..nrows {
            let row_off = i * ncols;
            for j in 0..ncols {
                // dst[i,j] (row-major) = col_slice[i + j*nrows] (col-major)
                *dst.get_unchecked_mut(row_off + j) =
                    *col_slice.get_unchecked(i + j * nrows);
            }
        }
    }
    arr
}

#[derive(Serialize, Deserialize)]
#[pyclass(module = "river.decomposition._rust_decomp")]
pub struct RustRollingDMD {
    dmd: OnlineDmd,
    window: VecDeque<(Vec<f64>, Option<Vec<f64>>)>,
    window_size: usize,
    // Store constructor args for pickle
    init_r: usize,
    init_w: f64,
    init_initialize: usize,
    init_exponential_weighting: bool,
    init_eig_rtol: Option<f64>,
    init_seed: Option<u64>,
}

#[pymethods]
impl RustRollingDMD {
    #[new]
    #[pyo3(signature = (r=0, w=1.0, window_size=301, initialize=1, exponential_weighting=false, eig_rtol=None, seed=None))]
    fn new(
        r: usize,
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
            dmd: OnlineDmd::new(r, w, initialize, exponential_weighting, eig_rtol),
            window: VecDeque::with_capacity(window_size),
            window_size,
            init_r: r,
            init_w: w,
            init_initialize: initialize,
            init_exponential_weighting: exponential_weighting,
            init_eig_rtol: eig_rtol,
            init_seed: seed,
        })
    }

    /// Update with a new sample. x can be dict or numpy array.
    #[pyo3(signature = (x, y=None))]
    fn update(
        &mut self,
        py: Python<'_>,
        x: Bound<'_, pyo3::PyAny>,
        y: Option<Bound<'_, pyo3::PyAny>>,
    ) -> PyResult<()> {
        let x_vec = extract_f64_vec(py, &x)?;
        let y_vec = if let Some(ref y_obj) = y {
            Some(extract_f64_vec(py, y_obj)?)
        } else {
            None
        };

        // If window is full, revert the oldest
        if self.window.len() >= self.window_size {
            if let Some((old_x, old_y)) = self.window.pop_front() {
                self.dmd.revert(&old_x, old_y.as_deref());
            }
        }

        // Update DMD
        self.dmd.update(&x_vec, y_vec.as_deref());

        // Push to window
        self.window.push_back((x_vec, y_vec));

        Ok(())
    }

    /// Batch initialization.
    #[pyo3(signature = (x, y=None))]
    fn learn_many(
        &mut self,
        x: Bound<'_, pyo3::PyAny>,
        y: Option<Bound<'_, pyo3::PyAny>>,
    ) -> PyResult<()> {
        let x_mat = extract_2d_array(&x)?;
        let y_mat = y.as_ref().map(extract_2d_array).transpose()?;
        self.dmd.learn_many(&x_mat, y_mat.as_ref());
        Ok(())
    }

    // ---- Properties ----

    /// Returns (eigenvalues, eigenvectors) as numpy arrays.
    #[getter]
    fn eig<'py>(&self, py: Python<'py>) -> PyResult<(PyObject, PyObject)> {
        let (evals, evecs) = self.dmd.eig();
        let n = evals.len();

        // Eigenvalues as complex numpy array
        let evals_data: Vec<num_complex::Complex64> = evals.clone();
        let evals_arr = numpy::PyArray1::from_vec(py, evals_data);

        // Eigenvectors as complex numpy 2D array
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
        dmatrix_to_numpy(py, &self.dmd.a).into_any().unbind()
    }

    #[getter]
    fn a_allclose(&self) -> bool {
        self.dmd.a_allclose()
    }

    #[getter]
    fn n_seen(&self) -> usize {
        self.dmd.n_seen
    }

    #[getter]
    fn r(&self) -> usize {
        self.dmd.r
    }

    #[getter]
    fn m(&self) -> usize {
        self.dmd.m
    }

    #[getter]
    fn get_window_size(&self) -> usize {
        self.window_size
    }

    // ---- SVD accessors (matches Python OnlineDMD._svd._U/_S/_Vt) ----

    /// Left singular vectors (m, r) of the backing OnlineSVDZhang.
    /// Returns ``None`` if the SVD was never fit (e.g. ``r == 0`` or ``r >= m``).
    #[getter]
    fn svd_u<'py>(&self, py: Python<'py>) -> PyResult<Option<PyObject>> {
        if let Some(ref svd) = self.dmd.svd {
            if svd.is_initialized() {
                return Ok(Some(dmatrix_to_numpy(py, &svd.u).into_any().unbind()));
            }
        }
        Ok(None)
    }

    /// Singular values (r,) of the backing OnlineSVDZhang.
    #[getter]
    fn svd_s<'py>(&self, py: Python<'py>) -> PyResult<Option<PyObject>> {
        if let Some(ref svd) = self.dmd.svd {
            if svd.is_initialized() {
                let arr = PyArray1::from_vec(py, svd.s.iter().copied().collect());
                return Ok(Some(arr.into_any().unbind()));
            }
        }
        Ok(None)
    }

    /// Right singular vectors transposed (r, n) of the backing OnlineSVDZhang.
    #[getter]
    fn svd_vt<'py>(&self, py: Python<'py>) -> PyResult<Option<PyObject>> {
        if let Some(ref svd) = self.dmd.svd {
            if svd.is_initialized() {
                return Ok(Some(dmatrix_to_numpy(py, &svd.vt).into_any().unbind()));
            }
        }
        Ok(None)
    }

    /// Y snapshot buffer reconstructed from the rolling window (n, m).
    /// Matches Python ``OnlineDMD._Y`` semantics post-initialization: the
    /// stack of past ``y`` snapshots in the current rolling window.
    /// Returns a zero-row array if no supervised samples have been seen.
    #[getter]
    fn y_buffer<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let m = self.dmd.m;
        let ys: Vec<&Vec<f64>> = self
            .window
            .iter()
            .filter_map(|(_x, y)| y.as_ref())
            .collect();
        let n = ys.len();
        let mut mat = DMatrix::<f64>::zeros(n, m.max(1));
        for (i, y) in ys.iter().enumerate() {
            for (j, &v) in y.iter().enumerate().take(m) {
                mat[(i, j)] = v;
            }
        }
        // Truncate columns to actual m if zero
        if n == 0 {
            let arr = numpy::PyArray2::<f64>::zeros(py, [0, m], false);
            return Ok(arr.into_any().unbind());
        }
        Ok(dmatrix_to_numpy(py, &mat.view((0, 0), (n, m)).clone_owned())
            .into_any()
            .unbind())
    }

    // ---- Transform ----

    fn transform_one<'py>(
        &self,
        py: Python<'py>,
        x: Bound<'_, pyo3::PyAny>,
    ) -> PyResult<PyObject> {
        let x_vec = extract_f64_vec(py, &x)?;
        let result = self.dmd.transform(&x_vec);
        let arr = PyArray1::from_vec(py, result);
        Ok(arr.into_any().unbind())
    }

    fn transform_many<'py>(
        &self,
        py: Python<'py>,
        x: Bound<'_, pyo3::PyAny>,
    ) -> PyResult<PyObject> {
        let x_mat = extract_2d_array(&x)?;
        let result = self.dmd.transform_many(&x_mat);
        let arr = dmatrix_to_numpy(py, &result);
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
    ) -> PyResult<(usize, f64, usize, usize, bool, Option<f64>, Option<u64>)> {
        Ok((
            self.init_r,
            self.init_w,
            self.window_size,
            self.init_initialize,
            self.init_exponential_weighting,
            self.init_eig_rtol,
            self.init_seed,
        ))
    }
}
