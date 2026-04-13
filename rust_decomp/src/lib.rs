pub mod linalg;
pub mod online_dmd;
pub mod online_dmdwc;
pub mod online_svd;
pub mod rolling_dmd;
pub mod rolling_dmdwc;

use pyo3::prelude::*;

#[pymodule]
fn _rust_decomp(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<rolling_dmd::RustRollingDMD>()?;
    m.add_class::<rolling_dmdwc::RustRollingDMDwC>()?;
    Ok(())
}
