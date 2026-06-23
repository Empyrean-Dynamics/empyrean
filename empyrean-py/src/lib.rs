//! PyO3 bindings for the empyrean v0.7.0 safe wrapper.
//!
//! Surfaces the v0.7.0 community-tier API: propagation, ephemeris,
//! orbit determination, transforms. Thrust, planning/visibility, and
//! the Full force model tier are excluded per the empyrean-core
//! release.toml manifest.

// PyO3 `#[pyfunction]` signatures mirror the Python API surface one-to-one, so
// several take more than clippy's 7-argument threshold by design. The numpy
// marshaling also uses explicit index loops over fixed-size covariance/state
// matrices, and a few PyO3 return types are genuinely complex tuples.
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::type_complexity)]

use numpy::ndarray::{Array1, Array2, Array3, Array4};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::sync::OnceLock;

// ══════════════════════════════════════════════════════════
//  Global state
// ══════════════════════════════════════════════════════════

static CONTEXT: OnceLock<empyrean::Context> = OnceLock::new();

fn get_context() -> PyResult<&'static empyrean::Context> {
    CONTEXT.get().ok_or_else(|| {
        PyRuntimeError::new_err("empyrean not initialized. Call empyrean.initialize() first.")
    })
}

fn to_pyerr(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Map a NAIF integer code (the wire format the Python side emits) to
/// the typed [`empyrean::Origin`] the wrapper now requires. Surfaces
/// the bad code in the error so a caller passing a stale int (e.g. 499
/// for "Mars body" instead of 4 for the Mars barycenter) sees what
/// went wrong.
fn origin_from_naif(naif: i32) -> PyResult<empyrean::Origin> {
    empyrean::Origin::from_naif_id(naif)
        .ok_or_else(|| PyValueError::new_err(format!("unknown NAIF id for origin: {naif}")))
}

// ══════════════════════════════════════════════════════════
//  _initialize
// ══════════════════════════════════════════════════════════

#[pyfunction]
#[pyo3(signature = (data_dir=None, de440_path=None, gm_path=None))]
fn _initialize(
    py: Python<'_>,
    data_dir: Option<&str>,
    de440_path: Option<&str>,
    gm_path: Option<&str>,
) -> PyResult<()> {
    if CONTEXT.get().is_some() {
        return Ok(());
    }

    let ctx = py.detach(|| {
        if let (Some(de440), Some(gm)) = (de440_path, gm_path) {
            empyrean::Context::new_minimal(std::path::Path::new(de440), std::path::Path::new(gm))
                .map_err(to_pyerr)
        } else {
            let dir = data_dir.map(std::path::Path::new);
            empyrean::Context::from_data_dir(dir).map_err(to_pyerr)
        }
    })?;

    let _ = CONTEXT.set(ctx);
    Ok(())
}

// ══════════════════════════════════════════════════════════
//  _download_data
// ══════════════════════════════════════════════════════════

#[pyfunction]
#[pyo3(signature = (data_dir=None))]
fn _download_data(py: Python<'_>, data_dir: Option<&str>) -> PyResult<String> {
    let dir = py
        .detach(|| empyrean::download_data(data_dir.map(std::path::Path::new)))
        .map_err(to_pyerr)?;
    Ok(dir.to_string_lossy().into_owned())
}

// ══════════════════════════════════════════════════════════
//  _default_data_dir
// ══════════════════════════════════════════════════════════

#[pyfunction]
fn _default_data_dir() -> PyResult<String> {
    let dir = empyrean::default_data_dir().map_err(to_pyerr)?;
    Ok(dir.to_string_lossy().into_owned())
}

// ══════════════════════════════════════════════════════════
//  _version_string / _versions
// ══════════════════════════════════════════════════════════

/// Multi-line version report — `empyrean-core <ver>\nvilleneuve <ver>\n…`.
/// Mirrors `empyrean::version_string`.
#[pyfunction]
fn _version_string() -> PyResult<String> {
    empyrean::version_string().map_err(to_pyerr)
}

/// Per-crate versions of the empyrean stack as a 4-tuple
/// `(empyrean_core, villeneuve, scott, nolan)`. The Python wrapper
/// turns this into a `Versions` dataclass on import.
#[pyfunction]
fn _versions() -> PyResult<(String, String, String, String)> {
    let v = empyrean::versions().map_err(to_pyerr)?;
    Ok((v.empyrean_core, v.villeneuve, v.scott, v.nolan))
}

// ══════════════════════════════════════════════════════════
//  _transform_coordinates
// ══════════════════════════════════════════════════════════

#[pyfunction]
#[pyo3(signature = (
    epochs,
    elements,
    covariances,
    has_covariance,
    representations,
    frames,
    origins,
    target_rep,
    target_frame,
    target_origin,
))]
fn _transform_coordinates<'py>(
    py: Python<'py>,
    epochs: PyReadonlyArray1<'py, f64>,
    elements: PyReadonlyArray2<'py, f64>,
    covariances: PyReadonlyArray3<'py, f64>,
    has_covariance: PyReadonlyArray1<'py, bool>,
    representations: PyReadonlyArray1<'py, i32>,
    frames: PyReadonlyArray1<'py, i32>,
    origins: PyReadonlyArray1<'py, i32>,
    target_rep: i32,
    target_frame: i32,
    target_origin: i32,
) -> PyResult<Bound<'py, PyDict>> {
    let ctx = get_context()?;

    let epochs_arr = epochs.as_array().to_owned();
    let elements_arr = elements.as_array().to_owned();
    let covariances_arr = covariances.as_array().to_owned();
    let has_cov_arr = has_covariance.as_array().to_owned();
    let reps_arr = representations.as_array().to_owned();
    let frames_arr = frames.as_array().to_owned();
    let origins_arr = origins.as_array().to_owned();

    let n = epochs_arr.len();
    let trep = empyrean::int_to_rep(target_rep).map_err(to_pyerr)?;
    let tframe = empyrean::int_to_frame(target_frame).map_err(to_pyerr)?;
    let torigin = origin_from_naif(target_origin)?;

    let results: Vec<empyrean::CoordinateState> = py.detach(|| {
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut elems = [0.0f64; 6];
            for j in 0..6 {
                elems[j] = elements_arr[[i, j]];
            }
            let covariance = if has_cov_arr[i] {
                let mut cov = [[0.0f64; 6]; 6];
                for r in 0..6 {
                    for c in 0..6 {
                        cov[r][c] = covariances_arr[[i, r, c]];
                    }
                }
                Some(cov)
            } else {
                None
            };
            let state = empyrean::CoordinateState {
                epoch: empyrean::Epoch::from_mjd_tdb(epochs_arr[i]),
                elements: elems,
                covariance,
                representation: empyrean::int_to_rep(reps_arr[i]).map_err(to_pyerr)?,
                frame: empyrean::int_to_frame(frames_arr[i]).map_err(to_pyerr)?,
                origin: origin_from_naif(origins_arr[i])?,
            };
            out.push(
                ctx.transform(&state, trep, tframe, torigin)
                    .map_err(to_pyerr)?,
            );
        }
        Ok::<_, PyErr>(out)
    })?;

    let mut out_epochs = Array1::<f64>::zeros(n);
    let mut out_elements = Array2::<f64>::zeros((n, 6));
    let mut out_covariances = Array3::<f64>::zeros((n, 6, 6));
    let mut out_has_cov = Array1::<bool>::default(n);
    let mut out_reps = Array1::<i32>::zeros(n);
    let mut out_frames = Array1::<i32>::zeros(n);
    let mut out_origins = Array1::<i32>::zeros(n);

    for (i, s) in results.iter().enumerate() {
        out_epochs[i] = s.epoch.mjd_tdb().map_err(to_pyerr)?;
        for j in 0..6 {
            out_elements[[i, j]] = s.elements[j];
        }
        if let Some(cov) = &s.covariance {
            out_has_cov[i] = true;
            for r in 0..6 {
                for c in 0..6 {
                    out_covariances[[i, r, c]] = cov[r][c];
                }
            }
        }
        out_reps[i] = empyrean::rep_to_int(s.representation);
        out_frames[i] = empyrean::frame_to_int(s.frame);
        out_origins[i] = s.origin.naif_id();
    }

    let dict = PyDict::new(py);
    dict.set_item("epochs", PyArray1::from_owned_array(py, out_epochs))?;
    dict.set_item("elements", PyArray2::from_owned_array(py, out_elements))?;
    dict.set_item(
        "covariances",
        PyArray3::from_owned_array(py, out_covariances),
    )?;
    dict.set_item(
        "has_covariance",
        PyArray1::from_owned_array(py, out_has_cov),
    )?;
    dict.set_item("representations", PyArray1::from_owned_array(py, out_reps))?;
    dict.set_item("frames", PyArray1::from_owned_array(py, out_frames))?;
    dict.set_item("origins", PyArray1::from_owned_array(py, out_origins))?;

    Ok(dict)
}

// ══════════════════════════════════════════════════════════
//  _query_sbdb / _query_horizons / _query_horizons_vectors
// ══════════════════════════════════════════════════════════

#[pyfunction]
#[pyo3(signature = (names, cache_dir = None))]
fn _query_sbdb<'py>(
    py: Python<'py>,
    names: Vec<String>,
    cache_dir: Option<String>,
) -> PyResult<Bound<'py, PyDict>> {
    let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    let cache_path = cache_dir.as_ref().map(std::path::Path::new);
    let batch = py
        .detach(|| empyrean::query_sbdb(&name_refs, cache_path))
        .map_err(to_pyerr)?;
    orbit_batch_to_pydict(py, &batch)
}

#[pyfunction]
#[pyo3(signature = (names, obs_code, times, cache_dir = None))]
fn _query_horizons<'py>(
    py: Python<'py>,
    names: Vec<String>,
    obs_code: String,
    times: PyReadonlyArray1<'py, f64>,
    cache_dir: Option<String>,
) -> PyResult<Bound<'py, PyDict>> {
    let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    let times_vec = times.as_array().to_vec();
    let cache_path = cache_dir.as_ref().map(std::path::Path::new);
    let entries = py
        .detach(|| empyrean::query_horizons(&name_refs, &obs_code, &times_vec, cache_path))
        .map_err(to_pyerr)?;

    // Pack into the same flat-dict shape `_generate_ephemeris` returns.
    let m = entries.len();
    let mut orbit_ids: Vec<String> = Vec::with_capacity(m);
    let mut object_ids: Vec<String> = Vec::with_capacity(m);
    let mut obs_codes: Vec<String> = Vec::with_capacity(m);
    let mut epoch = Vec::with_capacity(m);
    let mut ra = Vec::with_capacity(m);
    let mut dec = Vec::with_capacity(m);
    let mut rho = Vec::with_capacity(m);
    let mut vrho = Vec::with_capacity(m);
    let mut vra = Vec::with_capacity(m);
    let mut vdec = Vec::with_capacity(m);
    let mut light_time = Vec::with_capacity(m);
    let mut phase_angle = Vec::with_capacity(m);
    let mut elongation = Vec::with_capacity(m);
    let mut helio = Vec::with_capacity(m);
    let mut mag = Vec::with_capacity(m);
    let mut mag_sigma = Vec::with_capacity(m);
    for e in &entries {
        orbit_ids.push(e.orbit_id.clone());
        object_ids.push(String::new()); // wrapper doesn't carry object_id distinct from orbit_id
        obs_codes.push(e.obs_code.clone());
        epoch.push(e.epoch.mjd_tdb().map_err(to_pyerr)?);
        ra.push(e.ra_deg);
        dec.push(e.dec_deg);
        rho.push(e.rho_au);
        vrho.push(e.vrho_au_day);
        vra.push(e.vra_deg_day);
        vdec.push(e.vdec_deg_day);
        light_time.push(e.light_time_days);
        phase_angle.push(e.phase_angle_deg);
        elongation.push(e.elongation_deg);
        helio.push(e.heliocentric_distance_au);
        mag.push(e.mag);
        mag_sigma.push(e.mag_sigma);
    }
    let dict = PyDict::new(py);
    dict.set_item("orbit_id", orbit_ids)?;
    dict.set_item("object_id", object_ids)?;
    dict.set_item("obs_code", obs_codes)?;
    dict.set_item("epoch", PyArray1::from_vec(py, epoch))?;
    dict.set_item("ra", PyArray1::from_vec(py, ra))?;
    dict.set_item("dec", PyArray1::from_vec(py, dec))?;
    dict.set_item("rho", PyArray1::from_vec(py, rho))?;
    dict.set_item("vrho", PyArray1::from_vec(py, vrho))?;
    dict.set_item("vra", PyArray1::from_vec(py, vra))?;
    dict.set_item("vdec", PyArray1::from_vec(py, vdec))?;
    dict.set_item("light_time", PyArray1::from_vec(py, light_time))?;
    dict.set_item("phase_angle", PyArray1::from_vec(py, phase_angle))?;
    dict.set_item("elongation", PyArray1::from_vec(py, elongation))?;
    dict.set_item("heliocentric_distance", PyArray1::from_vec(py, helio))?;
    dict.set_item("mag", PyArray1::from_vec(py, mag))?;
    dict.set_item("mag_sigma", PyArray1::from_vec(py, mag_sigma))?;
    Ok(dict)
}

#[pyfunction]
#[pyo3(signature = (command, epoch_mjd_tdb, cache_dir = None))]
fn _query_horizons_vectors<'py>(
    py: Python<'py>,
    command: String,
    epoch_mjd_tdb: f64,
    cache_dir: Option<String>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let cache_path = cache_dir.as_ref().map(std::path::Path::new);
    let (pos, vel) = py
        .detach(|| empyrean::query_horizons_vectors(&command, epoch_mjd_tdb, cache_path))
        .map_err(to_pyerr)?;
    Ok((
        PyArray1::from_vec(py, pos.to_vec()),
        PyArray1::from_vec(py, vel.to_vec()),
    ))
}

#[pyfunction]
#[pyo3(signature = (designations, cache_dir = None))]
fn _query_observations<'py>(
    py: Python<'py>,
    designations: Vec<String>,
    cache_dir: Option<String>,
) -> PyResult<Bound<'py, PyDict>> {
    let id_refs: Vec<&str> = designations.iter().map(|s| s.as_str()).collect();
    let cache_path = cache_dir.as_ref().map(std::path::Path::new);
    let observations = py
        .detach(|| empyrean::query_observations(&id_refs, cache_path))
        .map_err(to_pyerr)?;

    // Reuse the same shape `_read_ades` produces, so the Python
    // wrapper can hand the dict straight to `Observations.from_kwargs`.
    let n = observations.len();
    let mut perm_ids: Vec<String> = Vec::with_capacity(n);
    let mut prov_ids: Vec<String> = Vec::with_capacity(n);
    let mut trk_subs: Vec<String> = Vec::with_capacity(n);
    let mut obs_ids: Vec<String> = Vec::with_capacity(n);
    let mut obs_sub_ids: Vec<String> = Vec::with_capacity(n);
    let mut trk_ids: Vec<String> = Vec::with_capacity(n);
    let mut stns: Vec<String> = Vec::with_capacity(n);
    let mut modes: Vec<String> = Vec::with_capacity(n);
    let mut progs: Vec<String> = Vec::with_capacity(n);
    let mut sys_v: Vec<String> = Vec::with_capacity(n);
    let mut ctr_arr = Vec::with_capacity(n);
    let mut pos1_arr = Vec::with_capacity(n);
    let mut pos2_arr = Vec::with_capacity(n);
    let mut pos3_arr = Vec::with_capacity(n);
    let mut obs_times: Vec<String> = Vec::with_capacity(n);
    let mut ra_arr = Vec::with_capacity(n);
    let mut dec_arr = Vec::with_capacity(n);
    let mut rms_ra_arr = Vec::with_capacity(n);
    let mut rms_dec_arr = Vec::with_capacity(n);
    let mut rms_corr_arr = Vec::with_capacity(n);
    let mut ast_cats: Vec<String> = Vec::with_capacity(n);
    let mut mag_arr = Vec::with_capacity(n);
    let mut rms_mag_arr = Vec::with_capacity(n);
    let mut bands: Vec<String> = Vec::with_capacity(n);
    let mut phot_cats: Vec<String> = Vec::with_capacity(n);
    let mut phot_ap_arr = Vec::with_capacity(n);
    let mut log_snr_arr = Vec::with_capacity(n);
    let mut seeing_arr = Vec::with_capacity(n);
    let mut exp_arr = Vec::with_capacity(n);
    let mut rms_fit_arr = Vec::with_capacity(n);
    let mut n_stars_arr = Vec::with_capacity(n);
    let mut notes_v: Vec<String> = Vec::with_capacity(n);
    let mut remarks_v: Vec<String> = Vec::with_capacity(n);

    let opt_to_nan = |v: Option<f64>| v.unwrap_or(f64::NAN);
    for obs in observations.iter() {
        perm_ids.push(obs.perm_id.unwrap_or_default());
        prov_ids.push(obs.prov_id.unwrap_or_default());
        trk_subs.push(obs.trk_sub.unwrap_or_default());
        obs_ids.push(obs.obs_id.unwrap_or_default());
        obs_sub_ids.push(obs.obs_sub_id.unwrap_or_default());
        trk_ids.push(obs.trk_id.unwrap_or_default());
        stns.push(obs.obs_code);
        modes.push(obs.mode.unwrap_or_default());
        progs.push(obs.prog.unwrap_or_default());
        sys_v.push(obs.sys.unwrap_or_default());
        ctr_arr.push(opt_to_nan(obs.ctr));
        pos1_arr.push(opt_to_nan(obs.pos1));
        pos2_arr.push(opt_to_nan(obs.pos2));
        pos3_arr.push(opt_to_nan(obs.pos3));
        obs_times.push(obs.obs_time);
        ra_arr.push(obs.ra_deg);
        dec_arr.push(obs.dec_deg);
        rms_ra_arr.push(obs.rms_ra_arcsec);
        rms_dec_arr.push(obs.rms_dec_arcsec);
        rms_corr_arr.push(opt_to_nan(obs.rms_corr));
        ast_cats.push(obs.ast_cat.unwrap_or_default());
        mag_arr.push(opt_to_nan(obs.mag));
        rms_mag_arr.push(opt_to_nan(obs.rms_mag));
        bands.push(obs.band.unwrap_or_default());
        phot_cats.push(obs.phot_cat.unwrap_or_default());
        phot_ap_arr.push(opt_to_nan(obs.phot_ap));
        log_snr_arr.push(opt_to_nan(obs.log_snr));
        seeing_arr.push(opt_to_nan(obs.seeing));
        exp_arr.push(opt_to_nan(obs.exp));
        rms_fit_arr.push(opt_to_nan(obs.rms_fit));
        n_stars_arr.push(obs.n_stars.map(|v| v as i32).unwrap_or(-1));
        notes_v.push(obs.notes.unwrap_or_default());
        remarks_v.push(obs.remarks.unwrap_or_default());
    }

    let dict = PyDict::new(py);
    dict.set_item("perm_id", perm_ids)?;
    dict.set_item("prov_id", prov_ids)?;
    dict.set_item("trk_sub", trk_subs)?;
    dict.set_item("obs_id", obs_ids)?;
    dict.set_item("obs_sub_id", obs_sub_ids)?;
    dict.set_item("trk_id", trk_ids)?;
    dict.set_item("stn", stns)?;
    dict.set_item("mode", modes)?;
    dict.set_item("prog", progs)?;
    dict.set_item("sys", sys_v)?;
    dict.set_item("ctr", PyArray1::from_vec(py, ctr_arr))?;
    dict.set_item("pos1", PyArray1::from_vec(py, pos1_arr))?;
    dict.set_item("pos2", PyArray1::from_vec(py, pos2_arr))?;
    dict.set_item("pos3", PyArray1::from_vec(py, pos3_arr))?;
    dict.set_item("obs_time", obs_times)?;
    dict.set_item("ra", PyArray1::from_vec(py, ra_arr))?;
    dict.set_item("dec", PyArray1::from_vec(py, dec_arr))?;
    dict.set_item("rms_ra", PyArray1::from_vec(py, rms_ra_arr))?;
    dict.set_item("rms_dec", PyArray1::from_vec(py, rms_dec_arr))?;
    dict.set_item("rms_corr", PyArray1::from_vec(py, rms_corr_arr))?;
    dict.set_item("ast_cat", ast_cats)?;
    dict.set_item("mag", PyArray1::from_vec(py, mag_arr))?;
    dict.set_item("rms_mag", PyArray1::from_vec(py, rms_mag_arr))?;
    dict.set_item("band", bands)?;
    dict.set_item("phot_cat", phot_cats)?;
    dict.set_item("phot_ap", PyArray1::from_vec(py, phot_ap_arr))?;
    dict.set_item("log_snr", PyArray1::from_vec(py, log_snr_arr))?;
    dict.set_item("seeing", PyArray1::from_vec(py, seeing_arr))?;
    dict.set_item("exp", PyArray1::from_vec(py, exp_arr))?;
    dict.set_item("rms_fit", PyArray1::from_vec(py, rms_fit_arr))?;
    dict.set_item("n_stars", PyArray1::from_vec(py, n_stars_arr))?;
    dict.set_item("notes", notes_v)?;
    dict.set_item("remarks", remarks_v)?;
    Ok(dict)
}

#[pyfunction]
#[pyo3(signature = (designations, cache_dir = None))]
fn _query_radar<'py>(
    py: Python<'py>,
    designations: Vec<String>,
    cache_dir: Option<String>,
) -> PyResult<Bound<'py, PyDict>> {
    let id_refs: Vec<&str> = designations.iter().map(|s| s.as_str()).collect();
    let cache_path = cache_dir.as_ref().map(std::path::Path::new);
    let radar = py
        .detach(|| empyrean::query_radar(&id_refs, cache_path))
        .map_err(to_pyerr)?;

    // Reuse the same flat radar-dict shape `_read_ades` produces, so the
    // Python wrapper builds `ADESRadarObservations` the same way.
    radar_table_to_pydict(py, &radar)
}

// ══════════════════════════════════════════════════════════
//  _propagate
// ══════════════════════════════════════════════════════════

/// Propagate orbits to target epochs.
///
/// v0.7.0: drops thrust arcs, MonteCarlo / SigmaPoint /
/// GaussianMixture uncertainty methods, ForceModelTier::Full, and
/// num_threads. Those unused kwargs are accepted for ABI
/// compatibility with older Python callers and rejected with a
/// clear error if used. Photometric params (`phot_h`, `phot_slope1`,
/// `phot_slope2`, `phot_system`) ARE wired — propagation is
/// agnostic to V-magnitude, but the photometry attached here flows
/// downstream to ephemeris generation. `Auto` uncertainty (tag 4)
/// is wired.
#[pyfunction]
#[pyo3(signature = (
    orbit_ids,
    object_ids,
    epochs,
    elements,
    covariances,
    has_covariance,
    representations,
    frames,
    origins,
    times_mjd_tdb,
    force_model,
    uncertainty_method,
    a1s,
    a2s,
    a3s,
    phot_h,
    phot_slope1,
    phot_system,
    phot_slope2 = None,
    num_threads = None,
    epsilon = None,
    thrust_arcs = None,
    ng_alphas = None,
    ng_r0s = None,
    ng_ms = None,
    ng_ns = None,
    ng_ks = None,
    non_grav_dts = None,
    gm_threshold = 1.0,
    gm_max_depth = 3,
    gm_components_per_split = 3,
    sigma_n_sigma = 1.0,
    sigma_samples_per_plane = 8,
    mc_n_samples = 1000,
    mc_seed = None,
    propagation_config_dict = None,
    with_tagged_covariance = false,
))]
fn _propagate<'py>(
    py: Python<'py>,
    orbit_ids: Vec<String>,
    object_ids: Vec<String>,
    epochs: PyReadonlyArray1<'py, f64>,
    elements: PyReadonlyArray2<'py, f64>,
    covariances: PyReadonlyArray3<'py, f64>,
    has_covariance: PyReadonlyArray1<'py, bool>,
    representations: PyReadonlyArray1<'py, i32>,
    frames: PyReadonlyArray1<'py, i32>,
    origins: PyReadonlyArray1<'py, i32>,
    times_mjd_tdb: PyReadonlyArray1<'py, f64>,
    force_model: i32,
    uncertainty_method: i32,
    a1s: PyReadonlyArray1<'py, f64>,
    a2s: PyReadonlyArray1<'py, f64>,
    a3s: PyReadonlyArray1<'py, f64>,
    phot_h: PyReadonlyArray1<'py, f64>,
    phot_slope1: PyReadonlyArray1<'py, f64>,
    phot_system: PyReadonlyArray1<'py, i32>,
    phot_slope2: Option<PyReadonlyArray1<'py, f64>>,
    num_threads: Option<usize>,
    epsilon: Option<f64>,
    thrust_arcs: Option<String>,
    ng_alphas: Option<PyReadonlyArray1<'py, f64>>,
    ng_r0s: Option<PyReadonlyArray1<'py, f64>>,
    ng_ms: Option<PyReadonlyArray1<'py, f64>>,
    ng_ns: Option<PyReadonlyArray1<'py, f64>>,
    ng_ks: Option<PyReadonlyArray1<'py, f64>>,
    // SBDB non-grav time delay (days). NaN entries → no delay; whole
    // array None → no DT for any orbit.
    non_grav_dts: Option<PyReadonlyArray1<'py, f64>>,
    gm_threshold: f64,
    gm_max_depth: usize,
    gm_components_per_split: usize,
    sigma_n_sigma: f64,
    sigma_samples_per_plane: usize,
    mc_n_samples: usize,
    mc_seed: Option<u64>,
    propagation_config_dict: Option<&Bound<'py, PyDict>>,
    // Opt-in: also fill provenance-tagged resolved-kind covariance
    // readback arrays (aligned 1:1 with `states`) into the result dict.
    // The default non-tagged path is unchanged and pays nothing.
    with_tagged_covariance: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = (
        num_threads,
        gm_threshold,
        gm_max_depth,
        gm_components_per_split,
        sigma_n_sigma,
        sigma_samples_per_plane,
        mc_n_samples,
        mc_seed,
    );
    if thrust_arcs.is_some() {
        return Err(PyValueError::new_err(
            "thrust_arcs is not supported in empyrean v0.7.0",
        ));
    }
    let ctx = get_context()?;

    let epochs_arr = epochs.as_array().to_owned();
    let elements_arr = elements.as_array().to_owned();
    let covariances_arr = covariances.as_array().to_owned();
    let has_cov_arr = has_covariance.as_array().to_owned();
    let reps_arr = representations.as_array().to_owned();
    let frames_arr = frames.as_array().to_owned();
    let origins_arr = origins.as_array().to_owned();
    let times_arr = times_mjd_tdb.as_array().to_owned();
    let a1s_arr = a1s.as_array().to_owned();
    let a2s_arr = a2s.as_array().to_owned();
    let a3s_arr = a3s.as_array().to_owned();
    let phot_h_arr = phot_h.as_array().to_owned();
    let phot_slope1_arr = phot_slope1.as_array().to_owned();
    let phot_system_arr = phot_system.as_array().to_owned();
    let phot_slope2_arr = phot_slope2.as_ref().map(|a| a.as_array().to_owned());
    let ng_alpha_arr = ng_alphas.as_ref().map(|a| a.as_array().to_owned());
    let ng_r0_arr = ng_r0s.as_ref().map(|a| a.as_array().to_owned());
    let ng_m_arr = ng_ms.as_ref().map(|a| a.as_array().to_owned());
    let ng_n_arr = ng_ns.as_ref().map(|a| a.as_array().to_owned());
    let ng_k_arr = ng_ks.as_ref().map(|a| a.as_array().to_owned());
    let dt_arr = non_grav_dts.as_ref().map(|a| a.as_array().to_owned());

    let n = epochs_arr.len();

    let mut orbits: Vec<empyrean::Orbit> = Vec::with_capacity(n);
    for i in 0..n {
        let mut elems = [0.0f64; 6];
        for j in 0..6 {
            elems[j] = elements_arr[[i, j]];
        }
        let covariance = if has_cov_arr[i] {
            let mut cov = [[0.0f64; 6]; 6];
            for r in 0..6 {
                for c in 0..6 {
                    cov[r][c] = covariances_arr[[i, r, c]];
                }
            }
            Some(cov)
        } else {
            None
        };
        let state = empyrean::CoordinateState {
            epoch: empyrean::Epoch::from_mjd_tdb(epochs_arr[i]),
            elements: elems,
            covariance,
            representation: empyrean::int_to_rep(reps_arr[i]).map_err(to_pyerr)?,
            frame: empyrean::int_to_frame(frames_arr[i]).map_err(to_pyerr)?,
            origin: origin_from_naif(origins_arr[i])?,
        };
        let mut orbit = empyrean::Orbit::new(state);
        orbit = orbit.with_nongrav(a1s_arr[i], a2s_arr[i], a3s_arr[i]);
        // Optional g(r) overrides — when not provided, the C ABI
        // defaults to inverse_square (asteroid Yarkovsky / SRP).
        if let (Some(a), Some(r), Some(m), Some(n), Some(k)) =
            (&ng_alpha_arr, &ng_r0_arr, &ng_m_arr, &ng_n_arr, &ng_k_arr)
        {
            orbit = orbit.with_g_function(a[i], r[i], m[i], n[i], k[i]);
        }
        // Optional SBDB non-grav DT (days). NaN cells mean "no delay";
        // a `None` array means no DT for any orbit.
        if let Some(dts) = &dt_arr {
            let dt = dts[i];
            if dt.is_finite() {
                orbit = orbit.with_non_grav_dt(Some(dt));
            }
        }
        // Photometry: ephemeris generation downstream consumes (H, slope1,
        // slope2) per the chosen phase function. NaN H or model = -1 leaves
        // photometry unset and the row's mag = NaN.
        let pf_int = phot_system_arr[i];
        let h = phot_h_arr[i];
        let g = phot_slope1_arr[i];
        if h.is_finite() && pf_int >= 0 {
            let pf = match pf_int {
                0 => empyrean::PhaseFunction::HG,
                1 => empyrean::PhaseFunction::HG1G2,
                2 => empyrean::PhaseFunction::HG12,
                _ => empyrean::PhaseFunction::HG,
            };
            // HG1G2 uses G2 in slot 2; HG and HG12 ignore it. The
            // caller-supplied phot_slope2 array carries G2 when set;
            // otherwise default to 0.0 (which is correct for HG/HG12
            // and a wrong-but-non-crashing fallback for HG1G2 — file
            // a clear array if HG1G2 fits matter).
            let s2 = phot_slope2_arr.as_ref().map_or(0.0, |a| a[i]);
            orbit = orbit.with_photometry(pf, h, g, s2);
        }
        orbits.push(orbit);
    }

    let force_model_tier = match force_model {
        0 => empyrean::ForceModelTier::Approximate,
        1 => empyrean::ForceModelTier::Basic,
        2 => empyrean::ForceModelTier::Standard,
        _ => {
            return Err(PyRuntimeError::new_err(format!(
                "unknown or unsupported force model tier: {force_model}"
            )));
        }
    };

    let uncertainty = match uncertainty_method {
        0 => empyrean::UncertaintyMethod::FirstOrder,
        1 => empyrean::UncertaintyMethod::SecondOrder,
        4 => empyrean::UncertaintyMethod::auto(),
        _ => {
            return Err(PyRuntimeError::new_err(format!(
                "uncertainty method {uncertainty_method} is not supported \
                 (FirstOrder=0, SecondOrder=1, Auto=4)"
            )));
        }
    };

    let mut config = empyrean::PropagationConfig {
        force_model: force_model_tier,
        uncertainty_method: uncertainty,
        frame: empyrean::Frame::ICRF,
        ..empyrean::PropagationConfig::default()
    };
    if let Some(eps) = epsilon
        && eps > 0.0
    {
        config.advanced.epsilon = eps;
    }
    if let Some(threads) = num_threads {
        config.num_threads = std::num::NonZeroUsize::new(threads);
    }
    // The full nested config dict (events / diagnostics / advanced /
    // excluded_perturbers / max_propagation_time / etc.) overrides the
    // values built from the flat args above.
    if let Some(d) = propagation_config_dict {
        apply_propagation_config_dict(&mut config, d)?;
    }

    let times_slice: Vec<empyrean::Epoch> = times_arr
        .iter()
        .map(|&t| empyrean::Epoch::from_mjd_tdb(t))
        .collect();
    let n_times = times_slice.len();

    let prop_result = py
        .detach(|| ctx.propagate(&orbits, &times_slice, &config))
        .map_err(to_pyerr)?;

    let m = prop_result.states.len();

    let mut out_orbit_ids: Vec<String> = Vec::with_capacity(m);
    let mut out_object_ids: Vec<String> = Vec::with_capacity(m);
    let mut out_epochs = Array1::<f64>::zeros(m);
    let mut out_x = Array1::<f64>::zeros(m);
    let mut out_y = Array1::<f64>::zeros(m);
    let mut out_z = Array1::<f64>::zeros(m);
    let mut out_vx = Array1::<f64>::zeros(m);
    let mut out_vy = Array1::<f64>::zeros(m);
    let mut out_vz = Array1::<f64>::zeros(m);
    let mut out_frames = Array1::<i32>::zeros(m);
    let mut out_origins = Array1::<i32>::zeros(m);
    let mut out_covariances = Array3::<f64>::zeros((m, 6, 6));
    let mut out_has_cov = Array1::<bool>::default(m);
    let mut out_stms = Array3::<f64>::from_elem((m, 6, 6), f64::NAN);
    let mut out_has_stm = Array1::<bool>::default(m);
    // Second-order state-transition tensor (Park–Scheeres Ψ). Reshaped
    // to (m, 216) on the Python side; row-major [a][b][c] ordering
    // matches the STM's [r][c].
    let mut out_stts = Array4::<f64>::from_elem((m, 6, 6, 6), f64::NAN);
    let mut out_has_stt = Array1::<bool>::default(m);
    // Resolved covariance kind per output state (linear / second-order /
    // …) — the provenance the tagged-covariance path also carries.
    let mut out_resolved_kind = Array1::<u8>::zeros(m);

    for (i, state) in prop_result.states.iter().enumerate() {
        let orbit_idx = if n_times > 0 { i / n_times } else { 0 };
        let orbit_id = orbit_ids
            .get(orbit_idx)
            .cloned()
            .unwrap_or_else(String::new);
        let object_id = object_ids
            .get(orbit_idx)
            .cloned()
            .unwrap_or_else(String::new);
        out_orbit_ids.push(orbit_id);
        out_object_ids.push(object_id);
        out_epochs[i] = state.epoch.mjd_tdb().map_err(to_pyerr)?;
        out_x[i] = state.position[0];
        out_y[i] = state.position[1];
        out_z[i] = state.position[2];
        out_vx[i] = state.velocity[0];
        out_vy[i] = state.velocity[1];
        out_vz[i] = state.velocity[2];
        out_frames[i] = empyrean::frame_to_int(state.frame);
        out_origins[i] = state.origin.naif_id();

        if let Some(c) = &state.covariance {
            out_has_cov[i] = true;
            for r in 0..6 {
                for c_idx in 0..6 {
                    out_covariances[[i, r, c_idx]] = c[r][c_idx];
                }
            }
        }
        if let Some(stm) = &state.stm {
            out_has_stm[i] = true;
            for r in 0..6 {
                for c_idx in 0..6 {
                    out_stms[[i, r, c_idx]] = stm[r][c_idx];
                }
            }
        }
        if let Some(stt) = &state.stt {
            out_has_stt[i] = true;
            for a in 0..6 {
                for b in 0..6 {
                    for c_idx in 0..6 {
                        out_stts[[i, a, b, c_idx]] = stt[a][b][c_idx];
                    }
                }
            }
        }
        out_resolved_kind[i] = covariance_kind_to_u8(state.resolved_kind);
    }

    let n_events = prop_result.events.len();
    let mut ev_orbit_ids: Vec<String> = Vec::with_capacity(n_events);
    let mut ev_object_ids: Vec<String> = Vec::with_capacity(n_events);
    let mut ev_event_types: Vec<String> = Vec::with_capacity(n_events);
    let mut ev_bodies: Vec<String> = Vec::with_capacity(n_events);
    let mut ev_body_naif_ids = Array1::<i32>::zeros(n_events);
    let mut ev_epochs = Array1::<f64>::zeros(n_events);
    let mut ev_distance_au = Array1::<f64>::zeros(n_events);
    let mut ev_distance_km = Array1::<f64>::zeros(n_events);
    let mut ev_relative_velocity = Array1::<f64>::zeros(n_events);
    // capture_start / capture_end payload
    let mut ev_two_body_energy = Array1::<f64>::from_elem(n_events, f64::NAN);
    let mut ev_jacobi = Array1::<f64>::from_elem(n_events, f64::NAN);
    let mut ev_jacobi_sigma = Array1::<f64>::from_elem(n_events, f64::NAN);
    let mut ev_jacobi_l1 = Array1::<f64>::from_elem(n_events, f64::NAN);
    let mut ev_jacobi_l2 = Array1::<f64>::from_elem(n_events, f64::NAN);
    let mut ev_n_periapses = Array1::<i32>::from_elem(n_events, -1);
    // impact payload
    let mut ev_impact_lat = Array1::<f64>::from_elem(n_events, f64::NAN);
    let mut ev_impact_lon = Array1::<f64>::from_elem(n_events, f64::NAN);
    let mut ev_impact_alt = Array1::<f64>::from_elem(n_events, f64::NAN);
    // shadow_entry / shadow_exit payload
    let mut ev_shadow_fraction = Array1::<f64>::from_elem(n_events, f64::NAN);
    let mut ev_illumination = Array1::<f64>::from_elem(n_events, f64::NAN);
    // periapsis relative-state payload
    let mut ev_relative_x = Array1::<f64>::from_elem(n_events, f64::NAN);
    let mut ev_relative_y = Array1::<f64>::from_elem(n_events, f64::NAN);
    let mut ev_relative_z = Array1::<f64>::from_elem(n_events, f64::NAN);
    let mut ev_relative_vx = Array1::<f64>::from_elem(n_events, f64::NAN);
    let mut ev_relative_vy = Array1::<f64>::from_elem(n_events, f64::NAN);
    let mut ev_relative_vz = Array1::<f64>::from_elem(n_events, f64::NAN);
    // possible_impact probability payload
    let mut ev_effective_radius_au = Array1::<f64>::from_elem(n_events, f64::NAN);
    let mut ev_effective_radius_km = Array1::<f64>::from_elem(n_events, f64::NAN);
    let mut ev_sigma_distance_au = Array1::<f64>::from_elem(n_events, f64::NAN);
    let mut ev_ip_linear = Array1::<f64>::from_elem(n_events, f64::NAN);
    let mut ev_ip_second_order = Array1::<f64>::from_elem(n_events, f64::NAN);
    let mut ev_nonlinearity = Array1::<f64>::from_elem(n_events, f64::NAN);
    let mut ev_ip_agm = Array1::<f64>::from_elem(n_events, f64::NAN);
    let mut ev_ip_mc = Array1::<f64>::from_elem(n_events, f64::NAN);
    // covariance_regime_change payload (kind codes: -1 = N/A, else
    // EMPYREAN_COVARIANCE_KIND_* 0..4)
    let mut ev_previous_kind = Array1::<i16>::from_elem(n_events, -1);
    let mut ev_regime_resolved_kind = Array1::<i16>::from_elem(n_events, -1);
    let mut ev_kappa = Array1::<f64>::from_elem(n_events, f64::NAN);
    let mut ev_threshold_below = Array1::<f64>::from_elem(n_events, f64::NAN);
    let mut ev_threshold_above = Array1::<f64>::from_elem(n_events, f64::NAN);

    // The C ABI in empyrean_c/src/propagate.rs:512 fabricates each
    // event's orbit_id as `format!("orbit_{i}")` because the underlying
    // `EmpyreanOrbit` struct has no orbit_id field (TODO: add the field
    // to the C ABI so we can stop reverse-engineering it here). Recover
    // the user-supplied orbit_id and object_id by parsing the index out
    // of "orbit_N" and looking it up in the input arrays.
    for (i, ev) in prop_result.events.iter().enumerate() {
        let user_idx = parse_fabricated_orbit_index(&ev.orbit_id);
        let orbit_id = user_idx
            .and_then(|j| orbit_ids.get(j).cloned())
            .unwrap_or_else(|| ev.orbit_id.clone());
        let object_id = user_idx
            .and_then(|j| object_ids.get(j).cloned())
            .unwrap_or_default();
        ev_orbit_ids.push(orbit_id);
        ev_object_ids.push(object_id);
        ev_event_types.push(ev.event_type.clone());
        // Body is now an `Option<Origin>` on the wrapper Event;
        // serialize to canonical name (empty for None) and the C ABI's
        // -1 sentinel NAIF id for non-body events. Python side relies
        // on the canonical name as the typed `Origin`'s `name`.
        ev_bodies.push(ev.body.map(|o| o.to_string()).unwrap_or_default());
        ev_body_naif_ids[i] = ev.body.map(|o| o.naif_id()).unwrap_or(-1);
        ev_epochs[i] = ev.epoch.mjd_tdb().map_err(to_pyerr)?;
        ev_distance_au[i] = ev.distance_au;
        ev_distance_km[i] = ev.distance_km;
        ev_relative_velocity[i] = ev.relative_velocity_au_day;
        ev_two_body_energy[i] = ev.two_body_energy;
        ev_jacobi[i] = ev.jacobi_constant;
        ev_jacobi_sigma[i] = ev.jacobi_constant_sigma;
        ev_jacobi_l1[i] = ev.jacobi_constant_l1;
        ev_jacobi_l2[i] = ev.jacobi_constant_l2;
        ev_n_periapses[i] = ev.n_periapses.map(|x| x as i32).unwrap_or(-1);
        ev_impact_lat[i] = ev.impact_latitude_deg;
        ev_impact_lon[i] = ev.impact_longitude_deg;
        ev_impact_alt[i] = ev.impact_altitude_km;
        ev_shadow_fraction[i] = ev.shadow_fraction;
        ev_illumination[i] = ev.illumination;
        ev_relative_x[i] = ev.relative_x;
        ev_relative_y[i] = ev.relative_y;
        ev_relative_z[i] = ev.relative_z;
        ev_relative_vx[i] = ev.relative_vx;
        ev_relative_vy[i] = ev.relative_vy;
        ev_relative_vz[i] = ev.relative_vz;
        ev_effective_radius_au[i] = ev.effective_radius_au;
        ev_effective_radius_km[i] = ev.effective_radius_km;
        ev_sigma_distance_au[i] = ev.sigma_distance_au;
        ev_ip_linear[i] = ev.ip_linear;
        ev_ip_second_order[i] = ev.ip_second_order;
        ev_nonlinearity[i] = ev.nonlinearity;
        ev_ip_agm[i] = ev.ip_agm;
        ev_ip_mc[i] = ev.ip_mc;
        ev_previous_kind[i] = ev
            .previous_kind
            .map(|k| covariance_kind_to_u8(k) as i16)
            .unwrap_or(-1);
        ev_regime_resolved_kind[i] = ev
            .regime_resolved_kind
            .map(|k| covariance_kind_to_u8(k) as i16)
            .unwrap_or(-1);
        ev_kappa[i] = ev.kappa;
        ev_threshold_below[i] = ev.threshold_below;
        ev_threshold_above[i] = ev.threshold_above;
    }

    let dict = PyDict::new(py);
    dict.set_item("orbit_ids", out_orbit_ids)?;
    dict.set_item("object_ids", out_object_ids)?;
    dict.set_item("epochs", PyArray1::from_owned_array(py, out_epochs))?;
    dict.set_item("x", PyArray1::from_owned_array(py, out_x))?;
    dict.set_item("y", PyArray1::from_owned_array(py, out_y))?;
    dict.set_item("z", PyArray1::from_owned_array(py, out_z))?;
    dict.set_item("vx", PyArray1::from_owned_array(py, out_vx))?;
    dict.set_item("vy", PyArray1::from_owned_array(py, out_vy))?;
    dict.set_item("vz", PyArray1::from_owned_array(py, out_vz))?;
    dict.set_item("frames", PyArray1::from_owned_array(py, out_frames))?;
    dict.set_item("origins", PyArray1::from_owned_array(py, out_origins))?;
    dict.set_item(
        "covariances",
        PyArray3::from_owned_array(py, out_covariances),
    )?;
    dict.set_item(
        "has_covariance",
        PyArray1::from_owned_array(py, out_has_cov),
    )?;
    dict.set_item("stms", PyArray3::from_owned_array(py, out_stms))?;
    dict.set_item("has_stm", PyArray1::from_owned_array(py, out_has_stm))?;
    dict.set_item("stts", PyArray4::from_owned_array(py, out_stts))?;
    dict.set_item("has_stt", PyArray1::from_owned_array(py, out_has_stt))?;
    dict.set_item(
        "resolved_kind",
        PyArray1::from_owned_array(py, out_resolved_kind),
    )?;

    // ── Opt-in provenance-tagged covariance readback ──────────
    // Fill per-(orbit, epoch) arrays aligned 1:1 with `states`
    // (length m, orbit-major). On an orbit with no covariance the
    // wrapper accessor errors; that orbit's epochs get has_tagged=false
    // and zero-filled rows. Only emitted when the caller opted in, so
    // the default path above pays nothing.
    if with_tagged_covariance {
        fill_tagged_covariance(py, &dict, &prop_result, n_times)?;
    }

    let events_dict = PyDict::new(py);
    events_dict.set_item("orbit_ids", ev_orbit_ids)?;
    events_dict.set_item("object_ids", ev_object_ids)?;
    events_dict.set_item("event_types", ev_event_types)?;
    events_dict.set_item("bodies", ev_bodies)?;
    events_dict.set_item(
        "body_naif_ids",
        PyArray1::from_owned_array(py, ev_body_naif_ids),
    )?;
    events_dict.set_item("epochs", PyArray1::from_owned_array(py, ev_epochs))?;
    events_dict.set_item(
        "distance_au",
        PyArray1::from_owned_array(py, ev_distance_au),
    )?;
    events_dict.set_item(
        "distance_km",
        PyArray1::from_owned_array(py, ev_distance_km),
    )?;
    events_dict.set_item(
        "relative_velocity_au_day",
        PyArray1::from_owned_array(py, ev_relative_velocity),
    )?;
    events_dict.set_item(
        "two_body_energy",
        PyArray1::from_owned_array(py, ev_two_body_energy),
    )?;
    events_dict.set_item("jacobi_constant", PyArray1::from_owned_array(py, ev_jacobi))?;
    events_dict.set_item(
        "jacobi_constant_sigma",
        PyArray1::from_owned_array(py, ev_jacobi_sigma),
    )?;
    events_dict.set_item(
        "jacobi_constant_l1",
        PyArray1::from_owned_array(py, ev_jacobi_l1),
    )?;
    events_dict.set_item(
        "jacobi_constant_l2",
        PyArray1::from_owned_array(py, ev_jacobi_l2),
    )?;
    events_dict.set_item(
        "n_periapses",
        PyArray1::from_owned_array(py, ev_n_periapses),
    )?;
    events_dict.set_item(
        "impact_latitude_deg",
        PyArray1::from_owned_array(py, ev_impact_lat),
    )?;
    events_dict.set_item(
        "impact_longitude_deg",
        PyArray1::from_owned_array(py, ev_impact_lon),
    )?;
    events_dict.set_item(
        "impact_altitude_km",
        PyArray1::from_owned_array(py, ev_impact_alt),
    )?;
    events_dict.set_item(
        "shadow_fraction",
        PyArray1::from_owned_array(py, ev_shadow_fraction),
    )?;
    events_dict.set_item(
        "illumination",
        PyArray1::from_owned_array(py, ev_illumination),
    )?;
    events_dict.set_item("relative_x", PyArray1::from_owned_array(py, ev_relative_x))?;
    events_dict.set_item("relative_y", PyArray1::from_owned_array(py, ev_relative_y))?;
    events_dict.set_item("relative_z", PyArray1::from_owned_array(py, ev_relative_z))?;
    events_dict.set_item(
        "relative_vx",
        PyArray1::from_owned_array(py, ev_relative_vx),
    )?;
    events_dict.set_item(
        "relative_vy",
        PyArray1::from_owned_array(py, ev_relative_vy),
    )?;
    events_dict.set_item(
        "relative_vz",
        PyArray1::from_owned_array(py, ev_relative_vz),
    )?;
    events_dict.set_item(
        "effective_radius_au",
        PyArray1::from_owned_array(py, ev_effective_radius_au),
    )?;
    events_dict.set_item(
        "effective_radius_km",
        PyArray1::from_owned_array(py, ev_effective_radius_km),
    )?;
    events_dict.set_item(
        "sigma_distance_au",
        PyArray1::from_owned_array(py, ev_sigma_distance_au),
    )?;
    events_dict.set_item("ip_linear", PyArray1::from_owned_array(py, ev_ip_linear))?;
    events_dict.set_item(
        "ip_second_order",
        PyArray1::from_owned_array(py, ev_ip_second_order),
    )?;
    events_dict.set_item(
        "nonlinearity",
        PyArray1::from_owned_array(py, ev_nonlinearity),
    )?;
    events_dict.set_item("ip_agm", PyArray1::from_owned_array(py, ev_ip_agm))?;
    events_dict.set_item("ip_mc", PyArray1::from_owned_array(py, ev_ip_mc))?;
    events_dict.set_item(
        "previous_kind",
        PyArray1::from_owned_array(py, ev_previous_kind),
    )?;
    events_dict.set_item(
        "regime_resolved_kind",
        PyArray1::from_owned_array(py, ev_regime_resolved_kind),
    )?;
    events_dict.set_item("kappa", PyArray1::from_owned_array(py, ev_kappa))?;
    events_dict.set_item(
        "threshold_below",
        PyArray1::from_owned_array(py, ev_threshold_below),
    )?;
    events_dict.set_item(
        "threshold_above",
        PyArray1::from_owned_array(py, ev_threshold_above),
    )?;
    dict.set_item("events", events_dict)?;

    Ok(dict)
}

/// Map a wrapper [`empyrean::CovarianceKind`] to its C-ABI integer code
/// (`EMPYREAN_COVARIANCE_KIND_*`): Linear=0 … MonteCarlo=4. Explicit
/// match — never an enum-as-int cast — so a reordering of the wrapper
/// enum can't silently desync the Python-side code.
fn covariance_kind_to_u8(kind: empyrean::CovarianceKind) -> u8 {
    match kind {
        empyrean::CovarianceKind::Linear => 0,
        empyrean::CovarianceKind::SecondOrder => 1,
        empyrean::CovarianceKind::ThirdOrder => 2,
        empyrean::CovarianceKind::Mixture => 3,
        empyrean::CovarianceKind::MonteCarlo => 4,
    }
}

/// Map a wrapper [`empyrean::CovarianceQuality`] to its `(u8, min_eig)`
/// pair: PositiveDefinite=0 (min_eig NaN), Indefinite=1, Repaired=2.
fn covariance_quality_to_u8(quality: empyrean::CovarianceQuality) -> (u8, f64) {
    match quality {
        empyrean::CovarianceQuality::PositiveDefinite => (0, f64::NAN),
        empyrean::CovarianceQuality::Indefinite { min_eig } => (1, min_eig),
        empyrean::CovarianceQuality::Repaired { min_eig } => (2, min_eig),
    }
}

/// Map a wrapper [`empyrean::TargetFunctional`] to its C-ABI integer
/// code: CartesianState=0, CloseApproachMissDistance=1.
fn target_functional_to_u8(tf: empyrean::TargetFunctional) -> u8 {
    match tf {
        empyrean::TargetFunctional::CartesianState => 0,
        empyrean::TargetFunctional::CloseApproachMissDistance => 1,
    }
}

/// Fill the opt-in provenance-tagged covariance arrays into `dict`,
/// aligned 1:1 with the flat orbit-major `states` (length `m`).
///
/// For each `orbit_index` it calls the wrapper accessor
/// [`empyrean::PropagationResult::covariance_series_cartesian`]; on `Ok`
/// the per-epoch entries are written at rows `orbit_index*n_times + k`,
/// on `Err` (e.g. an orbit carrying no covariance) that orbit's whole
/// epoch span keeps `has_tagged=false` with zero-filled rows.
///
/// All emitted arrays are length `m` (orbit-major) so the Python side
/// can lay them straight into a per-`(orbit, epoch)` table aligned with
/// the propagated states.
fn fill_tagged_covariance(
    py: Python<'_>,
    dict: &Bound<'_, PyDict>,
    prop_result: &empyrean::PropagationResult,
    n_times: usize,
) -> PyResult<()> {
    let m = prop_result.states.len();
    let num_orbits = if n_times > 0 { m / n_times } else { 0 };

    let mut matrix = Array3::<f64>::zeros((m, 6, 6));
    let mut state = Array2::<f64>::zeros((m, 6));
    let mut kind = Array1::<u8>::zeros(m);
    let mut mc_seed = Array1::<u64>::zeros(m);
    let mut has_mc_seed = Array1::<bool>::default(m);
    let mut mean_shift_prop = Array2::<f64>::zeros((m, 6));
    let mut has_mean_shift_prop = Array1::<bool>::default(m);
    let mut mean_shift_input = Array2::<f64>::zeros((m, 6));
    let mut has_mean_shift_input = Array1::<bool>::default(m);
    let mut quality = Array1::<u8>::zeros(m);
    let mut quality_min_eig = Array1::<f64>::from_elem(m, f64::NAN);
    let mut non_grav = Array2::<bool>::default((m, 3));
    let mut thrust_segments = Array1::<u32>::zeros(m);
    let mut solved_width = Array1::<u32>::zeros(m);
    let mut target_functional = Array1::<u8>::zeros(m);
    let mut origin = Array1::<i32>::zeros(m);
    let mut frame = Array1::<i32>::zeros(m);
    let mut has_tagged = Array1::<bool>::default(m);

    for orbit_index in 0..num_orbits {
        // An orbit with no covariance makes the accessor error; that's
        // expected, not fatal — leave its epochs flagged false.
        let series = match prop_result.covariance_series_cartesian(orbit_index) {
            Ok(s) => s,
            Err(_) => continue,
        };
        for (k, tagged) in series.iter().enumerate() {
            let i = orbit_index * n_times + k;
            if i >= m {
                break;
            }
            for r in 0..6 {
                state[[i, r]] = tagged.state[r];
                for c in 0..6 {
                    matrix[[i, r, c]] = tagged.matrix[r][c];
                }
            }
            kind[i] = covariance_kind_to_u8(tagged.kind);
            if let Some(seed) = tagged.mc_seed {
                mc_seed[i] = seed;
                has_mc_seed[i] = true;
            }
            if let Some(shift) = tagged.mean_shift_prop {
                has_mean_shift_prop[i] = true;
                for r in 0..6 {
                    mean_shift_prop[[i, r]] = shift[r];
                }
            }
            if let Some(shift) = tagged.mean_shift_input {
                has_mean_shift_input[i] = true;
                for r in 0..6 {
                    mean_shift_input[[i, r]] = shift[r];
                }
            }
            let (q, min_eig) = covariance_quality_to_u8(tagged.quality);
            quality[i] = q;
            quality_min_eig[i] = min_eig;
            for a in 0..3 {
                non_grav[[i, a]] = tagged.non_grav[a];
            }
            thrust_segments[i] = tagged.thrust_segments;
            solved_width[i] = tagged.solved_width;
            target_functional[i] = target_functional_to_u8(tagged.target_functional);
            origin[i] = tagged.origin.naif_id();
            frame[i] = empyrean::frame_to_int(tagged.frame);
            has_tagged[i] = true;
        }
    }

    let tagged_dict = PyDict::new(py);
    tagged_dict.set_item("matrix", PyArray3::from_owned_array(py, matrix))?;
    tagged_dict.set_item("state", PyArray2::from_owned_array(py, state))?;
    tagged_dict.set_item("kind", PyArray1::from_owned_array(py, kind))?;
    tagged_dict.set_item("mc_seed", PyArray1::from_owned_array(py, mc_seed))?;
    tagged_dict.set_item("has_mc_seed", PyArray1::from_owned_array(py, has_mc_seed))?;
    tagged_dict.set_item(
        "mean_shift_prop",
        PyArray2::from_owned_array(py, mean_shift_prop),
    )?;
    tagged_dict.set_item(
        "has_mean_shift_prop",
        PyArray1::from_owned_array(py, has_mean_shift_prop),
    )?;
    tagged_dict.set_item(
        "mean_shift_input",
        PyArray2::from_owned_array(py, mean_shift_input),
    )?;
    tagged_dict.set_item(
        "has_mean_shift_input",
        PyArray1::from_owned_array(py, has_mean_shift_input),
    )?;
    tagged_dict.set_item("quality", PyArray1::from_owned_array(py, quality))?;
    tagged_dict.set_item(
        "quality_min_eig",
        PyArray1::from_owned_array(py, quality_min_eig),
    )?;
    tagged_dict.set_item("non_grav", PyArray2::from_owned_array(py, non_grav))?;
    tagged_dict.set_item(
        "thrust_segments",
        PyArray1::from_owned_array(py, thrust_segments),
    )?;
    tagged_dict.set_item("solved_width", PyArray1::from_owned_array(py, solved_width))?;
    tagged_dict.set_item(
        "target_functional",
        PyArray1::from_owned_array(py, target_functional),
    )?;
    tagged_dict.set_item("origin", PyArray1::from_owned_array(py, origin))?;
    tagged_dict.set_item("frame", PyArray1::from_owned_array(py, frame))?;
    tagged_dict.set_item("has_tagged", PyArray1::from_owned_array(py, has_tagged))?;

    dict.set_item("tagged_covariance", tagged_dict)?;
    Ok(())
}

// ══════════════════════════════════════════════════════════
//  _compute_impact_probabilities / _compute_b_planes
//
//  Multi-method IP and B-plane wrappers — one full propagation per
//  supplied UncertaintyMethod, results returned as flat numpy arrays
//  tagged with the method that produced each row. Mirrors
//  empyrean::Context::compute_impact_probabilities /
//  compute_b_planes (which mirrors empyrean_core::impact). Method
//  tags use the same integer encoding as `_propagate`'s
//  `uncertainty_method` argument: 0 = FirstOrder, 1 = SecondOrder,
//  2 = SigmaPoint (default params), 3 = MonteCarlo (default params).
// ══════════════════════════════════════════════════════════

/// Parse an index out of the fabricated `"orbit_N"` strings that the
/// C ABI emits in event `orbit_id` fields (see
/// `empyrean-c/src/propagate.rs:512`). Returns `None` for anything
/// else so callers can fall back to the original string.
fn parse_fabricated_orbit_index(orbit_id: &str) -> Option<usize> {
    orbit_id.strip_prefix("orbit_").and_then(|s| s.parse().ok())
}

/// Build a `Vec<empyrean::Orbit>` from the same flat-arrays input
/// shape `_propagate` already accepts. Factored out so the IP /
/// B-plane wrappers don't duplicate ~50 lines of orbit assembly.
fn build_orbits_from_arrays(
    epochs: &numpy::ndarray::Array1<f64>,
    elements: &numpy::ndarray::Array2<f64>,
    covariances: &numpy::ndarray::Array3<f64>,
    has_covariance: &numpy::ndarray::Array1<bool>,
    representations: &numpy::ndarray::Array1<i32>,
    frames: &numpy::ndarray::Array1<i32>,
    origins: &numpy::ndarray::Array1<i32>,
    a1s: &numpy::ndarray::Array1<f64>,
    a2s: &numpy::ndarray::Array1<f64>,
    a3s: &numpy::ndarray::Array1<f64>,
    ng_alphas: Option<&numpy::ndarray::Array1<f64>>,
    ng_r0s: Option<&numpy::ndarray::Array1<f64>>,
    ng_ms: Option<&numpy::ndarray::Array1<f64>>,
    ng_ns: Option<&numpy::ndarray::Array1<f64>>,
    ng_ks: Option<&numpy::ndarray::Array1<f64>>,
    non_grav_dts: Option<&numpy::ndarray::Array1<f64>>,
) -> PyResult<Vec<empyrean::Orbit>> {
    let n = epochs.len();
    let mut orbits: Vec<empyrean::Orbit> = Vec::with_capacity(n);
    for i in 0..n {
        let mut elems = [0.0f64; 6];
        for j in 0..6 {
            elems[j] = elements[[i, j]];
        }
        let covariance = if has_covariance[i] {
            let mut cov = [[0.0f64; 6]; 6];
            for r in 0..6 {
                for c in 0..6 {
                    cov[r][c] = covariances[[i, r, c]];
                }
            }
            Some(cov)
        } else {
            None
        };
        let state = empyrean::CoordinateState {
            epoch: empyrean::Epoch::from_mjd_tdb(epochs[i]),
            elements: elems,
            covariance,
            representation: empyrean::int_to_rep(representations[i]).map_err(to_pyerr)?,
            frame: empyrean::int_to_frame(frames[i]).map_err(to_pyerr)?,
            origin: origin_from_naif(origins[i])?,
        };
        let mut orbit = empyrean::Orbit::new(state);
        orbit = orbit.with_nongrav(a1s[i], a2s[i], a3s[i]);
        if let (Some(a), Some(r), Some(m), Some(n_), Some(k)) =
            (ng_alphas, ng_r0s, ng_ms, ng_ns, ng_ks)
        {
            orbit = orbit.with_g_function(a[i], r[i], m[i], n_[i], k[i]);
        }
        if let Some(dts) = non_grav_dts {
            let dt = dts[i];
            if dt.is_finite() {
                orbit = orbit.with_non_grav_dt(Some(dt));
            }
        }
        orbits.push(orbit);
    }
    Ok(orbits)
}

fn methods_from_tags(tags: &[i32]) -> PyResult<Vec<empyrean::UncertaintyMethod>> {
    let mut out = Vec::with_capacity(tags.len());
    for &t in tags {
        out.push(match t {
            0 => empyrean::UncertaintyMethod::FirstOrder,
            1 => empyrean::UncertaintyMethod::SecondOrder,
            2 => empyrean::UncertaintyMethod::sigma_point(),
            3 => empyrean::UncertaintyMethod::monte_carlo(1000),
            4 => empyrean::UncertaintyMethod::auto(),
            other => {
                return Err(PyRuntimeError::new_err(format!(
                    "unknown uncertainty method tag: {other} (expected 0..=4)"
                )));
            }
        });
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (
    epochs, elements, covariances, has_covariance, representations, frames, origins,
    end_mjd_tdb,
    a1s, a2s, a3s,
    method_tags,
    body_filter_naif=None,
    ng_alphas=None, ng_r0s=None, ng_ms=None, ng_ns=None, ng_ks=None,
    non_grav_dts=None,
))]
#[allow(clippy::too_many_arguments)]
fn _compute_impact_probabilities<'py>(
    py: Python<'py>,
    epochs: PyReadonlyArray1<'py, f64>,
    elements: PyReadonlyArray2<'py, f64>,
    covariances: PyReadonlyArray3<'py, f64>,
    has_covariance: PyReadonlyArray1<'py, bool>,
    representations: PyReadonlyArray1<'py, i32>,
    frames: PyReadonlyArray1<'py, i32>,
    origins: PyReadonlyArray1<'py, i32>,
    end_mjd_tdb: f64,
    a1s: PyReadonlyArray1<'py, f64>,
    a2s: PyReadonlyArray1<'py, f64>,
    a3s: PyReadonlyArray1<'py, f64>,
    method_tags: Vec<i32>,
    body_filter_naif: Option<Vec<i32>>,
    ng_alphas: Option<PyReadonlyArray1<'py, f64>>,
    ng_r0s: Option<PyReadonlyArray1<'py, f64>>,
    ng_ms: Option<PyReadonlyArray1<'py, f64>>,
    ng_ns: Option<PyReadonlyArray1<'py, f64>>,
    ng_ks: Option<PyReadonlyArray1<'py, f64>>,
    non_grav_dts: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let ctx = get_context()?;

    let orbits = build_orbits_from_arrays(
        &epochs.as_array().to_owned(),
        &elements.as_array().to_owned(),
        &covariances.as_array().to_owned(),
        &has_covariance.as_array().to_owned(),
        &representations.as_array().to_owned(),
        &frames.as_array().to_owned(),
        &origins.as_array().to_owned(),
        &a1s.as_array().to_owned(),
        &a2s.as_array().to_owned(),
        &a3s.as_array().to_owned(),
        ng_alphas.as_ref().map(|a| a.as_array().to_owned()).as_ref(),
        ng_r0s.as_ref().map(|a| a.as_array().to_owned()).as_ref(),
        ng_ms.as_ref().map(|a| a.as_array().to_owned()).as_ref(),
        ng_ns.as_ref().map(|a| a.as_array().to_owned()).as_ref(),
        ng_ks.as_ref().map(|a| a.as_array().to_owned()).as_ref(),
        non_grav_dts
            .as_ref()
            .map(|a| a.as_array().to_owned())
            .as_ref(),
    )?;
    let methods = methods_from_tags(&method_tags)?;
    let filter: Vec<empyrean::Origin> = body_filter_naif
        .unwrap_or_default()
        .into_iter()
        .map(origin_from_naif)
        .collect::<PyResult<_>>()?;

    let end_epoch = empyrean::Epoch::from_mjd_tdb(end_mjd_tdb);
    let records = py
        .detach(|| ctx.compute_impact_probabilities(&orbits, end_epoch, &methods, &filter))
        .map_err(to_pyerr)?;

    let n = records.len();
    let mut out_method = Array1::<i32>::zeros(n);
    let mut out_orbit_id: Vec<String> = Vec::with_capacity(n);
    let mut out_object_id: Vec<String> = Vec::with_capacity(n);
    let mut out_body: Vec<String> = Vec::with_capacity(n);
    let mut out_body_naif = Array1::<i32>::zeros(n);
    let mut out_epoch = Array1::<f64>::zeros(n);
    let mut out_miss_au = Array1::<f64>::zeros(n);
    let mut out_miss_km = Array1::<f64>::zeros(n);
    let mut out_eff_radius_au = Array1::<f64>::zeros(n);
    let mut out_eff_radius_km = Array1::<f64>::zeros(n);
    let mut out_sigma_au = Array1::<f64>::zeros(n);
    let mut out_sigma_km = Array1::<f64>::zeros(n);
    let mut out_ip_linear = Array1::<f64>::zeros(n);
    let mut out_v_rel = Array1::<f64>::zeros(n);
    let mut out_ip_second = Array1::<f64>::from_elem(n, f64::NAN);
    let mut out_nonlin = Array1::<f64>::from_elem(n, f64::NAN);
    let mut out_ip_agm = Array1::<f64>::from_elem(n, f64::NAN);
    let mut out_ip_mc = Array1::<f64>::from_elem(n, f64::NAN);
    let mut out_mc_n = Array1::<u64>::zeros(n);
    let mut out_mc_imp = Array1::<u64>::zeros(n);
    for (i, r) in records.iter().enumerate() {
        out_method[i] = match &r.method {
            empyrean::UncertaintyMethod::FirstOrder => 0,
            empyrean::UncertaintyMethod::SecondOrder => 1,
            empyrean::UncertaintyMethod::SigmaPoint { .. } => 2,
            empyrean::UncertaintyMethod::MonteCarlo { .. } => 3,
            empyrean::UncertaintyMethod::Auto { .. } => 4,
        };
        out_orbit_id.push(r.orbit_id.clone());
        out_object_id.push(r.object_id.clone());
        // The wrapper now carries `body: Origin`; emit the canonical
        // string and its NAIF id so existing Python schemas keep
        // round-tripping until they migrate to the typed Origin.
        out_body.push(r.body.to_string());
        out_body_naif[i] = r.body.naif_id();
        out_epoch[i] = r.epoch.mjd_tdb().map_err(to_pyerr)?;
        out_miss_au[i] = r.miss_distance_au;
        out_miss_km[i] = r.miss_distance_km;
        out_eff_radius_au[i] = r.effective_radius_au;
        out_eff_radius_km[i] = r.effective_radius_km;
        out_sigma_au[i] = r.sigma_distance_au;
        out_sigma_km[i] = r.sigma_distance_km;
        out_ip_linear[i] = r.ip_linear;
        out_v_rel[i] = r.relative_velocity_au_day;
        out_ip_second[i] = r.ip_second_order;
        out_nonlin[i] = r.nonlinearity;
        out_ip_agm[i] = r.ip_agm;
        out_ip_mc[i] = r.ip_mc;
        out_mc_n[i] = r.mc_n_samples;
        out_mc_imp[i] = r.mc_n_impacts;
    }

    let dict = PyDict::new(py);
    dict.set_item("method_tag", out_method.into_pyarray(py))?;
    dict.set_item("orbit_id", out_orbit_id)?;
    dict.set_item("object_id", out_object_id)?;
    dict.set_item("body", out_body)?;
    dict.set_item("body_naif_id", out_body_naif.into_pyarray(py))?;
    dict.set_item("epoch_mjd_tdb", out_epoch.into_pyarray(py))?;
    dict.set_item("miss_distance_au", out_miss_au.into_pyarray(py))?;
    dict.set_item("miss_distance_km", out_miss_km.into_pyarray(py))?;
    dict.set_item("effective_radius_au", out_eff_radius_au.into_pyarray(py))?;
    dict.set_item("effective_radius_km", out_eff_radius_km.into_pyarray(py))?;
    dict.set_item("sigma_distance_au", out_sigma_au.into_pyarray(py))?;
    dict.set_item("sigma_distance_km", out_sigma_km.into_pyarray(py))?;
    dict.set_item("ip_linear", out_ip_linear.into_pyarray(py))?;
    dict.set_item("relative_velocity_au_day", out_v_rel.into_pyarray(py))?;
    dict.set_item("ip_second_order", out_ip_second.into_pyarray(py))?;
    dict.set_item("nonlinearity", out_nonlin.into_pyarray(py))?;
    dict.set_item("ip_agm", out_ip_agm.into_pyarray(py))?;
    dict.set_item("ip_mc", out_ip_mc.into_pyarray(py))?;
    dict.set_item("mc_n_samples", out_mc_n.into_pyarray(py))?;
    dict.set_item("mc_n_impacts", out_mc_imp.into_pyarray(py))?;
    Ok(dict)
}

#[pyfunction]
#[pyo3(signature = (
    epochs, elements, covariances, has_covariance, representations, frames, origins,
    end_mjd_tdb,
    a1s, a2s, a3s,
    method_tags,
    body_filter_naif=None,
    ng_alphas=None, ng_r0s=None, ng_ms=None, ng_ns=None, ng_ks=None,
    non_grav_dts=None,
))]
#[allow(clippy::too_many_arguments)]
fn _compute_b_planes<'py>(
    py: Python<'py>,
    epochs: PyReadonlyArray1<'py, f64>,
    elements: PyReadonlyArray2<'py, f64>,
    covariances: PyReadonlyArray3<'py, f64>,
    has_covariance: PyReadonlyArray1<'py, bool>,
    representations: PyReadonlyArray1<'py, i32>,
    frames: PyReadonlyArray1<'py, i32>,
    origins: PyReadonlyArray1<'py, i32>,
    end_mjd_tdb: f64,
    a1s: PyReadonlyArray1<'py, f64>,
    a2s: PyReadonlyArray1<'py, f64>,
    a3s: PyReadonlyArray1<'py, f64>,
    method_tags: Vec<i32>,
    body_filter_naif: Option<Vec<i32>>,
    ng_alphas: Option<PyReadonlyArray1<'py, f64>>,
    ng_r0s: Option<PyReadonlyArray1<'py, f64>>,
    ng_ms: Option<PyReadonlyArray1<'py, f64>>,
    ng_ns: Option<PyReadonlyArray1<'py, f64>>,
    ng_ks: Option<PyReadonlyArray1<'py, f64>>,
    non_grav_dts: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let ctx = get_context()?;

    let orbits = build_orbits_from_arrays(
        &epochs.as_array().to_owned(),
        &elements.as_array().to_owned(),
        &covariances.as_array().to_owned(),
        &has_covariance.as_array().to_owned(),
        &representations.as_array().to_owned(),
        &frames.as_array().to_owned(),
        &origins.as_array().to_owned(),
        &a1s.as_array().to_owned(),
        &a2s.as_array().to_owned(),
        &a3s.as_array().to_owned(),
        ng_alphas.as_ref().map(|a| a.as_array().to_owned()).as_ref(),
        ng_r0s.as_ref().map(|a| a.as_array().to_owned()).as_ref(),
        ng_ms.as_ref().map(|a| a.as_array().to_owned()).as_ref(),
        ng_ns.as_ref().map(|a| a.as_array().to_owned()).as_ref(),
        ng_ks.as_ref().map(|a| a.as_array().to_owned()).as_ref(),
        non_grav_dts
            .as_ref()
            .map(|a| a.as_array().to_owned())
            .as_ref(),
    )?;
    let methods = methods_from_tags(&method_tags)?;
    let filter: Vec<empyrean::Origin> = body_filter_naif
        .unwrap_or_default()
        .into_iter()
        .map(origin_from_naif)
        .collect::<PyResult<_>>()?;

    let end_epoch = empyrean::Epoch::from_mjd_tdb(end_mjd_tdb);
    let records = py
        .detach(|| ctx.compute_b_planes(&orbits, end_epoch, &methods, &filter))
        .map_err(to_pyerr)?;

    let n = records.len();
    let mut out_method = Array1::<i32>::zeros(n);
    let mut out_body: Vec<String> = Vec::with_capacity(n);
    let mut out_epoch = Array1::<f64>::zeros(n);
    let mut out_b_dot_t = Array1::<f64>::zeros(n);
    let mut out_b_dot_r = Array1::<f64>::zeros(n);
    let mut out_b_mag = Array1::<f64>::zeros(n);
    let mut out_v_inf = Array1::<f64>::zeros(n);
    let mut out_eff_radius = Array1::<f64>::zeros(n);
    let mut out_body_radius = Array1::<f64>::zeros(n);
    let mut out_cov_tt = Array1::<f64>::from_elem(n, f64::NAN);
    let mut out_cov_tr = Array1::<f64>::from_elem(n, f64::NAN);
    let mut out_cov_rr = Array1::<f64>::from_elem(n, f64::NAN);
    let mut out_smaj = Array1::<f64>::from_elem(n, f64::NAN);
    let mut out_smin = Array1::<f64>::from_elem(n, f64::NAN);
    let mut out_angle = Array1::<f64>::from_elem(n, f64::NAN);
    let mut out_ip_linear = Array1::<f64>::from_elem(n, f64::NAN);
    for (i, r) in records.iter().enumerate() {
        out_method[i] = match &r.method {
            empyrean::UncertaintyMethod::FirstOrder => 0,
            empyrean::UncertaintyMethod::SecondOrder => 1,
            empyrean::UncertaintyMethod::SigmaPoint { .. } => 2,
            empyrean::UncertaintyMethod::MonteCarlo { .. } => 3,
            empyrean::UncertaintyMethod::Auto { .. } => 4,
        };
        out_body.push(r.body.to_string());
        out_epoch[i] = r.epoch.mjd_tdb().map_err(to_pyerr)?;
        out_b_dot_t[i] = r.b_dot_t_km;
        out_b_dot_r[i] = r.b_dot_r_km;
        out_b_mag[i] = r.b_mag_km;
        out_v_inf[i] = r.v_inf_km_s;
        out_eff_radius[i] = r.effective_radius_km;
        out_body_radius[i] = r.body_radius_km;
        out_cov_tt[i] = r.cov_b_plane[0];
        out_cov_tr[i] = r.cov_b_plane[1];
        out_cov_rr[i] = r.cov_b_plane[2];
        out_smaj[i] = r.semi_major_3sig_km;
        out_smin[i] = r.semi_minor_3sig_km;
        out_angle[i] = r.ellipse_angle_rad;
        out_ip_linear[i] = r.ip_linear;
    }

    let dict = PyDict::new(py);
    dict.set_item("method_tag", out_method.into_pyarray(py))?;
    dict.set_item("body", out_body)?;
    dict.set_item("epoch_mjd_tdb", out_epoch.into_pyarray(py))?;
    dict.set_item("b_dot_t_km", out_b_dot_t.into_pyarray(py))?;
    dict.set_item("b_dot_r_km", out_b_dot_r.into_pyarray(py))?;
    dict.set_item("b_mag_km", out_b_mag.into_pyarray(py))?;
    dict.set_item("v_inf_km_s", out_v_inf.into_pyarray(py))?;
    dict.set_item("effective_radius_km", out_eff_radius.into_pyarray(py))?;
    dict.set_item("body_radius_km", out_body_radius.into_pyarray(py))?;
    dict.set_item("cov_tt_km2", out_cov_tt.into_pyarray(py))?;
    dict.set_item("cov_tr_km2", out_cov_tr.into_pyarray(py))?;
    dict.set_item("cov_rr_km2", out_cov_rr.into_pyarray(py))?;
    dict.set_item("semi_major_3sig_km", out_smaj.into_pyarray(py))?;
    dict.set_item("semi_minor_3sig_km", out_smin.into_pyarray(py))?;
    dict.set_item("ellipse_angle_rad", out_angle.into_pyarray(py))?;
    dict.set_item("ip_linear", out_ip_linear.into_pyarray(py))?;
    Ok(dict)
}

// ══════════════════════════════════════════════════════════
//  _get_observers
// ══════════════════════════════════════════════════════════

#[pyfunction]
#[pyo3(signature = (obs_codes, epochs_mjd_tdb))]
fn _get_observers<'py>(
    py: Python<'py>,
    obs_codes: Vec<String>,
    epochs_mjd_tdb: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let ctx = get_context()?;
    let epochs_arr = epochs_mjd_tdb.as_array().to_owned();
    let code_refs: Vec<&str> = obs_codes.iter().map(|s| s.as_str()).collect();
    let epochs_vec: Vec<empyrean::Epoch> = epochs_arr
        .iter()
        .map(|&t| empyrean::Epoch::from_mjd_tdb(t))
        .collect();

    let observers = py.detach(|| ctx.get_observers(&code_refs, &epochs_vec).map_err(to_pyerr))?;

    let total = observers.len();
    let mut out_codes: Vec<String> = Vec::with_capacity(total);
    let mut out_epochs = Array1::<f64>::zeros(total);
    let mut out_x = Array1::<f64>::zeros(total);
    let mut out_y = Array1::<f64>::zeros(total);
    let mut out_z = Array1::<f64>::zeros(total);
    let mut out_vx = Array1::<f64>::zeros(total);
    let mut out_vy = Array1::<f64>::zeros(total);
    let mut out_vz = Array1::<f64>::zeros(total);
    let mut out_nights = Array1::<i32>::zeros(total);

    for (i, obs) in observers.iter().enumerate() {
        out_codes.push(obs.obs_code.clone());
        out_epochs[i] = obs.epoch.mjd_tdb().map_err(to_pyerr)?;
        out_x[i] = obs.position[0];
        out_y[i] = obs.position[1];
        out_z[i] = obs.position[2];
        out_vx[i] = obs.velocity[0];
        out_vy[i] = obs.velocity[1];
        out_vz[i] = obs.velocity[2];
        out_nights[i] = obs.observing_night;
    }

    let dict = PyDict::new(py);
    dict.set_item("obs_code", out_codes)?;
    dict.set_item("epoch", PyArray1::from_owned_array(py, out_epochs))?;
    dict.set_item("x", PyArray1::from_owned_array(py, out_x))?;
    dict.set_item("y", PyArray1::from_owned_array(py, out_y))?;
    dict.set_item("z", PyArray1::from_owned_array(py, out_z))?;
    dict.set_item("vx", PyArray1::from_owned_array(py, out_vx))?;
    dict.set_item("vy", PyArray1::from_owned_array(py, out_vy))?;
    dict.set_item("vz", PyArray1::from_owned_array(py, out_vz))?;
    dict.set_item(
        "observing_night",
        PyArray1::from_owned_array(py, out_nights),
    )?;
    Ok(dict)
}

// ══════════════════════════════════════════════════════════
//  _generate_ephemeris
// ══════════════════════════════════════════════════════════

#[pyfunction]
#[pyo3(signature = (
    orbit_ids,
    object_ids,
    epochs,
    elements,
    covariances,
    has_covariance,
    representations,
    frames,
    origins,
    a1s,
    a2s,
    a3s,
    phot_h,
    phot_slope1,
    phot_system,
    obs_codes,
    obs_epochs,
    obs_x,
    obs_y,
    obs_z,
    obs_vx,
    obs_vy,
    obs_vz,
    force_model,
    epsilon = None,
    uncertainty_method = 0,
    ng_alphas = None,
    ng_r0s = None,
    ng_ms = None,
    ng_ns = None,
    ng_ks = None,
    non_grav_dts = None,
    phot_slope2 = None,
    gm_threshold = 1.0,
    gm_max_depth = 3,
    gm_components_per_split = 3,
    sigma_n_sigma = 1.0,
    sigma_samples_per_plane = 8,
    mc_n_samples = 1000,
    mc_seed = None,
    ephemeris_config_dict = None,
))]
fn _generate_ephemeris<'py>(
    py: Python<'py>,
    orbit_ids: Vec<String>,
    object_ids: Vec<String>,
    epochs: PyReadonlyArray1<'py, f64>,
    elements: PyReadonlyArray2<'py, f64>,
    covariances: PyReadonlyArray3<'py, f64>,
    has_covariance: PyReadonlyArray1<'py, bool>,
    representations: PyReadonlyArray1<'py, i32>,
    frames: PyReadonlyArray1<'py, i32>,
    origins: PyReadonlyArray1<'py, i32>,
    a1s: PyReadonlyArray1<'py, f64>,
    a2s: PyReadonlyArray1<'py, f64>,
    a3s: PyReadonlyArray1<'py, f64>,
    phot_h: PyReadonlyArray1<'py, f64>,
    phot_slope1: PyReadonlyArray1<'py, f64>,
    phot_system: PyReadonlyArray1<'py, i32>,
    obs_codes: Vec<String>,
    obs_epochs: PyReadonlyArray1<'py, f64>,
    obs_x: PyReadonlyArray1<'py, f64>,
    obs_y: PyReadonlyArray1<'py, f64>,
    obs_z: PyReadonlyArray1<'py, f64>,
    obs_vx: PyReadonlyArray1<'py, f64>,
    obs_vy: PyReadonlyArray1<'py, f64>,
    obs_vz: PyReadonlyArray1<'py, f64>,
    force_model: i32,
    epsilon: Option<f64>,
    uncertainty_method: i32,
    ng_alphas: Option<PyReadonlyArray1<'py, f64>>,
    ng_r0s: Option<PyReadonlyArray1<'py, f64>>,
    ng_ms: Option<PyReadonlyArray1<'py, f64>>,
    ng_ns: Option<PyReadonlyArray1<'py, f64>>,
    ng_ks: Option<PyReadonlyArray1<'py, f64>>,
    non_grav_dts: Option<PyReadonlyArray1<'py, f64>>,
    phot_slope2: Option<PyReadonlyArray1<'py, f64>>,
    gm_threshold: f64,
    gm_max_depth: usize,
    gm_components_per_split: usize,
    sigma_n_sigma: f64,
    sigma_samples_per_plane: usize,
    mc_n_samples: usize,
    mc_seed: Option<u64>,
    ephemeris_config_dict: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = (
        epsilon,
        uncertainty_method,
        gm_threshold,
        gm_max_depth,
        gm_components_per_split,
        sigma_n_sigma,
        sigma_samples_per_plane,
        mc_n_samples,
        mc_seed,
    );
    let ctx = get_context()?;

    let epochs_arr = epochs.as_array().to_owned();
    let elements_arr = elements.as_array().to_owned();
    let covariances_arr = covariances.as_array().to_owned();
    let has_cov_arr = has_covariance.as_array().to_owned();
    let reps_arr = representations.as_array().to_owned();
    let frames_arr = frames.as_array().to_owned();
    let origins_arr = origins.as_array().to_owned();
    let a1s_arr = a1s.as_array().to_owned();
    let a2s_arr = a2s.as_array().to_owned();
    let a3s_arr = a3s.as_array().to_owned();
    let phot_h_arr = phot_h.as_array().to_owned();
    let phot_slope1_arr = phot_slope1.as_array().to_owned();
    let phot_system_arr = phot_system.as_array().to_owned();
    let phot_slope2_arr = phot_slope2.as_ref().map(|a| a.as_array().to_owned());
    let ng_alpha_arr = ng_alphas.as_ref().map(|a| a.as_array().to_owned());
    let ng_r0_arr = ng_r0s.as_ref().map(|a| a.as_array().to_owned());
    let ng_m_arr = ng_ms.as_ref().map(|a| a.as_array().to_owned());
    let ng_n_arr = ng_ns.as_ref().map(|a| a.as_array().to_owned());
    let ng_k_arr = ng_ks.as_ref().map(|a| a.as_array().to_owned());
    let dt_arr = non_grav_dts.as_ref().map(|a| a.as_array().to_owned());

    let obs_epochs_arr = obs_epochs.as_array().to_owned();
    let obs_x_arr = obs_x.as_array().to_owned();
    let obs_y_arr = obs_y.as_array().to_owned();
    let obs_z_arr = obs_z.as_array().to_owned();
    let obs_vx_arr = obs_vx.as_array().to_owned();
    let obs_vy_arr = obs_vy.as_array().to_owned();
    let obs_vz_arr = obs_vz.as_array().to_owned();

    let n = epochs_arr.len();
    let mut orbits: Vec<empyrean::Orbit> = Vec::with_capacity(n);
    for i in 0..n {
        let mut elems = [0.0f64; 6];
        for j in 0..6 {
            elems[j] = elements_arr[[i, j]];
        }
        let covariance = if has_cov_arr[i] {
            let mut cov = [[0.0f64; 6]; 6];
            for r in 0..6 {
                for c in 0..6 {
                    cov[r][c] = covariances_arr[[i, r, c]];
                }
            }
            Some(cov)
        } else {
            None
        };
        let state = empyrean::CoordinateState {
            epoch: empyrean::Epoch::from_mjd_tdb(epochs_arr[i]),
            elements: elems,
            covariance,
            representation: empyrean::int_to_rep(reps_arr[i]).map_err(to_pyerr)?,
            frame: empyrean::int_to_frame(frames_arr[i]).map_err(to_pyerr)?,
            origin: origin_from_naif(origins_arr[i])?,
        };
        let mut orbit =
            empyrean::Orbit::new(state).with_nongrav(a1s_arr[i], a2s_arr[i], a3s_arr[i]);
        if let (Some(a), Some(r), Some(m), Some(n), Some(k)) =
            (&ng_alpha_arr, &ng_r0_arr, &ng_m_arr, &ng_n_arr, &ng_k_arr)
        {
            orbit = orbit.with_g_function(a[i], r[i], m[i], n[i], k[i]);
        }
        if let Some(dts) = &dt_arr {
            let dt = dts[i];
            if dt.is_finite() {
                orbit = orbit.with_non_grav_dt(Some(dt));
            }
        }
        // Photometry — when phot_system[i] is one of {0, 1, 2} the
        // ephemeris pipeline produces apparent magnitude from H +
        // slope params. -1 (the absent sentinel) leaves
        // `phot_system = None` and the row's mag = NaN.
        let pf_int = phot_system_arr[i];
        let h = phot_h_arr[i];
        let g = phot_slope1_arr[i];
        if h.is_finite() && pf_int >= 0 {
            let pf = match pf_int {
                0 => empyrean::PhaseFunction::HG,
                1 => empyrean::PhaseFunction::HG1G2,
                2 => empyrean::PhaseFunction::HG12,
                _ => empyrean::PhaseFunction::HG,
            };
            // The ephemeris path only consumes (H, slope1, slope2);
            // for HG and HG12 we feed slope1=g, slope2=0, which the
            // wrapper records and the C ABI translates to the right
            // PhotometricParams::* constructor on the upstream side.
            // HG1G2 uses G2 in slot 2; HG and HG12 ignore it. The
            // caller-supplied phot_slope2 array carries G2 when set;
            // otherwise default to 0.0 (which is correct for HG/HG12
            // and a wrong-but-non-crashing fallback for HG1G2 — file
            // a clear array if HG1G2 fits matter).
            let s2 = phot_slope2_arr.as_ref().map_or(0.0, |a| a[i]);
            orbit = orbit.with_photometry(pf, h, g, s2);
        }
        orbits.push(orbit);
    }

    let n_obs = obs_epochs_arr.len();
    let mut observers: Vec<empyrean::Observer> = Vec::with_capacity(n_obs);
    for i in 0..n_obs {
        // Recompute the observer fully through `ctx.get_observers` so we
        // get the same `observing_night` and any internal rounding the
        // rust validation runner sees. Falls back to caller-supplied
        // arrays if the lookup fails.
        let (pos, vel, night) = match ctx.get_observers(
            &[obs_codes[i].as_str()],
            &[empyrean::Epoch::from_mjd_tdb(obs_epochs_arr[i])],
        ) {
            Ok(v) if !v.is_empty() => (v[0].position, v[0].velocity, v[0].observing_night),
            _ => (
                [obs_x_arr[i], obs_y_arr[i], obs_z_arr[i]],
                [obs_vx_arr[i], obs_vy_arr[i], obs_vz_arr[i]],
                -1,
            ),
        };
        observers.push(empyrean::Observer {
            obs_code: obs_codes[i].clone(),
            epoch: empyrean::Epoch::from_mjd_tdb(obs_epochs_arr[i]),
            position: pos,
            velocity: vel,
            observing_night: night,
        });
    }

    let force_model_tier = match force_model {
        0 => empyrean::ForceModelTier::Approximate,
        1 => empyrean::ForceModelTier::Basic,
        2 => empyrean::ForceModelTier::Standard,
        _ => {
            return Err(PyRuntimeError::new_err(format!(
                "unknown or unsupported force model tier: {force_model}"
            )));
        }
    };

    let mut eph_config = empyrean::EphemerisConfig::with_force_model(force_model_tier);
    if let Some(d) = ephemeris_config_dict {
        eph_config = build_ephemeris_config_from_dict(d)?;
        // Honor the per-call force-model selection: villeneuve's
        // ephemeris pipeline assumes EclipticJ2000 for the integration
        // frame regardless of the user-facing output frame, so we
        // preserve that override even when the caller supplied a full
        // config dict.
        eph_config.propagation.force_model = force_model_tier;
        eph_config.propagation.frame = empyrean::Frame::EclipticJ2000;
    }
    let eph_result = py
        .detach(|| ctx.generate_ephemeris(&orbits, &observers, &eph_config))
        .map_err(to_pyerr)?;
    let entries = &eph_result.entries;

    let m = entries.len();
    let mut out_orbit_ids: Vec<String> = Vec::with_capacity(m);
    let mut out_object_ids: Vec<String> = Vec::with_capacity(m);
    let mut out_epochs_arr = Array1::<f64>::zeros(m);
    let mut out_ra = Array1::<f64>::zeros(m);
    let mut out_dec = Array1::<f64>::zeros(m);
    let mut out_rho = Array1::<f64>::zeros(m);
    let mut out_vrho = Array1::<f64>::zeros(m);
    let mut out_vra = Array1::<f64>::zeros(m);
    let mut out_vdec = Array1::<f64>::zeros(m);
    let mut out_light_time = Array1::<f64>::zeros(m);
    let mut out_phase_angle = Array1::<f64>::zeros(m);
    let mut out_elongation = Array1::<f64>::zeros(m);
    let mut out_helio_dist = Array1::<f64>::zeros(m);
    let mut out_app_mag = Array1::<f64>::zeros(m);
    let mut out_mag_unc = Array1::<f64>::zeros(m);
    let mut out_zenith = Array1::<f64>::zeros(m);
    let mut out_azimuth = Array1::<f64>::zeros(m);
    let mut out_hour_angle = Array1::<f64>::zeros(m);
    let mut out_lunar_elong = Array1::<f64>::zeros(m);
    let mut out_position_angle = Array1::<f64>::zeros(m);
    let mut out_sky_rate = Array1::<f64>::zeros(m);
    let mut out_obs_codes_v: Vec<String> = Vec::with_capacity(m);

    // Same mechanism as in `_propagate`'s events output: the C ABI
    // fabricates per-entry `orbit_id` as `"orbit_{i}"` because
    // `EmpyreanOrbit` carries no orbit_id field. Recover the user's
    // orbit_id and object_id by parsing the index and looking it up
    // in the input arrays. (TODO empyrean-3ud6: drop this once the
    // C struct carries orbit_id / object_id directly.)
    for (i, e) in entries.iter().enumerate() {
        let user_idx = parse_fabricated_orbit_index(&e.orbit_id);
        let orbit_id = user_idx
            .and_then(|j| orbit_ids.get(j).cloned())
            .unwrap_or_else(|| e.orbit_id.clone());
        let object_id = user_idx
            .and_then(|j| object_ids.get(j).cloned())
            .unwrap_or_default();
        out_orbit_ids.push(orbit_id);
        out_object_ids.push(object_id);
        out_epochs_arr[i] = e.epoch.mjd_tdb().map_err(to_pyerr)?;
        out_ra[i] = e.ra_deg;
        out_dec[i] = e.dec_deg;
        out_rho[i] = e.rho_au;
        out_vrho[i] = e.vrho_au_day;
        out_vra[i] = e.vra_deg_day;
        out_vdec[i] = e.vdec_deg_day;
        out_light_time[i] = e.light_time_days;
        out_phase_angle[i] = e.phase_angle_deg;
        out_elongation[i] = e.elongation_deg;
        out_helio_dist[i] = e.heliocentric_distance_au;
        out_app_mag[i] = e.mag;
        out_mag_unc[i] = e.mag_sigma;
        out_zenith[i] = e.zenith_angle_deg;
        out_azimuth[i] = e.azimuth_deg;
        out_hour_angle[i] = e.hour_angle_deg;
        out_lunar_elong[i] = e.lunar_elongation_deg;
        out_position_angle[i] = e.position_angle_deg;
        out_sky_rate[i] = e.sky_rate_deg_day;
        out_obs_codes_v.push(e.obs_code.clone());
    }

    let dict = PyDict::new(py);
    dict.set_item("orbit_id", out_orbit_ids)?;
    dict.set_item("object_id", out_object_ids)?;
    dict.set_item("epoch", PyArray1::from_owned_array(py, out_epochs_arr))?;
    dict.set_item("ra", PyArray1::from_owned_array(py, out_ra))?;
    dict.set_item("dec", PyArray1::from_owned_array(py, out_dec))?;
    dict.set_item("rho", PyArray1::from_owned_array(py, out_rho))?;
    dict.set_item("vrho", PyArray1::from_owned_array(py, out_vrho))?;
    dict.set_item("vra", PyArray1::from_owned_array(py, out_vra))?;
    dict.set_item("vdec", PyArray1::from_owned_array(py, out_vdec))?;
    dict.set_item("light_time", PyArray1::from_owned_array(py, out_light_time))?;
    dict.set_item(
        "phase_angle",
        PyArray1::from_owned_array(py, out_phase_angle),
    )?;
    dict.set_item("elongation", PyArray1::from_owned_array(py, out_elongation))?;
    dict.set_item(
        "heliocentric_distance",
        PyArray1::from_owned_array(py, out_helio_dist),
    )?;
    dict.set_item("mag", PyArray1::from_owned_array(py, out_app_mag))?;
    dict.set_item("mag_sigma", PyArray1::from_owned_array(py, out_mag_unc))?;
    dict.set_item("zenith_angle", PyArray1::from_owned_array(py, out_zenith))?;
    dict.set_item("azimuth", PyArray1::from_owned_array(py, out_azimuth))?;
    dict.set_item("hour_angle", PyArray1::from_owned_array(py, out_hour_angle))?;
    dict.set_item(
        "lunar_elongation",
        PyArray1::from_owned_array(py, out_lunar_elong),
    )?;
    dict.set_item(
        "position_angle",
        PyArray1::from_owned_array(py, out_position_angle),
    )?;
    dict.set_item("sky_rate", PyArray1::from_owned_array(py, out_sky_rate))?;
    dict.set_item("obs_code", out_obs_codes_v)?;

    // ── Observation sensitivities (bd empyrean-14cz.4) — one row per
    // (orbit, observer, epoch). jacobian/hessian are row-major-flattened
    // (6×n_params / 6×n_params²); hessian is None unless a second-order
    // method ran. Empty on the f64-only path. ──
    let ns = eph_result.sensitivity.len();
    let mut s_orbit_id: Vec<String> = Vec::with_capacity(ns);
    let mut s_object_id: Vec<Option<String>> = Vec::with_capacity(ns);
    let mut s_obs_code: Vec<String> = Vec::with_capacity(ns);
    let mut s_epoch: Vec<f64> = Vec::with_capacity(ns);
    // u32 (not u8) — PyO3 maps Vec<u8> to Python `bytes`, not a list of ints.
    let mut s_n_params: Vec<u32> = Vec::with_capacity(ns);
    let mut s_jacobian: Vec<Option<Vec<f64>>> = Vec::with_capacity(ns);
    let mut s_hessian: Vec<Option<Vec<f64>>> = Vec::with_capacity(ns);
    for row in &eph_result.sensitivity {
        s_orbit_id.push(row.orbit_id.clone());
        s_object_id.push(row.object_id.clone());
        s_obs_code.push(row.obs_code.clone());
        s_epoch.push(row.epoch_mjd_tdb);
        s_n_params.push(row.n_params as u32);
        s_jacobian.push((!row.jacobian.is_empty()).then(|| row.jacobian.clone()));
        s_hessian.push((!row.hessian.is_empty()).then(|| row.hessian.clone()));
    }
    dict.set_item("sensitivity_orbit_id", s_orbit_id)?;
    dict.set_item("sensitivity_object_id", s_object_id)?;
    dict.set_item("sensitivity_obs_code", s_obs_code)?;
    dict.set_item("sensitivity_epoch_mjd_tdb", s_epoch)?;
    dict.set_item("sensitivity_n_params", s_n_params)?;
    dict.set_item("sensitivity_jacobian", s_jacobian)?;
    dict.set_item("sensitivity_hessian", s_hessian)?;

    Ok(dict)
}

// ══════════════════════════════════════════════════════════
//  _get_states
// ══════════════════════════════════════════════════════════

#[pyfunction]
#[pyo3(signature = (target_naif_id, center_naif_id, epochs_mjd_tdb, frame))]
fn _get_states<'py>(
    py: Python<'py>,
    target_naif_id: i32,
    center_naif_id: i32,
    epochs_mjd_tdb: PyReadonlyArray1<'py, f64>,
    frame: i32,
) -> PyResult<Bound<'py, PyDict>> {
    let ctx = get_context()?;
    let epochs_arr = epochs_mjd_tdb.as_array().to_owned();
    let epochs_vec: Vec<empyrean::Epoch> = epochs_arr
        .iter()
        .map(|&t| empyrean::Epoch::from_mjd_tdb(t))
        .collect();

    let target = origin_from_naif(target_naif_id)?;
    let center = origin_from_naif(center_naif_id)?;
    let e_frame = empyrean::int_to_frame(frame).map_err(to_pyerr)?;

    let states = py
        .detach(|| ctx.get_states(target, center, &epochs_vec, e_frame))
        .map_err(to_pyerr)?;

    let n = states.len();
    let mut out_epochs = Array1::<f64>::zeros(n);
    let mut out_x = Array1::<f64>::zeros(n);
    let mut out_y = Array1::<f64>::zeros(n);
    let mut out_z = Array1::<f64>::zeros(n);
    let mut out_vx = Array1::<f64>::zeros(n);
    let mut out_vy = Array1::<f64>::zeros(n);
    let mut out_vz = Array1::<f64>::zeros(n);
    let mut out_frames = Array1::<i32>::zeros(n);
    let mut out_origins = Array1::<i32>::zeros(n);

    for (i, state) in states.iter().enumerate() {
        out_epochs[i] = state.epoch.mjd_tdb().map_err(to_pyerr)?;
        out_x[i] = state.position[0];
        out_y[i] = state.position[1];
        out_z[i] = state.position[2];
        out_vx[i] = state.velocity[0];
        out_vy[i] = state.velocity[1];
        out_vz[i] = state.velocity[2];
        out_frames[i] = empyrean::frame_to_int(state.frame);
        out_origins[i] = state.origin.naif_id();
    }

    let dict = PyDict::new(py);
    dict.set_item("epoch", PyArray1::from_owned_array(py, out_epochs))?;
    dict.set_item("x", PyArray1::from_owned_array(py, out_x))?;
    dict.set_item("y", PyArray1::from_owned_array(py, out_y))?;
    dict.set_item("z", PyArray1::from_owned_array(py, out_z))?;
    dict.set_item("vx", PyArray1::from_owned_array(py, out_vx))?;
    dict.set_item("vy", PyArray1::from_owned_array(py, out_vy))?;
    dict.set_item("vz", PyArray1::from_owned_array(py, out_vz))?;
    dict.set_item("frame", PyArray1::from_owned_array(py, out_frames))?;
    dict.set_item("origin", PyArray1::from_owned_array(py, out_origins))?;
    Ok(dict)
}

// ══════════════════════════════════════════════════════════
//  Orbit determination helpers
// ══════════════════════════════════════════════════════════

fn nan_to_value(v: Option<f64>) -> f64 {
    v.unwrap_or(f64::NAN)
}

fn build_observations<'py>(
    ctx: &empyrean::Context,
    obs_dict: &Bound<'py, PyDict>,
) -> PyResult<empyrean::Observations> {
    // Optical-only entry point (evaluate / refine). The legacy ``ades``
    // key path carries radar internally; the flat-dict path is
    // optical-only and builds the radar array null/empty.
    if let Some(item) = obs_dict.get_item("ades")? {
        let ades: String = item.extract()?;
        return ctx.read_ades(&ades).map_err(to_pyerr);
    }
    let observations = build_observation_vec(obs_dict)?;
    empyrean::Observations::from_array(&observations).map_err(to_pyerr)
}

/// Build a `Vec<Observation>` from the flat-column optical dict shape.
///
/// Python supplies the deconstructed ADES fields (perm_id / prov_id /
/// stn / obs_time / ra / dec / rms_ra / rms_dec, plus the optional
/// supplementary columns). Splitting this out lets `_determine` combine
/// the optical vec with a radar vec via [`empyrean::Observations::from_arrays`]
/// without a PSV round-trip; `build_observations` (evaluate / refine)
/// uses it for the optical-only path.
fn build_observation_vec(obs_dict: &Bound<'_, PyDict>) -> PyResult<Vec<empyrean::Observation>> {
    fn req_strs(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<Option<String>>> {
        d.get_item(key)?
            .ok_or_else(|| PyRuntimeError::new_err(format!("missing '{key}'")))?
            .extract()
    }

    fn opt_strs(d: &Bound<'_, PyDict>, key: &str, n: usize) -> PyResult<Vec<Option<String>>> {
        match d.get_item(key)? {
            Some(obj) => obj.extract(),
            None => Ok(vec![None; n]),
        }
    }

    fn req_floats<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<PyReadonlyArray1<'py, f64>> {
        d.get_item(key)?
            .ok_or_else(|| PyRuntimeError::new_err(format!("missing '{key}'")))?
            .extract()
    }

    fn opt_floats<'py>(d: &Bound<'py, PyDict>, key: &str, n: usize) -> PyResult<Vec<f64>> {
        match d.get_item(key)? {
            Some(obj) => {
                let arr: PyReadonlyArray1<'_, f64> = obj.extract()?;
                Ok(arr.as_array().to_vec())
            }
            None => Ok(vec![f64::NAN; n]),
        }
    }

    let perm_ids = req_strs(obs_dict, "perm_id")?;
    let prov_ids = req_strs(obs_dict, "prov_id")?;
    let stns: Vec<String> = obs_dict
        .get_item("stn")?
        .ok_or_else(|| PyRuntimeError::new_err("missing 'stn'"))?
        .extract()?;
    let obs_times: Vec<String> = obs_dict
        .get_item("obs_time")?
        .ok_or_else(|| PyRuntimeError::new_err("missing 'obs_time'"))?
        .extract()?;
    let ra = req_floats(obs_dict, "ra")?;
    let dec = req_floats(obs_dict, "dec")?;
    let rms_ra = req_floats(obs_dict, "rms_ra")?;
    let rms_dec = req_floats(obs_dict, "rms_dec")?;

    let n = stns.len();
    let trk_subs = opt_strs(obs_dict, "trk_sub", n)?;
    let modes = opt_strs(obs_dict, "mode", n)?;
    let sys_v = opt_strs(obs_dict, "sys", n)?;
    let ast_cats = opt_strs(obs_dict, "ast_cat", n)?;
    let bands = opt_strs(obs_dict, "band", n)?;
    let phot_cats = opt_strs(obs_dict, "phot_cat", n)?;
    let obs_id_v = opt_strs(obs_dict, "obs_id", n)?;
    let obs_sub_id_v = opt_strs(obs_dict, "obs_sub_id", n)?;
    let trk_id_v = opt_strs(obs_dict, "trk_id", n)?;
    let progs = opt_strs(obs_dict, "prog", n)?;
    let notes_v = opt_strs(obs_dict, "notes", n)?;
    let remarks_v = opt_strs(obs_dict, "remarks", n)?;
    let ctr_v = opt_floats(obs_dict, "ctr", n)?;
    let pos1_v = opt_floats(obs_dict, "pos1", n)?;
    let pos2_v = opt_floats(obs_dict, "pos2", n)?;
    let pos3_v = opt_floats(obs_dict, "pos3", n)?;
    let rms_corr_v = opt_floats(obs_dict, "rms_corr", n)?;
    let mag_v = opt_floats(obs_dict, "mag", n)?;
    let rms_mag_v = opt_floats(obs_dict, "rms_mag", n)?;
    let phot_ap_v = opt_floats(obs_dict, "phot_ap", n)?;
    let log_snr_v = opt_floats(obs_dict, "log_snr", n)?;
    let seeing_v = opt_floats(obs_dict, "seeing", n)?;
    let exp_v = opt_floats(obs_dict, "exp", n)?;
    let rms_fit_v = opt_floats(obs_dict, "rms_fit", n)?;
    // n_stars: Option<u32> with -1 sentinel (i32 array on the wire)
    let n_stars_v: Vec<i32> = match obs_dict.get_item("n_stars")? {
        Some(obj) => {
            let arr: PyReadonlyArray1<'_, i32> = obj.extract()?;
            arr.as_array().to_vec()
        }
        None => vec![-1; n],
    };

    if perm_ids.len() != n
        || prov_ids.len() != n
        || obs_times.len() != n
        || ra.as_array().len() != n
        || dec.as_array().len() != n
        || rms_ra.as_array().len() != n
        || rms_dec.as_array().len() != n
    {
        return Err(PyRuntimeError::new_err(
            "observations dict has mismatched column lengths",
        ));
    }

    let ra = ra.as_array();
    let dec = dec.as_array();
    let rms_ra = rms_ra.as_array();
    let rms_dec = rms_dec.as_array();

    let nan_to_opt = |v: f64| if v.is_nan() { None } else { Some(v) };
    let s_to_opt = |s: Option<String>| s.filter(|x| !x.is_empty());

    let mut observations = Vec::with_capacity(n);
    for i in 0..n {
        observations.push(empyrean::Observation {
            perm_id: s_to_opt(perm_ids[i].clone()),
            prov_id: s_to_opt(prov_ids[i].clone()),
            trk_sub: s_to_opt(trk_subs[i].clone()),
            obs_id: s_to_opt(obs_id_v[i].clone()),
            obs_sub_id: s_to_opt(obs_sub_id_v[i].clone()),
            trk_id: s_to_opt(trk_id_v[i].clone()),
            obs_code: stns[i].clone(),
            mode: s_to_opt(modes[i].clone()),
            prog: s_to_opt(progs[i].clone()),
            sys: s_to_opt(sys_v[i].clone()),
            ctr: nan_to_opt(ctr_v[i]),
            pos1: nan_to_opt(pos1_v[i]),
            pos2: nan_to_opt(pos2_v[i]),
            pos3: nan_to_opt(pos3_v[i]),
            obs_time: obs_times[i].clone(),
            ra_deg: ra[i],
            dec_deg: dec[i],
            rms_ra_arcsec: rms_ra[i],
            rms_dec_arcsec: rms_dec[i],
            rms_corr: nan_to_opt(rms_corr_v[i]),
            ast_cat: s_to_opt(ast_cats[i].clone()),
            mag: nan_to_opt(mag_v[i]),
            rms_mag: nan_to_opt(rms_mag_v[i]),
            band: s_to_opt(bands[i].clone()),
            phot_cat: s_to_opt(phot_cats[i].clone()),
            phot_ap: nan_to_opt(phot_ap_v[i]),
            log_snr: nan_to_opt(log_snr_v[i]),
            seeing: nan_to_opt(seeing_v[i]),
            exp: nan_to_opt(exp_v[i]),
            rms_fit: nan_to_opt(rms_fit_v[i]),
            n_stars: if n_stars_v[i] >= 0 {
                Some(n_stars_v[i] as u32)
            } else {
                None
            },
            notes: s_to_opt(notes_v[i].clone()),
            remarks: s_to_opt(remarks_v[i].clone()),
        });
    }
    Ok(observations)
}

/// Build a `Vec<RadarObservation>` from a Python radar dict — the radar
/// analogue of [`build_observations`].
///
/// The dict is the flat-column shape produced by `_radar_to_dict` on the
/// Python side: string column lists (`perm_id`/`prov_id`/`trk_sub`/`trx`/
/// `rcv`/`obs_time`/`observable`/`remarks`) plus float arrays (`delay`/
/// `rms_delay`/`doppler`/`rms_doppler`/`frq`/`log_snr`) and an i8 `com`
/// array. All values are ADES-native — delay in **s**, rmsDelay in **µs**,
/// doppler/rmsDoppler in **Hz**, frq in **MHz** — and no conversion is
/// applied here.
///
/// The `observable` discriminator (`"delay"` / `"doppler"`) — *not* a NaN
/// probe — selects the [`RadarMeasurement`] variant, so a genuine 0.0-Hz
/// Doppler is never confused with an absent one. An absent or empty radar
/// dict (`None`, or no `obs_time`) yields an empty `Vec`.
fn build_radar_observations(
    radar_dict: Option<&Bound<'_, PyDict>>,
) -> PyResult<Vec<empyrean::RadarObservation>> {
    let Some(radar_dict) = radar_dict else {
        return Ok(Vec::new());
    };

    fn req_strs(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<Option<String>>> {
        d.get_item(key)?
            .ok_or_else(|| PyRuntimeError::new_err(format!("missing radar '{key}'")))?
            .extract()
    }
    fn opt_floats(d: &Bound<'_, PyDict>, key: &str, n: usize) -> PyResult<Vec<f64>> {
        match d.get_item(key)? {
            Some(obj) => {
                let arr: PyReadonlyArray1<'_, f64> = obj.extract()?;
                Ok(arr.as_array().to_vec())
            }
            None => Ok(vec![f64::NAN; n]),
        }
    }

    // `obs_time` is the canonical length anchor (always present per row).
    // An absent `obs_time` key means "no radar table" → empty Vec.
    let obs_times: Vec<String> = match radar_dict.get_item("obs_time")? {
        Some(obj) => obj.extract()?,
        None => return Ok(Vec::new()),
    };
    let n = obs_times.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let observables: Vec<String> = radar_dict
        .get_item("observable")?
        .ok_or_else(|| PyRuntimeError::new_err("missing radar 'observable'"))?
        .extract()?;
    let trx: Vec<String> = radar_dict
        .get_item("trx")?
        .ok_or_else(|| PyRuntimeError::new_err("missing radar 'trx'"))?
        .extract()?;
    let rcv: Vec<String> = radar_dict
        .get_item("rcv")?
        .ok_or_else(|| PyRuntimeError::new_err("missing radar 'rcv'"))?
        .extract()?;
    let perm_ids = req_strs(radar_dict, "perm_id")?;
    let prov_ids = req_strs(radar_dict, "prov_id")?;
    let trk_subs = req_strs(radar_dict, "trk_sub")?;
    let remarks_v = req_strs(radar_dict, "remarks")?;

    let delay_v = opt_floats(radar_dict, "delay", n)?;
    let rms_delay_v = opt_floats(radar_dict, "rms_delay", n)?;
    let doppler_v = opt_floats(radar_dict, "doppler", n)?;
    let rms_doppler_v = opt_floats(radar_dict, "rms_doppler", n)?;
    let frq_v = opt_floats(radar_dict, "frq", n)?;
    let log_snr_v = opt_floats(radar_dict, "log_snr", n)?;
    // `com`: tri-state Option<bool> with -1 sentinel (i8 array on the wire).
    let com_v: Vec<i8> = match radar_dict.get_item("com")? {
        Some(obj) => {
            let arr: PyReadonlyArray1<'_, i8> = obj.extract()?;
            arr.as_array().to_vec()
        }
        None => vec![-1; n],
    };

    if observables.len() != n
        || trx.len() != n
        || rcv.len() != n
        || perm_ids.len() != n
        || prov_ids.len() != n
        || trk_subs.len() != n
        || remarks_v.len() != n
        || com_v.len() != n
    {
        return Err(PyRuntimeError::new_err(
            "radar dict has mismatched column lengths",
        ));
    }

    let s_to_opt = |s: Option<String>| s.filter(|x| !x.is_empty());

    let mut radar = Vec::with_capacity(n);
    for i in 0..n {
        // The `observable` discriminator chooses the variant; it must be
        // one of the two known kinds — there is no silent default.
        let measurement = match observables[i].as_str() {
            "delay" => empyrean::RadarMeasurement::Delay {
                delay_seconds: delay_v[i],
                rms_delay_microseconds: rms_delay_v[i],
            },
            "doppler" => empyrean::RadarMeasurement::Doppler {
                doppler_hz: doppler_v[i],
                rms_doppler_hz: rms_doppler_v[i],
            },
            other => {
                return Err(PyValueError::new_err(format!(
                    "radar row {i}: unknown observable '{other}' (expected 'delay' or 'doppler')"
                )));
            }
        };
        let com = match com_v[i] {
            1 => Some(true),
            0 => Some(false),
            _ => None,
        };
        radar.push(empyrean::RadarObservation {
            perm_id: s_to_opt(perm_ids[i].clone()),
            prov_id: s_to_opt(prov_ids[i].clone()),
            trk_sub: s_to_opt(trk_subs[i].clone()),
            trx: trx[i].clone(),
            rcv: rcv[i].clone(),
            obs_time: obs_times[i].clone(),
            measurement,
            frq_mhz: frq_v[i],
            com,
            log_snr: if log_snr_v[i].is_nan() {
                None
            } else {
                Some(log_snr_v[i])
            },
            remarks: s_to_opt(remarks_v[i].clone()),
        });
    }
    Ok(radar)
}

fn build_orbit_from_dict<'py>(orbit_dict: &Bound<'py, PyDict>) -> PyResult<empyrean::Orbit> {
    let epochs_obj = orbit_dict
        .get_item("epochs")?
        .ok_or_else(|| PyRuntimeError::new_err("missing epochs"))?;
    let epochs_arr: PyReadonlyArray1<'_, f64> = epochs_obj.extract()?;
    let epochs = epochs_arr.as_array();

    let elements_obj = orbit_dict
        .get_item("elements")?
        .ok_or_else(|| PyRuntimeError::new_err("missing elements"))?;
    let elements_arr: PyReadonlyArray2<'_, f64> = elements_obj.extract()?;
    let elements = elements_arr.as_array();

    let cov_obj = orbit_dict
        .get_item("covariances")?
        .ok_or_else(|| PyRuntimeError::new_err("missing covariances"))?;
    let cov_arr: PyReadonlyArray3<'_, f64> = cov_obj.extract()?;
    let covariances = cov_arr.as_array();

    let has_cov_obj = orbit_dict
        .get_item("has_covariance")?
        .ok_or_else(|| PyRuntimeError::new_err("missing has_covariance"))?;
    let has_cov_arr: PyReadonlyArray1<'_, bool> = has_cov_obj.extract()?;
    let has_cov = has_cov_arr.as_array();

    let reps_obj = orbit_dict
        .get_item("representations")?
        .ok_or_else(|| PyRuntimeError::new_err("missing representations"))?;
    let reps_arr: PyReadonlyArray1<'_, i32> = reps_obj.extract()?;
    let reps = reps_arr.as_array();

    let frames_obj = orbit_dict
        .get_item("frames")?
        .ok_or_else(|| PyRuntimeError::new_err("missing frames"))?;
    let frames_arr: PyReadonlyArray1<'_, i32> = frames_obj.extract()?;
    let frames = frames_arr.as_array();

    let origins_obj = orbit_dict
        .get_item("origins")?
        .ok_or_else(|| PyRuntimeError::new_err("missing origins"))?;
    let origins_arr: PyReadonlyArray1<'_, i32> = origins_obj.extract()?;
    let origins = origins_arr.as_array();

    if epochs.is_empty() {
        return Err(PyRuntimeError::new_err("orbit dict is empty"));
    }

    let mut elems = [0.0f64; 6];
    for j in 0..6 {
        elems[j] = elements[[0, j]];
    }
    let covariance = if has_cov[0] {
        let mut cov = [[0.0f64; 6]; 6];
        for r in 0..6 {
            for c in 0..6 {
                cov[r][c] = covariances[[0, r, c]];
            }
        }
        Some(cov)
    } else {
        None
    };
    let state = empyrean::CoordinateState {
        epoch: empyrean::Epoch::from_mjd_tdb(epochs[0]),
        elements: elems,
        covariance,
        representation: empyrean::int_to_rep(reps[0]).map_err(to_pyerr)?,
        frame: empyrean::int_to_frame(frames[0]).map_err(to_pyerr)?,
        origin: origin_from_naif(origins[0])?,
    };
    let mut orbit = empyrean::Orbit::new(state);

    // Non-grav (optional): thread the seed orbit's force model so evaluate /
    // refine operate on the actual non-grav (not silently gravity-only), and
    // so a StateAndNonGrav refine keeps its fitted non-grav prior
    // (empyrean-wo4n). Missing keys (e.g. an unseeded determine) leave the
    // orbit gravity-only.
    let arr1 = |key: &str| -> PyResult<Option<Vec<f64>>> {
        match orbit_dict.get_item(key)? {
            Some(o) => {
                let a: PyReadonlyArray1<'_, f64> = o.extract()?;
                Ok(Some(a.as_array().to_vec()))
            }
            None => Ok(None),
        }
    };
    if let (Some(a1), Some(a2), Some(a3)) = (arr1("a1s")?, arr1("a2s")?, arr1("a3s")?)
        && (a1[0] != 0.0 || a2[0] != 0.0 || a3[0] != 0.0)
    {
        orbit = orbit.with_nongrav(a1[0], a2[0], a3[0]);
        // g(r) exponents (optional; all-zero = inverse-square default).
        if let (Some(al), Some(r0), Some(m), Some(n), Some(k)) = (
            arr1("ng_alphas")?,
            arr1("ng_r0s")?,
            arr1("ng_ms")?,
            arr1("ng_ns")?,
            arr1("ng_ks")?,
        ) && (al[0] != 0.0 || r0[0] != 0.0 || m[0] != 0.0 || n[0] != 0.0 || k[0] != 0.0)
        {
            orbit = orbit.with_g_function(al[0], r0[0], m[0], n[0], k[0]);
        }
        // Thermal-lag dt (optional; NaN sentinel = no delay).
        if let Some(dt) = arr1("non_grav_dts")?
            && dt[0].is_finite()
        {
            orbit = orbit.with_non_grav_dt(Some(dt[0]));
        }
        // Fitted non-grav covariance (optional; empyrean-wo4n).
        if let Some(has) = orbit_dict.get_item("has_non_grav_cov")? {
            let has_arr: PyReadonlyArray1<'_, bool> = has.extract()?;
            if has_arr.as_array()[0]
                && let Some(cov_obj) = orbit_dict.get_item("non_grav_cov")?
            {
                let cov_arr: PyReadonlyArray3<'_, f64> = cov_obj.extract()?;
                let c = cov_arr.as_array();
                let mut cov = [[0.0f64; 3]; 3];
                for i in 0..3 {
                    for j in 0..3 {
                        cov[i][j] = c[[0, i, j]];
                    }
                }
                orbit = orbit.with_nongrav_covariance(Some(cov));
            }
        }
    }
    Ok(orbit)
}

/// Map a [`empyrean::RejectionReason`] to a stable wire string for the
/// Python side. Matches the variant naming exactly so Python consumers
/// can pattern-match without going through the integer encoding.
fn rejection_reason_str(r: empyrean::RejectionReason) -> &'static str {
    use empyrean::RejectionReason as R;
    match r {
        R::Accepted => "accepted",
        R::ChiSquared => "chi_squared",
        R::SigmaClip => "sigma_clip",
        R::CooksDistance => "cooks_distance",
        R::Adaptive => "adaptive",
        R::UnsupportedObservatory => "unsupported_observatory",
        R::CMC2003 => "cmc2003",
        R::RadarObservationsUnsupported => "radar_observations_unsupported",
        R::OccultationObservationsUnsupported => "occultation_observations_unsupported",
        R::OutsideArc => "outside_arc",
        R::NotEvaluated => "not_evaluated",
    }
}

fn add_residuals_to_dict(
    dict: &Bound<'_, PyDict>,
    residuals: &[empyrean::ObservationResidual],
) -> PyResult<()> {
    let py = dict.py();
    let n = residuals.len();

    // Identification
    let mut obs_ids: Vec<String> = Vec::with_capacity(n);
    let mut obs_codes: Vec<String> = Vec::with_capacity(n);
    let mut ast_cats: Vec<Option<String>> = Vec::with_capacity(n);
    let mut epochs = Vec::with_capacity(n);
    // Core residuals
    let mut ra_residuals = Vec::with_capacity(n);
    let mut dec_residuals = Vec::with_capacity(n);
    let mut chi2s = Vec::with_capacity(n);
    let mut dofs: Vec<u32> = Vec::with_capacity(n);
    let mut probabilities = Vec::with_capacity(n);
    let mut selecteds = Vec::with_capacity(n);
    // Residual covariance
    let mut residual_cov_ras = Vec::with_capacity(n);
    let mut residual_cov_decs = Vec::with_capacity(n);
    let mut residual_cov_corrs = Vec::with_capacity(n);
    // Rejection
    let mut rejection_reasons: Vec<&'static str> = Vec::with_capacity(n);
    let mut rejection_criterions = Vec::with_capacity(n);
    let mut rejection_thresholds = Vec::with_capacity(n);
    let mut rejection_effective_thresholds = Vec::with_capacity(n);
    let mut rejection_information_losses = Vec::with_capacity(n);
    // Influence
    let mut cooks_distances = Vec::with_capacity(n);
    let mut leverages = Vec::with_capacity(n);
    let mut fractional_informations = Vec::with_capacity(n);
    // Along/cross-track
    let mut along_tracks = Vec::with_capacity(n);
    let mut cross_tracks = Vec::with_capacity(n);
    let mut along_track_errors = Vec::with_capacity(n);
    let mut cross_track_errors = Vec::with_capacity(n);
    let mut track_position_angles = Vec::with_capacity(n);

    for r in residuals {
        obs_ids.push(r.obs_id.clone());
        obs_codes.push(r.obs_code.clone());
        ast_cats.push(r.ast_cat.clone());
        epochs.push(r.epoch.mjd_tdb().map_err(to_pyerr)?);
        ra_residuals.push(r.ra_residual_arcsec);
        dec_residuals.push(r.dec_residual_arcsec);
        chi2s.push(r.chi2);
        dofs.push(r.dof);
        probabilities.push(r.probability);
        selecteds.push(r.selected);
        residual_cov_ras.push(r.residual_cov_ra);
        residual_cov_decs.push(r.residual_cov_dec);
        residual_cov_corrs.push(r.residual_cov_corr);
        rejection_reasons.push(rejection_reason_str(r.rejection_reason));
        rejection_criterions.push(r.rejection_criterion);
        rejection_thresholds.push(r.rejection_threshold);
        rejection_effective_thresholds.push(r.rejection_effective_threshold);
        rejection_information_losses.push(r.rejection_information_loss);
        cooks_distances.push(r.cooks_distance);
        leverages.push(r.leverage);
        fractional_informations.push(r.fractional_information);
        along_tracks.push(r.along_track_arcsec);
        cross_tracks.push(r.cross_track_arcsec);
        along_track_errors.push(r.along_track_error_arcsec);
        cross_track_errors.push(r.cross_track_error_arcsec);
        track_position_angles.push(r.track_position_angle_deg);
    }

    dict.set_item("obs_ids", obs_ids)?;
    dict.set_item("obs_codes", obs_codes)?;
    dict.set_item("ast_cats", ast_cats)?;
    dict.set_item("obs_epochs", PyArray1::from_vec(py, epochs))?;
    dict.set_item("ra_residuals", PyArray1::from_vec(py, ra_residuals))?;
    dict.set_item("dec_residuals", PyArray1::from_vec(py, dec_residuals))?;
    dict.set_item("chi2s", PyArray1::from_vec(py, chi2s))?;
    dict.set_item("dofs", PyArray1::from_vec(py, dofs))?;
    dict.set_item("probabilities", PyArray1::from_vec(py, probabilities))?;
    dict.set_item("selecteds", PyArray1::from_vec(py, selecteds))?;
    dict.set_item("residual_cov_ras", PyArray1::from_vec(py, residual_cov_ras))?;
    dict.set_item(
        "residual_cov_decs",
        PyArray1::from_vec(py, residual_cov_decs),
    )?;
    dict.set_item(
        "residual_cov_corrs",
        PyArray1::from_vec(py, residual_cov_corrs),
    )?;
    dict.set_item("rejection_reasons", rejection_reasons)?;
    dict.set_item(
        "rejection_criterions",
        PyArray1::from_vec(py, rejection_criterions),
    )?;
    dict.set_item(
        "rejection_thresholds",
        PyArray1::from_vec(py, rejection_thresholds),
    )?;
    dict.set_item(
        "rejection_effective_thresholds",
        PyArray1::from_vec(py, rejection_effective_thresholds),
    )?;
    dict.set_item(
        "rejection_information_losses",
        PyArray1::from_vec(py, rejection_information_losses),
    )?;
    dict.set_item("cooks_distances", PyArray1::from_vec(py, cooks_distances))?;
    dict.set_item("leverages", PyArray1::from_vec(py, leverages))?;
    dict.set_item(
        "fractional_informations",
        PyArray1::from_vec(py, fractional_informations),
    )?;
    dict.set_item("along_tracks", PyArray1::from_vec(py, along_tracks))?;
    dict.set_item("cross_tracks", PyArray1::from_vec(py, cross_tracks))?;
    dict.set_item(
        "along_track_errors",
        PyArray1::from_vec(py, along_track_errors),
    )?;
    dict.set_item(
        "cross_track_errors",
        PyArray1::from_vec(py, cross_track_errors),
    )?;
    dict.set_item(
        "track_position_angles",
        PyArray1::from_vec(py, track_position_angles),
    )?;
    Ok(())
}

fn add_summary_to_dict(
    dict: &Bound<'_, PyDict>,
    summary: &empyrean::ResidualSummary,
    prefix: &str,
) -> PyResult<()> {
    dict.set_item(format!("{prefix}num_obs"), summary.num_obs)?;
    dict.set_item(format!("{prefix}num_selected"), summary.num_selected)?;
    dict.set_item(format!("{prefix}num_rejected"), summary.num_rejected)?;
    dict.set_item(format!("{prefix}chi2"), summary.chi2)?;
    dict.set_item(format!("{prefix}dof"), summary.dof)?;
    dict.set_item(format!("{prefix}reduced_chi2"), summary.reduced_chi2)?;
    dict.set_item(format!("{prefix}rms_ra"), summary.rms_ra_arcsec)?;
    dict.set_item(format!("{prefix}rms_dec"), summary.rms_dec_arcsec)?;
    dict.set_item(format!("{prefix}rms_combined"), summary.rms_combined_arcsec)?;
    dict.set_item(
        format!("{prefix}weighted_rms_ra"),
        summary.weighted_rms_ra_arcsec,
    )?;
    dict.set_item(
        format!("{prefix}weighted_rms_dec"),
        summary.weighted_rms_dec_arcsec,
    )?;
    dict.set_item(
        format!("{prefix}weighted_rms_combined"),
        summary.weighted_rms_combined_arcsec,
    )?;
    dict.set_item(format!("{prefix}mean_ra"), summary.mean_ra_arcsec)?;
    dict.set_item(format!("{prefix}mean_dec"), summary.mean_dec_arcsec)?;
    dict.set_item(format!("{prefix}std_ra"), summary.std_ra_arcsec)?;
    dict.set_item(format!("{prefix}std_dec"), summary.std_dec_arcsec)?;
    dict.set_item(
        format!("{prefix}rms_along_track"),
        summary.rms_along_track_arcsec,
    )?;
    dict.set_item(
        format!("{prefix}rms_cross_track"),
        summary.rms_cross_track_arcsec,
    )?;
    Ok(())
}

fn add_acceptability_to_dict(
    dict: &Bound<'_, PyDict>,
    acc: &empyrean::AcceptabilityReport,
    prefix: &str,
) -> PyResult<()> {
    dict.set_item(format!("{prefix}fit_acceptable"), acc.fit_acceptable)?;
    dict.set_item(
        format!("{prefix}extrapolation_acceptable"),
        acc.extrapolation_acceptable,
    )?;
    dict.set_item(format!("{prefix}converged_ok"), acc.converged_ok)?;
    dict.set_item(format!("{prefix}reduced_chi2_ok"), acc.reduced_chi2_ok)?;
    dict.set_item(
        format!("{prefix}reduced_chi2_value"),
        acc.reduced_chi2_value,
    )?;
    dict.set_item(
        format!("{prefix}reduced_chi2_threshold"),
        acc.reduced_chi2_threshold,
    )?;
    dict.set_item(format!("{prefix}rms_ok"), acc.rms_ok)?;
    dict.set_item(format!("{prefix}rms_value_arcsec"), acc.rms_value_arcsec)?;
    dict.set_item(
        format!("{prefix}rms_threshold_arcsec"),
        acc.rms_threshold_arcsec,
    )?;
    dict.set_item(
        format!("{prefix}residual_isotropy_ok"),
        acc.residual_isotropy_ok,
    )?;
    dict.set_item(format!("{prefix}at_ct_ratio_value"), acc.at_ct_ratio_value)?;
    dict.set_item(
        format!("{prefix}at_ct_ratio_threshold"),
        acc.at_ct_ratio_threshold,
    )?;
    dict.set_item(format!("{prefix}covariance_ok"), acc.covariance_ok)?;
    dict.set_item(format!("{prefix}arc_coverage_ok"), acc.arc_coverage_ok)?;
    dict.set_item(format!("{prefix}arc_days_value"), acc.arc_days_value)?;
    dict.set_item(
        format!("{prefix}arc_days_threshold"),
        acc.arc_days_threshold,
    )?;
    dict.set_item(
        format!("{prefix}fractional_sigma_a_ok"),
        acc.fractional_sigma_a_ok,
    )?;
    dict.set_item(
        format!("{prefix}fractional_sigma_a_value"),
        acc.fractional_sigma_a_value,
    )?;
    dict.set_item(
        format!("{prefix}fractional_sigma_a_threshold"),
        acc.fractional_sigma_a_threshold,
    )?;
    Ok(())
}

fn add_propagated_to_dict(
    dict: &Bound<'_, PyDict>,
    orbit_id: &str,
    object_id: &str,
    state: &empyrean::PropagatedState,
    prefix: &str,
) -> PyResult<()> {
    dict.set_item(format!("{prefix}orbit_id"), orbit_id)?;
    dict.set_item(format!("{prefix}object_id"), object_id)?;
    dict.set_item(
        format!("{prefix}epoch"),
        state.epoch.mjd_tdb().map_err(to_pyerr)?,
    )?;
    dict.set_item(format!("{prefix}x"), state.position[0])?;
    dict.set_item(format!("{prefix}y"), state.position[1])?;
    dict.set_item(format!("{prefix}z"), state.position[2])?;
    dict.set_item(format!("{prefix}vx"), state.velocity[0])?;
    dict.set_item(format!("{prefix}vy"), state.velocity[1])?;
    dict.set_item(format!("{prefix}vz"), state.velocity[2])?;
    dict.set_item(
        format!("{prefix}frame"),
        empyrean::frame_to_int(state.frame),
    )?;
    dict.set_item(format!("{prefix}origin"), state.origin.naif_id())?;
    if let Some(c) = &state.covariance {
        let mut flat = Vec::with_capacity(36);
        for r in 0..6 {
            for col in 0..6 {
                flat.push(c[r][col]);
            }
        }
        dict.set_item(format!("{prefix}covariance"), flat)?;
    }
    Ok(())
}

// ══════════════════════════════════════════════════════════
//  _read_ades
// ══════════════════════════════════════════════════════════

#[pyfunction]
fn _read_ades<'py>(py: Python<'py>, path_or_content: &str) -> PyResult<Bound<'py, PyDict>> {
    let ctx = get_context()?;
    let observations = ctx.read_ades(path_or_content).map_err(to_pyerr)?;

    let n = observations.len();
    let mut perm_ids: Vec<String> = Vec::with_capacity(n);
    let mut prov_ids: Vec<String> = Vec::with_capacity(n);
    let mut trk_subs: Vec<String> = Vec::with_capacity(n);
    let mut obs_ids: Vec<String> = Vec::with_capacity(n);
    let mut obs_sub_ids: Vec<String> = Vec::with_capacity(n);
    let mut trk_ids: Vec<String> = Vec::with_capacity(n);
    let mut stns: Vec<String> = Vec::with_capacity(n);
    let mut modes: Vec<String> = Vec::with_capacity(n);
    let mut progs: Vec<String> = Vec::with_capacity(n);
    let mut sys_v: Vec<String> = Vec::with_capacity(n);
    let mut ctr_arr = Vec::with_capacity(n);
    let mut pos1_arr = Vec::with_capacity(n);
    let mut pos2_arr = Vec::with_capacity(n);
    let mut pos3_arr = Vec::with_capacity(n);
    let mut obs_times: Vec<String> = Vec::with_capacity(n);
    let mut ra_arr = Vec::with_capacity(n);
    let mut dec_arr = Vec::with_capacity(n);
    let mut rms_ra_arr = Vec::with_capacity(n);
    let mut rms_dec_arr = Vec::with_capacity(n);
    let mut rms_corr_arr = Vec::with_capacity(n);
    let mut ast_cats: Vec<String> = Vec::with_capacity(n);
    let mut mag_arr = Vec::with_capacity(n);
    let mut rms_mag_arr = Vec::with_capacity(n);
    let mut bands: Vec<String> = Vec::with_capacity(n);
    let mut phot_cats: Vec<String> = Vec::with_capacity(n);
    let mut phot_ap_arr = Vec::with_capacity(n);
    let mut log_snr_arr = Vec::with_capacity(n);
    let mut seeing_arr = Vec::with_capacity(n);
    let mut exp_arr = Vec::with_capacity(n);
    let mut rms_fit_arr = Vec::with_capacity(n);
    let mut n_stars_arr = Vec::with_capacity(n);
    let mut notes_v: Vec<String> = Vec::with_capacity(n);
    let mut remarks_v: Vec<String> = Vec::with_capacity(n);

    let opt_to_nan = |v: Option<f64>| v.unwrap_or(f64::NAN);

    for obs in observations.iter() {
        perm_ids.push(obs.perm_id.unwrap_or_default());
        prov_ids.push(obs.prov_id.unwrap_or_default());
        trk_subs.push(obs.trk_sub.unwrap_or_default());
        obs_ids.push(obs.obs_id.unwrap_or_default());
        obs_sub_ids.push(obs.obs_sub_id.unwrap_or_default());
        trk_ids.push(obs.trk_id.unwrap_or_default());
        stns.push(obs.obs_code);
        modes.push(obs.mode.unwrap_or_default());
        progs.push(obs.prog.unwrap_or_default());
        sys_v.push(obs.sys.unwrap_or_default());
        ctr_arr.push(opt_to_nan(obs.ctr));
        pos1_arr.push(opt_to_nan(obs.pos1));
        pos2_arr.push(opt_to_nan(obs.pos2));
        pos3_arr.push(opt_to_nan(obs.pos3));
        obs_times.push(obs.obs_time);
        ra_arr.push(obs.ra_deg);
        dec_arr.push(obs.dec_deg);
        rms_ra_arr.push(obs.rms_ra_arcsec);
        rms_dec_arr.push(obs.rms_dec_arcsec);
        rms_corr_arr.push(opt_to_nan(obs.rms_corr));
        ast_cats.push(obs.ast_cat.unwrap_or_default());
        mag_arr.push(opt_to_nan(obs.mag));
        rms_mag_arr.push(opt_to_nan(obs.rms_mag));
        bands.push(obs.band.unwrap_or_default());
        phot_cats.push(obs.phot_cat.unwrap_or_default());
        phot_ap_arr.push(opt_to_nan(obs.phot_ap));
        log_snr_arr.push(opt_to_nan(obs.log_snr));
        seeing_arr.push(opt_to_nan(obs.seeing));
        exp_arr.push(opt_to_nan(obs.exp));
        rms_fit_arr.push(opt_to_nan(obs.rms_fit));
        n_stars_arr.push(obs.n_stars.map(|v| v as i32).unwrap_or(-1));
        notes_v.push(obs.notes.unwrap_or_default());
        remarks_v.push(obs.remarks.unwrap_or_default());
    }

    let dict = PyDict::new(py);
    dict.set_item("perm_id", perm_ids)?;
    dict.set_item("prov_id", prov_ids)?;
    dict.set_item("trk_sub", trk_subs)?;
    dict.set_item("obs_id", obs_ids)?;
    dict.set_item("obs_sub_id", obs_sub_ids)?;
    dict.set_item("trk_id", trk_ids)?;
    dict.set_item("stn", stns)?;
    dict.set_item("mode", modes)?;
    dict.set_item("prog", progs)?;
    dict.set_item("sys", sys_v)?;
    dict.set_item("ctr", PyArray1::from_vec(py, ctr_arr))?;
    dict.set_item("pos1", PyArray1::from_vec(py, pos1_arr))?;
    dict.set_item("pos2", PyArray1::from_vec(py, pos2_arr))?;
    dict.set_item("pos3", PyArray1::from_vec(py, pos3_arr))?;
    dict.set_item("obs_time", obs_times)?;
    dict.set_item("ra", PyArray1::from_vec(py, ra_arr))?;
    dict.set_item("dec", PyArray1::from_vec(py, dec_arr))?;
    dict.set_item("rms_ra", PyArray1::from_vec(py, rms_ra_arr))?;
    dict.set_item("rms_dec", PyArray1::from_vec(py, rms_dec_arr))?;
    dict.set_item("rms_corr", PyArray1::from_vec(py, rms_corr_arr))?;
    dict.set_item("ast_cat", ast_cats)?;
    dict.set_item("mag", PyArray1::from_vec(py, mag_arr))?;
    dict.set_item("rms_mag", PyArray1::from_vec(py, rms_mag_arr))?;
    dict.set_item("band", bands)?;
    dict.set_item("phot_cat", phot_cats)?;
    dict.set_item("phot_ap", PyArray1::from_vec(py, phot_ap_arr))?;
    dict.set_item("log_snr", PyArray1::from_vec(py, log_snr_arr))?;
    dict.set_item("seeing", PyArray1::from_vec(py, seeing_arr))?;
    dict.set_item("exp", PyArray1::from_vec(py, exp_arr))?;
    dict.set_item("rms_fit", PyArray1::from_vec(py, rms_fit_arr))?;
    dict.set_item("n_stars", PyArray1::from_vec(py, n_stars_arr))?;
    dict.set_item("notes", notes_v)?;
    dict.set_item("remarks", remarks_v)?;

    // ── Radar table (nested under "radar") ───────────────────────────
    // Parallel to the optical columns above: ADES models radar as its
    // own table. The `observable` discriminator ("delay" / "doppler")
    // crosses the boundary explicitly so a 0.0-Hz Doppler is never
    // mistaken for an absent one. Values stay ADES-native (delay s,
    // rmsDelay µs, doppler/rmsDoppler Hz, frq MHz) — no conversion.
    dict.set_item("radar", radar_table_to_pydict(py, &observations.radar())?)?;

    Ok(dict)
}

/// Materialize a slice of [`empyrean::RadarObservation`] into a flat
/// column-array [`PyDict`] mirroring the optical `_read_ades` shape and
/// the keys [`build_radar_observations`] decodes.
fn radar_table_to_pydict<'py>(
    py: Python<'py>,
    radar: &[empyrean::RadarObservation],
) -> PyResult<Bound<'py, PyDict>> {
    let n = radar.len();
    let mut perm_ids: Vec<String> = Vec::with_capacity(n);
    let mut prov_ids: Vec<String> = Vec::with_capacity(n);
    let mut trk_subs: Vec<String> = Vec::with_capacity(n);
    let mut trx_v: Vec<String> = Vec::with_capacity(n);
    let mut rcv_v: Vec<String> = Vec::with_capacity(n);
    let mut obs_times: Vec<String> = Vec::with_capacity(n);
    let mut observables: Vec<&'static str> = Vec::with_capacity(n);
    let mut delay_arr = Vec::with_capacity(n);
    let mut rms_delay_arr = Vec::with_capacity(n);
    let mut doppler_arr = Vec::with_capacity(n);
    let mut rms_doppler_arr = Vec::with_capacity(n);
    let mut frq_arr = Vec::with_capacity(n);
    let mut com_arr: Vec<i8> = Vec::with_capacity(n);
    let mut log_snr_arr = Vec::with_capacity(n);
    let mut remarks_v: Vec<String> = Vec::with_capacity(n);

    for obs in radar {
        perm_ids.push(obs.perm_id.clone().unwrap_or_default());
        prov_ids.push(obs.prov_id.clone().unwrap_or_default());
        trk_subs.push(obs.trk_sub.clone().unwrap_or_default());
        trx_v.push(obs.trx.clone());
        rcv_v.push(obs.rcv.clone());
        obs_times.push(obs.obs_time.clone());
        // The inactive value pair is NaN; the `observable` discriminator
        // (not the NaN) is what carries the choice across the boundary.
        match &obs.measurement {
            empyrean::RadarMeasurement::Delay {
                delay_seconds,
                rms_delay_microseconds,
            } => {
                observables.push("delay");
                delay_arr.push(*delay_seconds);
                rms_delay_arr.push(*rms_delay_microseconds);
                doppler_arr.push(f64::NAN);
                rms_doppler_arr.push(f64::NAN);
            }
            empyrean::RadarMeasurement::Doppler {
                doppler_hz,
                rms_doppler_hz,
            } => {
                observables.push("doppler");
                delay_arr.push(f64::NAN);
                rms_delay_arr.push(f64::NAN);
                doppler_arr.push(*doppler_hz);
                rms_doppler_arr.push(*rms_doppler_hz);
            }
        }
        frq_arr.push(obs.frq_mhz);
        com_arr.push(match obs.com {
            Some(true) => 1,
            Some(false) => 0,
            None => -1,
        });
        log_snr_arr.push(obs.log_snr.unwrap_or(f64::NAN));
        remarks_v.push(obs.remarks.clone().unwrap_or_default());
    }

    let radar_dict = PyDict::new(py);
    radar_dict.set_item("perm_id", perm_ids)?;
    radar_dict.set_item("prov_id", prov_ids)?;
    radar_dict.set_item("trk_sub", trk_subs)?;
    radar_dict.set_item("trx", trx_v)?;
    radar_dict.set_item("rcv", rcv_v)?;
    radar_dict.set_item("obs_time", obs_times)?;
    radar_dict.set_item("observable", observables)?;
    radar_dict.set_item("delay", PyArray1::from_vec(py, delay_arr))?;
    radar_dict.set_item("rms_delay", PyArray1::from_vec(py, rms_delay_arr))?;
    radar_dict.set_item("doppler", PyArray1::from_vec(py, doppler_arr))?;
    radar_dict.set_item("rms_doppler", PyArray1::from_vec(py, rms_doppler_arr))?;
    radar_dict.set_item("frq", PyArray1::from_vec(py, frq_arr))?;
    radar_dict.set_item("com", PyArray1::from_vec(py, com_arr))?;
    radar_dict.set_item("log_snr", PyArray1::from_vec(py, log_snr_arr))?;
    radar_dict.set_item("remarks", remarks_v)?;
    Ok(radar_dict)
}

// ══════════════════════════════════════════════════════════
//  _determine
// ══════════════════════════════════════════════════════════

#[pyfunction]
#[pyo3(signature = (obs_dict, config_dict, initial_orbits_dict = None, radar_dict = None))]
fn _determine<'py>(
    py: Python<'py>,
    obs_dict: &Bound<'py, PyDict>,
    config_dict: &Bound<'py, PyDict>,
    initial_orbits_dict: Option<&Bound<'py, PyDict>>,
    radar_dict: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let ctx = get_context()?;
    // The legacy ``ades`` key path carries both optical and radar through
    // `read_ades`; the flat-dict path builds the optical vec here and
    // combines it with the radar vec via `Observations::from_arrays`.
    let observations = if obs_dict.get_item("ades")?.is_some() {
        build_observations(ctx, obs_dict)?
    } else {
        let optical = build_observation_vec(obs_dict)?;
        let radar = build_radar_observations(radar_dict)?;
        empyrean::Observations::from_arrays(&optical, &radar).map_err(to_pyerr)?
    };

    let initial_orbits: Option<Vec<empyrean::Orbit>> = if let Some(init_dict) = initial_orbits_dict
    {
        let mut orbits = Vec::with_capacity(init_dict.len());
        for (_, value) in init_dict.iter() {
            let orbit_py_dict: Bound<'py, PyDict> = value.extract()?;
            orbits.push(build_orbit_from_dict(&orbit_py_dict)?);
        }
        Some(orbits)
    } else {
        None
    };

    let od_config = build_od_config_from_dict(config_dict)?;
    let determine_result = py
        .detach(|| ctx.determine(&observations, initial_orbits.as_deref(), &od_config))
        .map_err(to_pyerr)?;

    determine_result_to_pydict(py, &determine_result)
}

// ══════════════════════════════════════════════════════════
//  _evaluate_single
// ══════════════════════════════════════════════════════════

#[pyfunction]
#[pyo3(signature = (orbit_dict, obs_dict, config_dict))]
fn _evaluate_single<'py>(
    py: Python<'py>,
    orbit_dict: &Bound<'py, PyDict>,
    obs_dict: &Bound<'py, PyDict>,
    config_dict: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let ctx = get_context()?;
    let orbit = build_orbit_from_dict(orbit_dict)?;
    let observations = build_observations(ctx, obs_dict)?;
    let od_config = build_od_config_from_dict(config_dict)?;
    let eval_result = py
        .detach(|| ctx.evaluate(&orbit, &observations, &od_config))
        .map_err(to_pyerr)?;

    let dict = PyDict::new(py);
    add_residuals_to_dict(&dict, &eval_result.residuals)?;
    add_summary_to_dict(&dict, &eval_result.summary, "summary_")?;
    Ok(dict)
}

// ══════════════════════════════════════════════════════════
//  _refine_single
// ══════════════════════════════════════════════════════════

#[pyfunction]
#[pyo3(signature = (orbit_dict, obs_dict, config_dict))]
fn _refine_single<'py>(
    py: Python<'py>,
    orbit_dict: &Bound<'py, PyDict>,
    obs_dict: &Bound<'py, PyDict>,
    config_dict: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let ctx = get_context()?;
    let orbit = build_orbit_from_dict(orbit_dict)?;
    let observations = build_observations(ctx, obs_dict)?;
    let od_config = build_od_config_from_dict(config_dict)?;
    let od_result = py
        .detach(|| ctx.refine(&orbit, &observations, &od_config))
        .map_err(to_pyerr)?;

    determine_result_to_pydict(py, &od_result)
}

// ══════════════════════════════════════════════════════════
//  ODConfig dict ↔ Rust struct
// ══════════════════════════════════════════════════════════

/// Build an [`empyrean::ODConfig`] from a Python-side nested dict.
///
/// The dict structure mirrors `empyrean::ODConfig` exactly:
/// - top-level keys: `force_model`, `epsilon`, `max_iterations`, …
/// - nested sub-dicts: `iod`, `output_epoch`, `auto_escalation`,
///   `acceptability`, `rejection`, `station_radec`
///
/// Missing keys default to `ODConfig::default()` (which mirrors
/// scott's upstream defaults exactly). The Python ODConfig dataclass'
/// `to_dict()` produces the canonical input shape.
fn build_od_config_from_dict(d: &Bound<'_, PyDict>) -> PyResult<empyrean::ODConfig> {
    let mut cfg = empyrean::ODConfig::default();

    fn get_str(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
        match d.get_item(key)? {
            Some(v) => Ok(Some(v.extract()?)),
            None => Ok(None),
        }
    }
    fn get_u32(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<u32>> {
        match d.get_item(key)? {
            Some(v) => Ok(Some(v.extract()?)),
            None => Ok(None),
        }
    }
    fn get_i32(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<i32>> {
        match d.get_item(key)? {
            Some(v) if v.is_none() => Ok(None),
            Some(v) => Ok(Some(v.extract()?)),
            None => Ok(None),
        }
    }
    fn get_usize(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<usize>> {
        match d.get_item(key)? {
            Some(v) => Ok(Some(v.extract()?)),
            None => Ok(None),
        }
    }
    fn get_f64(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<f64>> {
        match d.get_item(key)? {
            Some(v) => Ok(Some(v.extract()?)),
            None => Ok(None),
        }
    }
    fn get_bool(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<bool>> {
        match d.get_item(key)? {
            Some(v) => Ok(Some(v.extract()?)),
            None => Ok(None),
        }
    }
    fn get_dict<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<Bound<'py, PyDict>>> {
        match d.get_item(key)? {
            Some(v) => Ok(Some(v.extract()?)),
            None => Ok(None),
        }
    }

    // ── Shared ────────────────────────────────────────────────────
    if let Some(s) = get_str(d, "force_model")? {
        cfg.force_model = match s.to_ascii_lowercase().as_str() {
            "approximate" => empyrean::ForceModelTier::Approximate,
            "basic" => empyrean::ForceModelTier::Basic,
            "standard" => empyrean::ForceModelTier::Standard,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown force_model: {other}"
                )));
            }
        };
    }
    if let Some(v) = get_f64(d, "epsilon")? {
        cfg.epsilon = v;
    }
    if let Some(v) = get_usize(d, "max_light_time_iterations")? {
        cfg.max_light_time_iterations = v;
    }
    if let Some(v) = get_usize(d, "num_threads")? {
        cfg.num_threads = std::num::NonZeroUsize::new(v).map_or(0, |n| n.get());
    }
    if let Some(s) = get_str(d, "frame")? {
        cfg.frame = match s.to_ascii_lowercase().as_str() {
            "icrf" => empyrean::Frame::ICRF,
            "eclipticj2000" | "ecliptic_j2000" | "ecliptic" => empyrean::Frame::EclipticJ2000,
            "itrf93" | "itrf_93" => empyrean::Frame::ITRF93,
            other => {
                return Err(PyValueError::new_err(format!("unknown frame: {other}")));
            }
        };
    }
    // ── Weighting (structured) ────────────────────────────────────
    if let Some(w) = get_dict(d, "weighting")? {
        let mut wc = empyrean::WeightingConfig::default();
        if let Some(v) = get_bool(&w, "enabled")? {
            wc.enabled = v;
        }
        if let Some(s) = get_str(&w, "preset")? {
            wc.preset = match s.to_ascii_lowercase().as_str() {
                "none" => empyrean::WeightingPreset::None,
                "vfc17" => empyrean::WeightingPreset::Vfc17,
                "neodys" => empyrean::WeightingPreset::Neodys,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unknown weighting preset: {other}"
                    )));
                }
            };
        }
        if let Some(v) = get_f64(&w, "default_sigma_arcsec")? {
            wc.default_sigma_arcsec = v;
        }
        if let Some(sp) = w.get_item("sigma_policy")?
            && !sp.is_none()
        {
            let s: String = sp.extract()?;
            wc.sigma_policy = Some(match s.to_ascii_lowercase().as_str() {
                "default_only" => empyrean::SigmaPolicy::DefaultOnly,
                "floor" => empyrean::SigmaPolicy::Floor,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unknown sigma_policy: {other}"
                    )));
                }
            });
        }
        if let Some(layers_obj) = w.get_item("additional_layers")?
            && !layers_obj.is_none()
        {
            let mut additional: Vec<empyrean::WeightingLayer> = Vec::new();
            for item in layers_obj.try_iter()? {
                let layer: Bound<'_, PyDict> = item?.extract()?;
                let kind = get_str(&layer, "kind")?
                    .ok_or_else(|| PyValueError::new_err("weighting layer missing 'kind'"))?;
                let layer_value = match kind.to_ascii_lowercase().as_str() {
                    "observatory_rule" => {
                        let obs_code = get_str(&layer, "obs_code")?.ok_or_else(|| {
                            PyValueError::new_err("observatory_rule layer missing obs_code")
                        })?;
                        let sigma_obj = layer.get_item("sigma")?.ok_or_else(|| {
                            PyValueError::new_err("observatory_rule layer missing sigma")
                        })?;
                        let sigma_vec: Vec<f64> = sigma_obj.extract()?;
                        if sigma_vec.len() != 2 {
                            return Err(PyValueError::new_err(format!(
                                "observatory_rule sigma must be 2-element list, got {}",
                                sigma_vec.len()
                            )));
                        }
                        let start = match layer.get_item("start_epoch_mjd_tdb")? {
                            Some(v) if !v.is_none() => Some(v.extract::<f64>()?),
                            _ => None,
                        };
                        let end = match layer.get_item("end_epoch_mjd_tdb")? {
                            Some(v) if !v.is_none() => Some(v.extract::<f64>()?),
                            _ => None,
                        };
                        let scale = get_f64(&layer, "scale")?.unwrap_or(1.0);
                        empyrean::WeightingLayer::ObservatoryRule {
                            obs_code,
                            sigma: [sigma_vec[0], sigma_vec[1]],
                            start_epoch_mjd_tdb: start,
                            end_epoch_mjd_tdb: end,
                            scale,
                        }
                    }
                    "nightly_deweighting" => {
                        let max_gap_days = get_f64(&layer, "max_gap_days")?.unwrap_or(0.5);
                        empyrean::WeightingLayer::NightlyDeweighting { max_gap_days }
                    }
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "unknown weighting layer kind: {other}"
                        )));
                    }
                };
                additional.push(layer_value);
            }
            wc.additional_layers = additional;
        }
        cfg.weighting = wc;
    }

    // ── Debiasing (structured) ────────────────────────────────────
    if let Some(b) = get_dict(d, "debiasing")? {
        let mut dc = empyrean::DebiasingConfig::default();
        if let Some(v) = get_bool(&b, "enabled")? {
            dc.enabled = v;
        }
        if let Some(s) = get_str(&b, "resolution")? {
            dc.resolution = match s.to_ascii_lowercase().as_str() {
                "standard" => empyrean::DebiasingResolution::Standard,
                "hires" => empyrean::DebiasingResolution::Hires,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unknown debiasing resolution: {other}"
                    )));
                }
            };
        }
        if let Some(p) = b.get_item("bias_dat_path")?
            && !p.is_none()
        {
            let s: String = p.extract()?;
            dc.bias_dat_path = Some(std::path::PathBuf::from(s));
        }
        cfg.debiasing = dc;
    }
    if let Some(v) = d.get_item("excluded_perturbers_naif")? {
        let list: Vec<i32> = v.extract()?;
        cfg.excluded_perturbers = list
            .into_iter()
            .map(|naif| {
                empyrean::Origin::from_naif_id(naif).ok_or_else(|| {
                    PyValueError::new_err(format!("unknown NAIF id in excluded_perturbers: {naif}"))
                })
            })
            .collect::<PyResult<_>>()?;
    }

    // ── Origin policy ─────────────────────────────────────────────
    if let Some(op) = get_dict(d, "origin")? {
        let mode = get_str(&op, "mode")?
            .ok_or_else(|| PyValueError::new_err("origin.mode is required"))?;
        cfg.origin = match mode.to_ascii_lowercase().as_str() {
            "auto" => empyrean::OriginPolicy::Auto,
            "explicit" => {
                let naif = get_i32(&op, "naif_id")?.ok_or_else(|| {
                    PyValueError::new_err("origin.naif_id is required when mode = 'explicit'")
                })?;
                let origin = empyrean::Origin::from_naif_id(naif).ok_or_else(|| {
                    PyValueError::new_err(format!("unknown NAIF id in origin.naif_id: {naif}"))
                })?;
                empyrean::OriginPolicy::Explicit(origin)
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown origin.mode: {other}"
                )));
            }
        };
    }

    // ── IOD ───────────────────────────────────────────────────────
    if let Some(iod) = get_dict(d, "iod")? {
        if let Some(v) = get_u32(&iod, "max_triplet_attempts")? {
            cfg.iod.max_triplet_attempts = v;
        }
        if let Some(v) = get_f64(&iod, "max_triplet_span_days")? {
            cfg.iod.max_triplet_span_days = v;
        }
        if let Some(v) = get_f64(&iod, "opposition_gap_days")? {
            cfg.iod.opposition_gap_days = v;
        }
        if let Some(v) = get_f64(&iod, "max_iod_arc_days")? {
            cfg.iod.max_iod_arc_days = v;
        }
        if let Some(v) = get_f64(&iod, "curvature_snr_threshold")? {
            cfg.iod.curvature_snr_threshold = v;
        }
        if let Some(v) = get_f64(&iod, "max_iod_fractional_sigma_a")? {
            cfg.iod.max_iod_fractional_sigma_a = v;
        }
    }

    // ── Output epoch ──────────────────────────────────────────────
    if let Some(oe) = get_dict(d, "output_epoch")? {
        let mode = get_str(&oe, "mode")?
            .ok_or_else(|| PyValueError::new_err("output_epoch.mode is required"))?;
        cfg.output_epoch = match mode.to_ascii_lowercase().as_str() {
            "mid_arc" | "midarc" => empyrean::OutputEpoch::MidArc,
            "last_observation" | "last" => empyrean::OutputEpoch::LastObservation,
            "iod_epoch" | "iod" => empyrean::OutputEpoch::IODEpoch,
            "explicit" | "epoch" => {
                let mjd = get_f64(&oe, "mjd_tdb")?.ok_or_else(|| {
                    PyValueError::new_err("output_epoch.mjd_tdb is required when mode = 'explicit'")
                })?;
                empyrean::OutputEpoch::Epoch(mjd)
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown output_epoch.mode: {other}"
                )));
            }
        };
    }

    // ── DC ────────────────────────────────────────────────────────
    if let Some(v) = get_u32(d, "max_iterations")? {
        cfg.max_iterations = v;
    }
    if let Some(v) = get_f64(d, "convergence_tol")? {
        cfg.convergence_tol = v;
    }
    if let Some(v) = get_bool(d, "use_stm_cache")? {
        cfg.use_stm_cache = v;
    }
    if let Some(s) = get_str(d, "solve_for")? {
        cfg.solve_for = match s.to_ascii_lowercase().as_str() {
            "state_only" | "state" => empyrean::SolveForParams::StateOnly,
            "state_and_nongrav" | "state_and_non_grav" => empyrean::SolveForParams::StateAndNonGrav,
            "auto" => empyrean::SolveForParams::Auto,
            other => {
                return Err(PyValueError::new_err(format!("unknown solve_for: {other}")));
            }
        };
    }

    if let Some(ae) = get_dict(d, "auto_escalation")? {
        if let Some(v) = get_f64(&ae, "reduced_chi2")? {
            cfg.auto_escalation.reduced_chi2 = v;
        }
        if let Some(v) = get_f64(&ae, "at_ct_ratio")? {
            cfg.auto_escalation.at_ct_ratio = v;
        }
        if let Some(v) = get_f64(&ae, "min_arc_days")? {
            cfg.auto_escalation.min_arc_days = v;
        }
        if let Some(v) = get_u32(&ae, "min_n_obs")? {
            cfg.auto_escalation.min_n_obs = v;
        }
    }
    if let Some(ac) = get_dict(d, "acceptability")? {
        if let Some(v) = get_f64(&ac, "reduced_chi2")? {
            cfg.acceptability.reduced_chi2 = v;
        }
        if let Some(v) = get_f64(&ac, "rms_arcsec")? {
            cfg.acceptability.rms_arcsec = v;
        }
        if let Some(v) = get_f64(&ac, "at_ct_ratio")? {
            cfg.acceptability.at_ct_ratio = v;
        }
        if let Some(v) = get_f64(&ac, "min_arc_days")? {
            cfg.acceptability.min_arc_days = v;
        }
        if let Some(v) = get_f64(&ac, "fractional_sigma_a")? {
            cfg.acceptability.fractional_sigma_a = v;
        }
    }
    if let Some(v) = get_bool(d, "fit_station_biases")? {
        cfg.fit_station_biases = v;
    }
    if let Some(srd) = get_dict(d, "station_radec")? {
        if let Some(v) = get_f64(&srd, "sigma_prior_arcsec")? {
            cfg.station_radec.sigma_prior_arcsec = v;
        }
        if let Some(v) = get_usize(&srd, "min_obs_per_station")? {
            cfg.station_radec.min_obs_per_station = v;
        }
    }
    if let Some(v) = get_bool(d, "use_span_grouping")? {
        cfg.use_span_grouping = v;
    }

    // ── Rejection ─────────────────────────────────────────────────
    if let Some(rej) = get_dict(d, "rejection")? {
        if let Some(v) = get_bool(&rej, "enabled")? {
            cfg.rejection.enabled = v;
        }
        if let Some(s) = get_str(&rej, "kind")? {
            cfg.rejection.kind = match s.to_ascii_lowercase().as_str() {
                "adaptive" => empyrean::RejectionKind::Adaptive,
                "cmc2003" => empyrean::RejectionKind::CMC2003,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "rejection.kind must be 'adaptive' or 'cmc2003' (got '{other}')"
                    )));
                }
            };
        }
        if let Some(v) = get_f64(&rej, "chi2_base")? {
            cfg.rejection.chi2_base = v;
        }
        if let Some(v) = get_f64(&rej, "lambda")? {
            cfg.rejection.lambda = v;
        }
        if let Some(v) = get_f64(&rej, "max_threshold")? {
            cfg.rejection.max_threshold = v;
        }
        if let Some(v) = get_f64(&rej, "chi2_rej")? {
            cfg.rejection.chi2_rej = v;
        }
        if let Some(v) = get_f64(&rej, "chi2_rec")? {
            cfg.rejection.chi2_rec = v;
        }
        if let Some(v) = get_u32(&rej, "max_passes")? {
            cfg.rejection.max_passes = v;
        }
    }
    if let Some(v) = get_bool(d, "auto_force_model")? {
        cfg.auto_force_model = v;
    }
    if let Some(s) = get_str(d, "output_representation")? {
        cfg.output_representation = match s.to_ascii_lowercase().as_str() {
            "cartesian" => empyrean::Representation::Cartesian,
            "keplerian" => empyrean::Representation::Keplerian,
            "cometary" => empyrean::Representation::Cometary,
            "spherical" => empyrean::Representation::Spherical,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown output_representation: {other}"
                )));
            }
        };
    }

    Ok(cfg)
}

// ══════════════════════════════════════════════════════════
//  PropagationConfig dict ↔ Rust struct
// ══════════════════════════════════════════════════════════

/// Apply a Python-side nested propagation-config dict (the shape
/// produced by `PropagationConfig.to_dict()`) onto an existing Rust
/// [`empyrean::PropagationConfig`].
///
/// Missing keys / sub-dicts leave the corresponding fields unchanged,
/// so callers can pass a partial dict to override just a few knobs on
/// top of `PropagationConfig::default()`.
fn apply_propagation_config_dict(
    cfg: &mut empyrean::PropagationConfig,
    d: &Bound<'_, PyDict>,
) -> PyResult<()> {
    fn get_str(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
        match d.get_item(key)? {
            Some(v) => Ok(Some(v.extract()?)),
            None => Ok(None),
        }
    }
    fn get_bool(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<bool>> {
        match d.get_item(key)? {
            Some(v) => Ok(Some(v.extract()?)),
            None => Ok(None),
        }
    }
    fn get_f64(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<f64>> {
        match d.get_item(key)? {
            Some(v) => match v.extract::<Option<f64>>() {
                Ok(opt) => Ok(opt),
                Err(_) => Ok(Some(v.extract()?)),
            },
            None => Ok(None),
        }
    }
    fn get_usize(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<usize>> {
        match d.get_item(key)? {
            Some(v) => match v.extract::<Option<usize>>() {
                Ok(opt) => Ok(opt),
                Err(_) => Ok(Some(v.extract()?)),
            },
            None => Ok(None),
        }
    }
    fn get_dict<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<Bound<'py, PyDict>>> {
        match d.get_item(key)? {
            Some(v) => Ok(Some(v.extract()?)),
            None => Ok(None),
        }
    }

    if let Some(s) = get_str(d, "force_model")? {
        cfg.force_model = match s.to_ascii_lowercase().as_str() {
            "approximate" => empyrean::ForceModelTier::Approximate,
            "basic" => empyrean::ForceModelTier::Basic,
            "standard" => empyrean::ForceModelTier::Standard,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown force_model: {other}"
                )));
            }
        };
    }
    if let Some(v) = d.get_item("excluded_perturbers_naif")? {
        let list: Vec<i32> = v.extract()?;
        cfg.excluded_perturbers = list
            .into_iter()
            .map(|naif| {
                empyrean::Origin::from_naif_id(naif).ok_or_else(|| {
                    PyValueError::new_err(format!("unknown NAIF id in excluded_perturbers: {naif}"))
                })
            })
            .collect::<PyResult<_>>()?;
    }
    if let Some(s) = get_str(d, "uncertainty_method")? {
        cfg.uncertainty_method = match s.to_ascii_lowercase().as_str() {
            "first_order" => empyrean::UncertaintyMethod::FirstOrder,
            "second_order" => empyrean::UncertaintyMethod::SecondOrder,
            // The AUTO uncertainty method. Matches the flat-arg tag-4 path so both the
            // `uncertainty_method=AUTO` sugar and a config carrying it
            // resolve to the same wrapper variant — without this arm the
            // wire dict silently overrode the flat-arg auto() with
            // first_order (empyrean-uogb).
            "auto" => empyrean::UncertaintyMethod::auto(),
            // Sigma-point / Monte Carlo / Gaussian Mixture map onto
            // FirstOrder at the wrapper level; their per-method params
            // travel via separate flat args on `_propagate`. The
            // wrapper's UncertaintyMethod enum only carries
            // First/Second; villeneuve's parametric variants are
            // selected at integration-time from those flat args.
            "sigma_point" | "monte_carlo" | "gaussian_mixture" => {
                empyrean::UncertaintyMethod::FirstOrder
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown uncertainty_method: {other}"
                )));
            }
        };
    }
    if let Some(v) = get_bool(d, "compute_stm")? {
        cfg.compute_stm = v;
    }
    if let Some(s) = get_str(d, "frame")? {
        cfg.frame = match s.to_ascii_lowercase().as_str() {
            "icrf" => empyrean::Frame::ICRF,
            "eclipticj2000" | "ecliptic_j2000" | "ecliptic" => empyrean::Frame::EclipticJ2000,
            "itrf93" | "itrf_93" => empyrean::Frame::ITRF93,
            other => {
                return Err(PyValueError::new_err(format!("unknown frame: {other}")));
            }
        };
    }
    if let Some(events) = get_dict(d, "events")? {
        if let Some(v) = get_bool(&events, "close_approaches")? {
            cfg.events.close_approaches = v;
        }
        if let Some(v) = get_bool(&events, "impacts")? {
            cfg.events.impacts = v;
        }
        if let Some(v) = get_bool(&events, "atmospheric")? {
            cfg.events.atmospheric = v;
        }
        if let Some(v) = get_bool(&events, "possible_impacts")? {
            cfg.events.possible_impacts = v;
        }
        if let Some(v) = get_bool(&events, "shadow_events")? {
            cfg.events.shadow_events = v;
        }
        if let Some(v) = events.get_item("body_filter_naif")? {
            let list: Vec<i32> = v.extract()?;
            cfg.events.body_filter = list
                .into_iter()
                .map(|naif| {
                    empyrean::Origin::from_naif_id(naif).ok_or_else(|| {
                        PyValueError::new_err(format!("unknown NAIF id in body_filter: {naif}"))
                    })
                })
                .collect::<PyResult<_>>()?;
        }
        if let Some(v) = get_bool(&events, "dense_output")? {
            cfg.events.dense_output = v;
        }
        if let Some(v) = get_f64(&events, "dense_output_cadence_days")? {
            cfg.events.dense_output_cadence_days = v;
        }
    }
    if let Some(diag) = get_dict(d, "diagnostics")? {
        if let Some(v) = get_bool(&diag, "sensitivity")? {
            cfg.diagnostics.sensitivity = v;
        }
        if let Some(v) = get_bool(&diag, "nonlinearity")? {
            cfg.diagnostics.nonlinearity = v;
        }
        if let Some(v) = get_bool(&diag, "lyapunov")? {
            cfg.diagnostics.lyapunov = v;
        }
        if let Some(v) = get_bool(&diag, "keyholes")? {
            cfg.diagnostics.keyholes = v;
        }
        if let Some(v) = get_bool(&diag, "bifurcations")? {
            cfg.diagnostics.bifurcations = v;
        }
        if let Some(v) = get_usize(&diag, "sample_stride")? {
            cfg.diagnostics.sample_stride = v;
        }
        // Threshold fields are Optional<f64> — Python None becomes None,
        // a number becomes Some(value).
        cfg.diagnostics.sensitivity_threshold =
            get_f64(&diag, "sensitivity_threshold")?.or(cfg.diagnostics.sensitivity_threshold);
        cfg.diagnostics.lyapunov_threshold =
            get_f64(&diag, "lyapunov_threshold")?.or(cfg.diagnostics.lyapunov_threshold);
        cfg.diagnostics.nonlinearity_threshold =
            get_f64(&diag, "nonlinearity_threshold")?.or(cfg.diagnostics.nonlinearity_threshold);
    }
    if let Some(v) = get_usize(d, "num_threads")? {
        cfg.num_threads = std::num::NonZeroUsize::new(v);
    }
    if let Some(adv) = get_dict(d, "advanced")? {
        if let Some(v) = get_str(&adv, "integrator")? {
            // Case-insensitive match — accepts "GR15", "gr15", "Gr15".
            cfg.advanced.integrator = match v.to_lowercase().as_str() {
                "gr15" => empyrean::IntegratorChoice::GR15,
                "dop853" => empyrean::IntegratorChoice::DOP853,
                other => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "advanced.integrator = {other:?}; valid choices are \"GR15\" \
                         (default) and \"DOP853\". IAS15 is not built into the empyrean \
                         distribution."
                    )));
                }
            };
        }
        if let Some(v) = get_f64(&adv, "epsilon")? {
            cfg.advanced.epsilon = v;
        }
        cfg.advanced.dt_initial = get_f64(&adv, "dt_initial")?.or(cfg.advanced.dt_initial);
        cfg.advanced.dt_min = get_f64(&adv, "dt_min")?.or(cfg.advanced.dt_min);
        if let Some(v) = get_f64(&adv, "encounter_timescale_divisor")? {
            cfg.advanced.encounter_timescale_divisor = v;
        }
        if let Some(v) = get_usize(&adv, "max_steps")? {
            cfg.advanced.max_steps = v;
        }
        if let Some(v) = get_usize(&adv, "max_dense_steps")? {
            cfg.advanced.max_dense_steps = v;
        }
        if let Some(v) = get_bool(&adv, "cache_integrator_steps")? {
            cfg.advanced.cache_integrator_steps = v;
        }
        if let Some(os) = get_dict(&adv, "origin_switching")? {
            if let Some(v) = get_bool(&os, "enabled")? {
                cfg.advanced.origin_switching.enabled = v;
            }
            if let Some(v) = get_f64(&os, "hysteresis")? {
                cfg.advanced.origin_switching.hysteresis = v;
            }
        }
    }
    Ok(())
}

/// Build an [`empyrean::EphemerisConfig`] from a Python-side nested dict
/// (the shape produced by `EphemerisConfig.to_dict()`).
fn build_ephemeris_config_from_dict(d: &Bound<'_, PyDict>) -> PyResult<empyrean::EphemerisConfig> {
    let mut cfg = empyrean::EphemerisConfig::default();
    if let Some(prop) = d.get_item("propagation")? {
        let prop_d: Bound<'_, PyDict> = prop.extract()?;
        apply_propagation_config_dict(&mut cfg.propagation, &prop_d)?;
    }
    if let Some(v) = d.get_item("max_light_time_iterations")? {
        cfg.max_light_time_iterations = v.extract()?;
    }
    if let Some(v) = d.get_item("light_time_tolerance_days")? {
        cfg.light_time_tolerance_days = v.extract()?;
    }
    if let Some(v) = d.get_item("compute_diagnostics")? {
        cfg.compute_diagnostics = v.extract()?;
    }
    Ok(cfg)
}

// ══════════════════════════════════════════════════════════
//  Time-scale and ISO 8601 conversion
// ══════════════════════════════════════════════════════════

fn parse_scale(s: &str) -> PyResult<empyrean::TimeScale> {
    empyrean::TimeScale::from_str(s).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn _convert_epochs<'py>(
    py: Python<'py>,
    mjd: PyReadonlyArray1<'py, f64>,
    from_scale: &str,
    to_scale: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let from = parse_scale(from_scale)?;
    let to = parse_scale(to_scale)?;
    let arr = mjd.as_array();
    let mut out = Array1::<f64>::zeros(arr.len());
    if from == to {
        out.assign(&arr.to_owned());
        return Ok(out.into_pyarray(py));
    }
    for (i, m) in arr.iter().enumerate() {
        // Round-trip via ISO is overkill for an array conversion. We
        // instead format → re-parse only when we have to: the raw
        // MJD-scale converters live behind ISO-format. For TDB↔UTC
        // we use the native converter via ISO formatted at the source
        // and re-read at the destination.
        let iso = empyrean::mjd_to_iso(*m, from).map_err(to_pyerr)?;
        out[i] = empyrean::iso_to_mjd(&iso, to).map_err(to_pyerr)?;
    }
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn _iso_to_mjd<'py>(
    py: Python<'py>,
    iso: Vec<String>,
    scale: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let target = parse_scale(scale)?;
    let mut out = Array1::<f64>::zeros(iso.len());
    for (i, s) in iso.iter().enumerate() {
        out[i] = empyrean::iso_to_mjd(s, target).map_err(to_pyerr)?;
    }
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn _mjd_to_iso(mjd: PyReadonlyArray1<'_, f64>, scale: &str) -> PyResult<Vec<String>> {
    let source = parse_scale(scale)?;
    let arr = mjd.as_array();
    let mut out = Vec::with_capacity(arr.len());
    for m in arr.iter() {
        out.push(empyrean::mjd_to_iso(*m, source).map_err(to_pyerr)?);
    }
    Ok(out)
}

// ══════════════════════════════════════════════════════════
//  _split_gaussian
// ══════════════════════════════════════════════════════════

#[pyfunction]
fn _split_gaussian<'py>(
    py: Python<'py>,
    mean: PyReadonlyArray1<'py, f64>,
    covariance: PyReadonlyArray2<'py, f64>,
    k: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let mean_arr = mean.as_array();
    let cov_arr = covariance.as_array();

    if mean_arr.len() != 6 {
        return Err(PyValueError::new_err("mean must have 6 elements"));
    }
    if cov_arr.shape() != [6, 6] {
        return Err(PyValueError::new_err("covariance must be 6x6"));
    }

    let mut m = [0.0; 6];
    for i in 0..6 {
        m[i] = mean_arr[i];
    }
    let mut c = [[0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            c[i][j] = cov_arr[[i, j]];
        }
    }

    let (eigenvalue, eigenvector) = empyrean::eigenvector_max_6x6(&c).map_err(to_pyerr)?;
    let sigma = eigenvalue.sqrt();
    let weight = 1.0 / k as f64;

    let spacing = if k > 1 {
        2.0 * sigma / (k as f64).sqrt()
    } else {
        0.0
    };
    let offset_var = if k > 1 {
        spacing * spacing * (k * k - 1) as f64 / 12.0
    } else {
        0.0
    };
    let comp_lambda = (eigenvalue - offset_var).max(eigenvalue * 0.01);
    let reduction = eigenvalue - comp_lambda;

    let mut comp_cov = c;
    for i in 0..6 {
        for j in 0..6 {
            comp_cov[i][j] -= reduction * eigenvector[i] * eigenvector[j];
        }
    }

    let mut weights = Array1::<f64>::zeros(k);
    let mut means = Array2::<f64>::zeros((k, 6));
    let mut covs = Array3::<f64>::zeros((k, 6, 6));

    for idx in 0..k {
        let t = if k > 1 {
            (idx as f64 - (k as f64 - 1.0) / 2.0) * spacing
        } else {
            0.0
        };
        weights[idx] = weight;
        for i in 0..6 {
            means[[idx, i]] = m[i] + t * eigenvector[i];
            for j in 0..6 {
                covs[[idx, i, j]] = comp_cov[i][j];
            }
        }
    }

    let dict = PyDict::new(py);
    dict.set_item("weights", PyArray1::from_owned_array(py, weights))?;
    dict.set_item("means", PyArray2::from_owned_array(py, means))?;
    dict.set_item("covariances", PyArray3::from_owned_array(py, covs))?;
    let _ = nan_to_value;
    Ok(dict)
}

// ══════════════════════════════════════════════════════════
//  _eigenvector_max_6x6
// ══════════════════════════════════════════════════════════

#[pyfunction]
fn _eigenvector_max_6x6<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyTuple>> {
    let arr = matrix.as_array();
    if arr.shape() != [6, 6] {
        return Err(PyValueError::new_err("matrix must be 6x6"));
    }

    let mut m = [[0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            m[i][j] = arr[[i, j]];
        }
    }

    let (eigenvalue, eigenvector) = empyrean::eigenvector_max_6x6(&m).map_err(to_pyerr)?;
    let ev_array = Array1::from_vec(eigenvector.to_vec());
    PyTuple::new(
        py,
        &[
            eigenvalue.into_pyobject(py)?.into_any(),
            PyArray1::from_owned_array(py, ev_array).into_any(),
        ],
    )
}

// ══════════════════════════════════════════════════════════
//  Module
// ══════════════════════════════════════════════════════════

// ══════════════════════════════════════════════════════════
//  OrbitBatch <-> Python dict conversion
// ══════════════════════════════════════════════════════════

fn orbit_batch_to_pydict<'py>(
    py: Python<'py>,
    batch: &empyrean::OrbitBatch,
) -> PyResult<Bound<'py, PyDict>> {
    let n = batch.len();
    let mut epoch = Array1::<f64>::zeros(n);
    let mut elements = Array2::<f64>::zeros((n, 6));
    let mut covariance = Array3::<f64>::zeros((n, 6, 6));
    let mut has_covariance = Array1::<u8>::zeros(n);
    let mut representation: Vec<String> = Vec::with_capacity(n);
    let mut frame: Vec<String> = Vec::with_capacity(n);
    let mut origin = Array1::<i32>::zeros(n);
    let mut a1 = Array1::<f64>::zeros(n);
    let mut a2 = Array1::<f64>::zeros(n);
    let mut a3 = Array1::<f64>::zeros(n);
    let mut ng_alpha = Array1::<f64>::zeros(n);
    let mut ng_r0 = Array1::<f64>::zeros(n);
    let mut ng_m = Array1::<f64>::zeros(n);
    let mut ng_n = Array1::<f64>::zeros(n);
    let mut ng_k = Array1::<f64>::zeros(n);
    // NaN = "no thermal-lag delay" (distinct from a real 0.0-day delay).
    let mut non_grav_dt = Array1::<f64>::from_elem(n, f64::NAN);
    let mut phot_h = Array1::<f64>::from_elem(n, f64::NAN);
    let mut phot_slope1 = Array1::<f64>::zeros(n);
    let mut phot_slope2 = Array1::<f64>::zeros(n);
    let mut phot_system: Vec<Option<String>> = Vec::with_capacity(n);
    for (i, orbit) in batch.orbits.iter().enumerate() {
        epoch[i] = orbit.state.epoch.mjd_tdb().map_err(to_pyerr)?;
        for j in 0..6 {
            elements[[i, j]] = orbit.state.elements[j];
        }
        if let Some(cov) = orbit.state.covariance {
            for r in 0..6 {
                for c in 0..6 {
                    covariance[[i, r, c]] = cov[r][c];
                }
            }
            has_covariance[i] = 1;
        }
        representation.push(
            match orbit.state.representation {
                empyrean::Representation::Cartesian => "cartesian",
                empyrean::Representation::Keplerian => "keplerian",
                empyrean::Representation::Cometary => "cometary",
                empyrean::Representation::Spherical => "spherical",
            }
            .to_string(),
        );
        frame.push(
            match orbit.state.frame {
                empyrean::Frame::ICRF => "icrf",
                empyrean::Frame::EclipticJ2000 => "ecliptic_j2000",
                empyrean::Frame::ITRF93 => "itrf93",
            }
            .to_string(),
        );
        origin[i] = orbit.state.origin.naif_id();
        a1[i] = orbit.a1;
        a2[i] = orbit.a2;
        a3[i] = orbit.a3;
        ng_alpha[i] = orbit.ng_alpha;
        ng_r0[i] = orbit.ng_r0;
        ng_m[i] = orbit.ng_m;
        ng_n[i] = orbit.ng_n;
        ng_k[i] = orbit.ng_k;
        non_grav_dt[i] = orbit.non_grav_dt.unwrap_or(f64::NAN);
        match orbit.phot_system {
            Some(empyrean::PhaseFunction::HG) => {
                phot_h[i] = orbit.h_mag;
                phot_slope1[i] = orbit.slope1;
                phot_slope2[i] = orbit.slope2;
                phot_system.push(Some("HG".to_string()));
            }
            Some(empyrean::PhaseFunction::HG1G2) => {
                phot_h[i] = orbit.h_mag;
                phot_slope1[i] = orbit.slope1;
                phot_slope2[i] = orbit.slope2;
                phot_system.push(Some("HG1G2".to_string()));
            }
            Some(empyrean::PhaseFunction::HG12) => {
                phot_h[i] = orbit.h_mag;
                phot_slope1[i] = orbit.slope1;
                phot_slope2[i] = orbit.slope2;
                phot_system.push(Some("HG12".to_string()));
            }
            None => {
                phot_system.push(None);
            }
        }
    }
    let dict = PyDict::new(py);
    dict.set_item("orbit_ids", batch.orbit_ids.clone())?;
    dict.set_item("object_ids", batch.object_ids.clone())?;
    dict.set_item("epoch_mjd_tdb", PyArray1::from_owned_array(py, epoch))?;
    dict.set_item("elements", PyArray2::from_owned_array(py, elements))?;
    dict.set_item("covariance", PyArray3::from_owned_array(py, covariance))?;
    dict.set_item(
        "has_covariance",
        PyArray1::from_owned_array(py, has_covariance),
    )?;
    dict.set_item("representation", representation)?;
    dict.set_item("frame", frame)?;
    dict.set_item("origin", PyArray1::from_owned_array(py, origin))?;
    dict.set_item("a1", PyArray1::from_owned_array(py, a1))?;
    dict.set_item("a2", PyArray1::from_owned_array(py, a2))?;
    dict.set_item("a3", PyArray1::from_owned_array(py, a3))?;
    dict.set_item("ng_alpha", PyArray1::from_owned_array(py, ng_alpha))?;
    dict.set_item("ng_r0", PyArray1::from_owned_array(py, ng_r0))?;
    dict.set_item("ng_m", PyArray1::from_owned_array(py, ng_m))?;
    dict.set_item("ng_n", PyArray1::from_owned_array(py, ng_n))?;
    dict.set_item("ng_k", PyArray1::from_owned_array(py, ng_k))?;
    dict.set_item("non_grav_dt", PyArray1::from_owned_array(py, non_grav_dt))?;
    dict.set_item("phot_h", PyArray1::from_owned_array(py, phot_h))?;
    dict.set_item("phot_slope1", PyArray1::from_owned_array(py, phot_slope1))?;
    dict.set_item("phot_slope2", PyArray1::from_owned_array(py, phot_slope2))?;
    dict.set_item("phot_system", phot_system)?;
    Ok(dict)
}

fn pydict_to_orbit_batch<'py>(dict: &Bound<'py, PyDict>) -> PyResult<empyrean::OrbitBatch> {
    let orbit_ids: Vec<String> = dict
        .get_item("orbit_ids")?
        .ok_or_else(|| PyValueError::new_err("missing 'orbit_ids' key"))?
        .extract()?;
    let object_ids: Vec<Option<String>> = dict
        .get_item("object_ids")?
        .ok_or_else(|| PyValueError::new_err("missing 'object_ids' key"))?
        .extract()?;
    let epoch_arr: PyReadonlyArray1<f64> = dict
        .get_item("epoch_mjd_tdb")?
        .ok_or_else(|| PyValueError::new_err("missing 'epoch_mjd_tdb' key"))?
        .extract()?;
    let elements_arr: PyReadonlyArray2<f64> = dict
        .get_item("elements")?
        .ok_or_else(|| PyValueError::new_err("missing 'elements' key"))?
        .extract()?;
    let covariance_arr: Option<PyReadonlyArray3<f64>> = dict
        .get_item("covariance")?
        .map(|o| o.extract())
        .transpose()?;
    let has_covariance_arr: Option<PyReadonlyArray1<u8>> = dict
        .get_item("has_covariance")?
        .map(|o| o.extract())
        .transpose()?;
    let representation: Vec<String> = dict
        .get_item("representation")?
        .ok_or_else(|| PyValueError::new_err("missing 'representation' key"))?
        .extract()?;
    let frame: Vec<String> = dict
        .get_item("frame")?
        .ok_or_else(|| PyValueError::new_err("missing 'frame' key"))?
        .extract()?;
    let origin_arr: PyReadonlyArray1<i32> = dict
        .get_item("origin")?
        .ok_or_else(|| PyValueError::new_err("missing 'origin' key"))?
        .extract()?;
    let n = orbit_ids.len();
    if object_ids.len() != n {
        return Err(PyValueError::new_err(
            "orbit_ids / object_ids length mismatch",
        ));
    }

    let epoch = epoch_arr.as_array();
    let elements = elements_arr.as_array();
    let origin = origin_arr.as_array();
    let cov_view = covariance_arr.as_ref().map(|c| c.as_array());
    let has_cov_view = has_covariance_arr.as_ref().map(|c| c.as_array());
    let a1_view = read_array_or_zero(dict, "a1", n)?;
    let a2_view = read_array_or_zero(dict, "a2", n)?;
    let a3_view = read_array_or_zero(dict, "a3", n)?;
    let g_alpha_view = read_array_or_zero(dict, "ng_alpha", n)?;
    let g_r0_view = read_array_or_zero(dict, "ng_r0", n)?;
    let g_m_view = read_array_or_zero(dict, "ng_m", n)?;
    let g_n_view = read_array_or_zero(dict, "ng_n", n)?;
    let g_k_view = read_array_or_zero(dict, "ng_k", n)?;
    let dt_view = read_array_or_nan(dict, "non_grav_dt", n)?;

    let mut orbits = Vec::with_capacity(n);
    for i in 0..n {
        let rep = match representation[i].to_ascii_lowercase().as_str() {
            "cartesian" => empyrean::Representation::Cartesian,
            "keplerian" => empyrean::Representation::Keplerian,
            "cometary" => empyrean::Representation::Cometary,
            "spherical" => empyrean::Representation::Spherical,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown representation '{other}'"
                )));
            }
        };
        let f = match frame[i].to_ascii_lowercase().as_str() {
            "icrf" => empyrean::Frame::ICRF,
            "ecliptic_j2000" | "eclipticj2000" | "ecliptic" => empyrean::Frame::EclipticJ2000,
            "itrf93" | "itrf_93" => empyrean::Frame::ITRF93,
            other => return Err(PyValueError::new_err(format!("unknown frame '{other}'"))),
        };
        let elements_row = [
            elements[[i, 0]],
            elements[[i, 1]],
            elements[[i, 2]],
            elements[[i, 3]],
            elements[[i, 4]],
            elements[[i, 5]],
        ];
        let mut state = empyrean::CoordinateState {
            epoch: empyrean::Epoch::from_mjd_tdb(epoch[i]),
            elements: elements_row,
            covariance: None,
            representation: rep,
            frame: f,
            origin: origin_from_naif(origin[i])?,
        };
        let want_cov = has_cov_view.map(|h| h[i] != 0).unwrap_or(false);
        if want_cov && let Some(cv) = cov_view.as_ref() {
            let mut m = [[0.0f64; 6]; 6];
            for r in 0..6 {
                for c in 0..6 {
                    m[r][c] = cv[[i, r, c]];
                }
            }
            state.covariance = Some(m);
        }
        orbits.push(empyrean::Orbit {
            // Read-orbits IO path doesn't currently expose orbit_id /
            // object_id at this in-memory call site. Leave unset.
            orbit_id: None,
            object_id: None,
            state,
            a1: a1_view[i],
            a2: a2_view[i],
            a3: a3_view[i],
            ng_alpha: g_alpha_view[i],
            ng_r0: g_r0_view[i],
            ng_m: g_m_view[i],
            ng_n: g_n_view[i],
            ng_k: g_k_view[i],
            // Finite = real thermal-lag delay (incl. a meaningful 0.0);
            // NaN / absent column = no delay (None).
            non_grav_dt: dt_view[i].is_finite().then_some(dt_view[i]),
            // Non-grav covariance is an OD-output concept; the OrbitBatch I/O
            // surface doesn't carry it.
            ng_covariance: None,
            // OrbitBatch I/O surface (parquet/JSON/CSV) does not yet
            // carry photometry — round-tripped orbits come back without
            // it. Use `Orbit::with_photometry` directly when populating
            // batches via the in-process API.
            phot_system: None,
            h_mag: f64::NAN,
            slope1: 0.0,
            slope2: 0.0,
        });
    }
    empyrean::OrbitBatch::new(orbits, orbit_ids, object_ids).map_err(to_pyerr)
}

fn read_array_or_zero(dict: &Bound<'_, PyDict>, key: &str, n: usize) -> PyResult<Vec<f64>> {
    match dict.get_item(key)? {
        Some(obj) => {
            let arr: PyReadonlyArray1<f64> = obj.extract()?;
            let v = arr.as_array().to_vec();
            if v.len() != n {
                return Err(PyValueError::new_err(format!(
                    "'{key}' length {} != orbit count {n}",
                    v.len()
                )));
            }
            Ok(v)
        }
        None => Ok(vec![0.0; n]),
    }
}

// ══════════════════════════════════════════════════════════
//  File I/O — orbits (read + write × parquet/JSON/CSV)
// ══════════════════════════════════════════════════════════

#[pyfunction]
fn _read_orbits_parquet<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyDict>> {
    let batch = py
        .detach(|| empyrean::read_orbits_parquet(path))
        .map_err(to_pyerr)?;
    orbit_batch_to_pydict(py, &batch)
}

#[pyfunction]
fn _read_orbits_json<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyDict>> {
    let batch = py
        .detach(|| empyrean::read_orbits_json(path))
        .map_err(to_pyerr)?;
    orbit_batch_to_pydict(py, &batch)
}

#[pyfunction]
fn _read_orbits_csv<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyDict>> {
    let batch = py
        .detach(|| empyrean::read_orbits_csv(path))
        .map_err(to_pyerr)?;
    orbit_batch_to_pydict(py, &batch)
}

#[pyfunction]
fn _write_orbits_parquet<'py>(
    py: Python<'py>,
    path: &str,
    batch: &Bound<'py, PyDict>,
) -> PyResult<()> {
    let owned = pydict_to_orbit_batch(batch)?;
    py.detach(|| empyrean::write_orbits_parquet(path, &owned))
        .map_err(to_pyerr)
}

#[pyfunction]
fn _write_orbits_json<'py>(
    py: Python<'py>,
    path: &str,
    batch: &Bound<'py, PyDict>,
) -> PyResult<()> {
    let owned = pydict_to_orbit_batch(batch)?;
    py.detach(|| empyrean::write_orbits_json(path, &owned))
        .map_err(to_pyerr)
}

#[pyfunction]
fn _write_orbits_csv<'py>(py: Python<'py>, path: &str, batch: &Bound<'py, PyDict>) -> PyResult<()> {
    let owned = pydict_to_orbit_batch(batch)?;
    py.detach(|| empyrean::write_orbits_csv(path, &owned))
        .map_err(to_pyerr)
}

// ══════════════════════════════════════════════════════════
//  File I/O — ephemeris write × parquet/JSON/CSV
// ══════════════════════════════════════════════════════════

fn pydict_to_ephemeris(dict: &Bound<'_, PyDict>) -> PyResult<Vec<empyrean::EphemerisEntry>> {
    let orbit_ids: Vec<String> = dict
        .get_item("orbit_ids")?
        .ok_or_else(|| PyValueError::new_err("missing 'orbit_ids'"))?
        .extract()?;
    let obs_codes: Vec<String> = dict
        .get_item("obs_codes")?
        .ok_or_else(|| PyValueError::new_err("missing 'obs_codes'"))?
        .extract()?;
    let epochs: PyReadonlyArray1<f64> = dict
        .get_item("epoch_mjd_tdb")?
        .ok_or_else(|| PyValueError::new_err("missing 'epoch_mjd_tdb'"))?
        .extract()?;
    let ra: PyReadonlyArray1<f64> = dict
        .get_item("ra_deg")?
        .ok_or_else(|| PyValueError::new_err("missing 'ra_deg'"))?
        .extract()?;
    let dec: PyReadonlyArray1<f64> = dict
        .get_item("dec_deg")?
        .ok_or_else(|| PyValueError::new_err("missing 'dec_deg'"))?
        .extract()?;
    let n = orbit_ids.len();
    let rho = read_array_or_zero(dict, "rho_au", n)?;
    let vrho = read_array_or_zero(dict, "vrho_au_day", n)?;
    let vra = read_array_or_zero(dict, "vra_deg_day", n)?;
    let vdec = read_array_or_zero(dict, "vdec_deg_day", n)?;
    let lt = read_array_or_nan(dict, "light_time_days", n)?;
    let pa = read_array_or_nan(dict, "phase_angle_deg", n)?;
    let elong = read_array_or_nan(dict, "elongation_deg", n)?;
    let hdist = read_array_or_nan(dict, "heliocentric_distance_au", n)?;
    let mag = read_array_or_nan(dict, "mag", n)?;
    let mag_s = read_array_or_nan(dict, "mag_sigma", n)?;
    let zenith = read_array_or_nan(dict, "zenith_angle", n)?;
    let azimuth = read_array_or_nan(dict, "azimuth", n)?;
    let hour_angle = read_array_or_nan(dict, "hour_angle", n)?;
    let lunar_elong = read_array_or_nan(dict, "lunar_elongation", n)?;
    let position_angle = read_array_or_nan(dict, "position_angle", n)?;
    let sky_rate = read_array_or_nan(dict, "sky_rate", n)?;
    let epochs = epochs.as_array();
    let ra = ra.as_array();
    let dec = dec.as_array();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(empyrean::EphemerisEntry {
            orbit_id: orbit_ids[i].clone(),
            epoch: empyrean::Epoch::from_mjd_tdb(epochs[i]),
            ra_deg: ra[i],
            dec_deg: dec[i],
            rho_au: rho[i],
            vrho_au_day: vrho[i],
            vra_deg_day: vra[i],
            vdec_deg_day: vdec[i],
            light_time_days: lt[i],
            phase_angle_deg: pa[i],
            elongation_deg: elong[i],
            heliocentric_distance_au: hdist[i],
            mag: mag[i],
            mag_sigma: mag_s[i],
            zenith_angle_deg: zenith[i],
            azimuth_deg: azimuth[i],
            hour_angle_deg: hour_angle[i],
            lunar_elongation_deg: lunar_elong[i],
            position_angle_deg: position_angle[i],
            sky_rate_deg_day: sky_rate[i],
            obs_code: obs_codes[i].clone(),
        });
    }
    Ok(out)
}

fn read_array_or_nan(dict: &Bound<'_, PyDict>, key: &str, n: usize) -> PyResult<Vec<f64>> {
    match dict.get_item(key)? {
        Some(obj) => {
            let arr: PyReadonlyArray1<f64> = obj.extract()?;
            let v = arr.as_array().to_vec();
            if v.len() != n {
                return Err(PyValueError::new_err(format!(
                    "'{key}' length {} != n {n}",
                    v.len()
                )));
            }
            Ok(v)
        }
        None => Ok(vec![f64::NAN; n]),
    }
}

#[pyfunction]
fn _write_ephemeris_parquet<'py>(
    py: Python<'py>,
    path: &str,
    entries: &Bound<'py, PyDict>,
) -> PyResult<()> {
    let owned = pydict_to_ephemeris(entries)?;
    py.detach(|| empyrean::write_ephemeris_parquet(path, &owned))
        .map_err(to_pyerr)
}

#[pyfunction]
fn _write_ephemeris_json<'py>(
    py: Python<'py>,
    path: &str,
    entries: &Bound<'py, PyDict>,
) -> PyResult<()> {
    let owned = pydict_to_ephemeris(entries)?;
    py.detach(|| empyrean::write_ephemeris_json(path, &owned))
        .map_err(to_pyerr)
}

#[pyfunction]
fn _write_ephemeris_csv<'py>(
    py: Python<'py>,
    path: &str,
    entries: &Bound<'py, PyDict>,
) -> PyResult<()> {
    let owned = pydict_to_ephemeris(entries)?;
    py.detach(|| empyrean::write_ephemeris_csv(path, &owned))
        .map_err(to_pyerr)
}

// ══════════════════════════════════════════════════════════
//  File I/O — events write × parquet/JSON/CSV
// ══════════════════════════════════════════════════════════

fn pydict_to_events(dict: &Bound<'_, PyDict>) -> PyResult<Vec<empyrean::Event>> {
    let orbit_ids: Vec<String> = dict
        .get_item("orbit_ids")?
        .ok_or_else(|| PyValueError::new_err("missing 'orbit_ids'"))?
        .extract()?;
    let event_types: Vec<String> = dict
        .get_item("event_types")?
        .ok_or_else(|| PyValueError::new_err("missing 'event_types'"))?
        .extract()?;
    let bodies: Vec<String> = dict
        .get_item("bodies")?
        .ok_or_else(|| PyValueError::new_err("missing 'bodies'"))?
        .extract()?;
    let body_naif_ids: PyReadonlyArray1<i32> = dict
        .get_item("body_naif_ids")?
        .ok_or_else(|| PyValueError::new_err("missing 'body_naif_ids'"))?
        .extract()?;
    let epochs: PyReadonlyArray1<f64> = dict
        .get_item("epochs")?
        .ok_or_else(|| PyValueError::new_err("missing 'epochs'"))?
        .extract()?;
    // Optional in older callers — older event dicts (pre v0.7.x) didn't
    // carry object_ids. Default to empty strings so I/O round-trips
    // without breaking the existing wire format.
    let object_ids: Vec<String> = match dict.get_item("object_ids")? {
        Some(o) => o.extract().unwrap_or_else(|_| Vec::new()),
        None => Vec::new(),
    };
    let n = orbit_ids.len();
    let dist_au = read_array_or_nan(dict, "distance_au", n)?;
    let dist_km = read_array_or_nan(dict, "distance_km", n)?;
    let rel_v = read_array_or_nan(dict, "relative_velocity_au_day", n)?;
    let body_naif_ids = body_naif_ids.as_array();
    let epochs = epochs.as_array();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        // Prefer the NAIF id for round-tripping (it's authoritative);
        // fall back to parsing the canonical name for older inputs
        // that didn't carry the int. -1 / 0 / unparseable → None
        // (non-body event).
        let body = if body_naif_ids[i] > 0 {
            empyrean::Origin::from_naif_id(body_naif_ids[i])
        } else if !bodies[i].is_empty() {
            bodies[i].parse().ok()
        } else {
            None
        };
        out.push(empyrean::Event {
            event_type: event_types[i].clone(),
            orbit_id: orbit_ids[i].clone(),
            object_id: object_ids.get(i).cloned().unwrap_or_default(),
            body,
            epoch: empyrean::Epoch::from_mjd_tdb(epochs[i]),
            distance_au: dist_au[i],
            distance_km: dist_km[i],
            relative_velocity_au_day: rel_v[i],
            // The events parquet/JSON/CSV write surface does not yet carry
            // the capture / impact / covariance-regime payload columns
            // (persisting them needs the C-ABI parquet schema extended);
            // sentinel-fill on this round-trip read. The live propagate()
            // event stream DOES carry them. Tracked: empyrean-evwr.
            two_body_energy: f64::NAN,
            jacobi_constant: f64::NAN,
            jacobi_constant_sigma: f64::NAN,
            jacobi_constant_l1: f64::NAN,
            jacobi_constant_l2: f64::NAN,
            n_periapses: None,
            impact_latitude_deg: f64::NAN,
            impact_longitude_deg: f64::NAN,
            impact_altitude_km: f64::NAN,
            shadow_fraction: f64::NAN,
            illumination: f64::NAN,
            relative_x: f64::NAN,
            relative_y: f64::NAN,
            relative_z: f64::NAN,
            relative_vx: f64::NAN,
            relative_vy: f64::NAN,
            relative_vz: f64::NAN,
            effective_radius_au: f64::NAN,
            effective_radius_km: f64::NAN,
            sigma_distance_au: f64::NAN,
            ip_linear: f64::NAN,
            ip_second_order: f64::NAN,
            nonlinearity: f64::NAN,
            ip_agm: f64::NAN,
            ip_mc: f64::NAN,
            previous_kind: None,
            regime_resolved_kind: None,
            kappa: f64::NAN,
            threshold_below: f64::NAN,
            threshold_above: f64::NAN,
        });
    }
    Ok(out)
}

#[pyfunction]
fn _write_events_parquet<'py>(
    py: Python<'py>,
    path: &str,
    events: &Bound<'py, PyDict>,
) -> PyResult<()> {
    let owned = pydict_to_events(events)?;
    py.detach(|| empyrean::write_events_parquet(path, &owned))
        .map_err(to_pyerr)
}

#[pyfunction]
fn _write_events_json<'py>(
    py: Python<'py>,
    path: &str,
    events: &Bound<'py, PyDict>,
) -> PyResult<()> {
    let owned = pydict_to_events(events)?;
    py.detach(|| empyrean::write_events_json(path, &owned))
        .map_err(to_pyerr)
}

#[pyfunction]
fn _write_events_csv<'py>(
    py: Python<'py>,
    path: &str,
    events: &Bound<'py, PyDict>,
) -> PyResult<()> {
    let owned = pydict_to_events(events)?;
    py.detach(|| empyrean::write_events_csv(path, &owned))
        .map_err(to_pyerr)
}

// ══════════════════════════════════════════════════════════
//  File I/O — residuals write × parquet/JSON/CSV
// ══════════════════════════════════════════════════════════

fn pydict_to_residuals(dict: &Bound<'_, PyDict>) -> PyResult<Vec<empyrean::ObservationResidual>> {
    let ra: PyReadonlyArray1<f64> = dict
        .get_item("ra_residuals_arcsec")?
        .ok_or_else(|| PyValueError::new_err("missing 'ra_residuals_arcsec'"))?
        .extract()?;
    let dec: PyReadonlyArray1<f64> = dict
        .get_item("dec_residuals_arcsec")?
        .ok_or_else(|| PyValueError::new_err("missing 'dec_residuals_arcsec'"))?
        .extract()?;
    let chi2: PyReadonlyArray1<f64> = dict
        .get_item("chi2")?
        .ok_or_else(|| PyValueError::new_err("missing 'chi2'"))?
        .extract()?;
    let prob: PyReadonlyArray1<f64> = dict
        .get_item("probability")?
        .ok_or_else(|| PyValueError::new_err("missing 'probability'"))?
        .extract()?;
    let selected: PyReadonlyArray1<u8> = dict
        .get_item("selected")?
        .ok_or_else(|| PyValueError::new_err("missing 'selected'"))?
        .extract()?;
    let ra = ra.as_array();
    let dec = dec.as_array();
    let chi2 = chi2.as_array();
    let prob = prob.as_array();
    let selected = selected.as_array();
    let n = ra.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        // The Python residuals-write path passes minimal residual data
        // (RA/Dec/χ²/probability/selected). Everything else fills with
        // NaN / NotEvaluated so the C ABI sees a uniform absent
        // sentinel; downstream serializers (parquet/json/csv) treat
        // NaN as a missing-cell write.
        out.push(empyrean::ObservationResidual {
            obs_id: String::new(),
            obs_code: String::new(),
            ast_cat: None,
            epoch: empyrean::Epoch::from_mjd_tdb(f64::NAN),
            ra_residual_arcsec: ra[i],
            dec_residual_arcsec: dec[i],
            chi2: chi2[i],
            dof: 2,
            probability: prob[i],
            selected: selected[i] != 0,
            residual_cov_ra: f64::NAN,
            residual_cov_dec: f64::NAN,
            residual_cov_corr: f64::NAN,
            rejection_reason: empyrean::RejectionReason::NotEvaluated,
            rejection_criterion: f64::NAN,
            rejection_threshold: f64::NAN,
            rejection_effective_threshold: f64::NAN,
            rejection_information_loss: f64::NAN,
            cooks_distance: f64::NAN,
            leverage: f64::NAN,
            fractional_information: f64::NAN,
            along_track_arcsec: f64::NAN,
            cross_track_arcsec: f64::NAN,
            along_track_error_arcsec: f64::NAN,
            cross_track_error_arcsec: f64::NAN,
            track_position_angle_deg: f64::NAN,
        });
    }
    Ok(out)
}

#[pyfunction]
fn _write_residuals_parquet<'py>(
    py: Python<'py>,
    path: &str,
    residuals: &Bound<'py, PyDict>,
) -> PyResult<()> {
    let owned = pydict_to_residuals(residuals)?;
    py.detach(|| empyrean::write_residuals_parquet(path, &owned))
        .map_err(to_pyerr)
}

#[pyfunction]
fn _write_residuals_json<'py>(
    py: Python<'py>,
    path: &str,
    residuals: &Bound<'py, PyDict>,
) -> PyResult<()> {
    let owned = pydict_to_residuals(residuals)?;
    py.detach(|| empyrean::write_residuals_json(path, &owned))
        .map_err(to_pyerr)
}

#[pyfunction]
fn _write_residuals_csv<'py>(
    py: Python<'py>,
    path: &str,
    residuals: &Bound<'py, PyDict>,
) -> PyResult<()> {
    let owned = pydict_to_residuals(residuals)?;
    py.detach(|| empyrean::write_residuals_csv(path, &owned))
        .map_err(to_pyerr)
}

// ══════════════════════════════════════════════════════════
//  Session — stateful OD with mask + history
// ══════════════════════════════════════════════════════════

/// Stateful orbit-determination session.
///
/// Wraps `empyrean::Session`. Owns its observation set (consumed at
/// construction from a Python dict that was previously read via
/// `_read_ades`), the mask state, and the fit history. The session is
/// **not** thread-safe — Python's GIL is sufficient for single-threaded
/// use; share via `multiprocessing` rather than threads.
#[pyclass(name = "Session", unsendable)]
struct PySession {
    inner: empyrean::Session,
}

#[pymethods]
impl PySession {
    /// Construct a session by parsing ADES PSV / MPC80 content.
    /// `force_model` matches the propagation tier ints (0/1/2).
    /// Other ODConfig knobs use upstream defaults; tweak via the
    /// dedicated wrappers if needed.
    #[new]
    #[pyo3(signature = (ades_path_or_content, config_dict))]
    fn new(ades_path_or_content: &str, config_dict: &Bound<'_, PyDict>) -> PyResult<Self> {
        let ctx = get_context()?;
        let observations = ctx.read_ades(ades_path_or_content).map_err(to_pyerr)?;
        let cfg = build_od_config_from_dict(config_dict)?;
        let session = empyrean::Session::new(observations, cfg).map_err(to_pyerr)?;
        Ok(Self { inner: session })
    }

    /// Construct a session from a pre-parsed observations flat-dict
    /// (the same shape `_determine` consumes). Used by the Python
    /// :class:`Session` to accept an :class:`ADESObservations` quivr
    /// table directly without an ADES-PSV round-trip.
    #[staticmethod]
    #[pyo3(signature = (obs_dict, config_dict))]
    fn from_observations_dict(
        obs_dict: &Bound<'_, PyDict>,
        config_dict: &Bound<'_, PyDict>,
    ) -> PyResult<Self> {
        let ctx = get_context()?;
        let observations = build_observations(ctx, obs_dict)?;
        let cfg = build_od_config_from_dict(config_dict)?;
        let session = empyrean::Session::new(observations, cfg).map_err(to_pyerr)?;
        Ok(Self { inner: session })
    }

    fn n_observations(&self) -> usize {
        self.inner.n_observations()
    }

    fn n_masked(&self) -> usize {
        self.inner.n_masked()
    }

    fn n_active(&self) -> usize {
        self.inner.n_active()
    }

    fn mask(&mut self, idx: usize) -> PyResult<()> {
        self.inner.mask(idx).map_err(to_pyerr)
    }

    fn unmask(&mut self, idx: usize) -> PyResult<()> {
        self.inner.unmask(idx).map_err(to_pyerr)
    }

    fn unmask_all(&mut self) -> PyResult<()> {
        self.inner.unmask_all().map_err(to_pyerr)
    }

    fn is_masked(&self, idx: usize) -> bool {
        self.inner.is_masked(idx)
    }

    /// Run an OD refine using the current mask state. Pushes the new
    /// fit onto the session's history and returns it as a dict matching
    /// the `_determine` output shape.
    fn refine<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let ctx = get_context()?;
        let result = py.detach(|| self.inner.refine(ctx)).map_err(to_pyerr)?;
        determine_result_to_pydict(py, &result)
    }

    fn history_len(&self) -> usize {
        self.inner.history_len()
    }

    fn history<'py>(&self, py: Python<'py>, idx: usize) -> PyResult<Bound<'py, PyDict>> {
        let entry = self.inner.history(idx).map_err(to_pyerr)?;
        determine_result_to_pydict(py, &entry)
    }

    /// Diff the current fit against the prior_idx-th history entry.
    /// Returns a dict with keys: reduced_chi2_delta, iterations_delta,
    /// n_observations_delta, update_norm_current, update_norm_prior.
    fn diff<'py>(&self, py: Python<'py>, prior_idx: usize) -> PyResult<Bound<'py, PyDict>> {
        let diff = self.inner.diff(prior_idx).map_err(to_pyerr)?;
        let dict = PyDict::new(py);
        dict.set_item("reduced_chi2_delta", diff.reduced_chi2_delta)?;
        dict.set_item("iterations_delta", diff.iterations_delta)?;
        dict.set_item("n_observations_delta", diff.n_observations_delta)?;
        dict.set_item("update_norm_current", diff.update_norm_current)?;
        dict.set_item("update_norm_prior", diff.update_norm_prior)?;
        Ok(dict)
    }
}

fn add_station_biases_to_dict(
    dict: &Bound<'_, PyDict>,
    biases: &[empyrean::StationBias],
) -> PyResult<()> {
    let py = dict.py();
    let n = biases.len();
    let mut obs_codes: Vec<String> = Vec::with_capacity(n);
    let mut n_obs: Vec<usize> = Vec::with_capacity(n);
    let mut bias_ras = Vec::with_capacity(n);
    let mut sigma_ras = Vec::with_capacity(n);
    let mut bias_decs = Vec::with_capacity(n);
    let mut sigma_decs = Vec::with_capacity(n);
    let mut bias_timings: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut sigma_timings: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut significances = Vec::with_capacity(n);
    for b in biases {
        obs_codes.push(b.obs_code.clone());
        n_obs.push(b.n_obs);
        bias_ras.push(b.bias_ra_arcsec);
        sigma_ras.push(b.sigma_ra_arcsec);
        bias_decs.push(b.bias_dec_arcsec);
        sigma_decs.push(b.sigma_dec_arcsec);
        bias_timings.push(b.bias_timing_sec);
        sigma_timings.push(b.sigma_timing_sec);
        significances.push(b.significance);
    }
    dict.set_item("station_bias_obs_codes", obs_codes)?;
    dict.set_item("station_bias_n_obs", n_obs)?;
    dict.set_item("station_bias_ra_arcsec", PyArray1::from_vec(py, bias_ras))?;
    dict.set_item(
        "station_bias_sigma_ra_arcsec",
        PyArray1::from_vec(py, sigma_ras),
    )?;
    dict.set_item("station_bias_dec_arcsec", PyArray1::from_vec(py, bias_decs))?;
    dict.set_item(
        "station_bias_sigma_dec_arcsec",
        PyArray1::from_vec(py, sigma_decs),
    )?;
    dict.set_item("station_bias_timing_sec", bias_timings)?;
    dict.set_item("station_bias_sigma_timing_sec", sigma_timings)?;
    dict.set_item(
        "station_bias_significance",
        PyArray1::from_vec(py, significances),
    )?;
    Ok(())
}

fn determine_result_to_pydict<'py>(
    py: Python<'py>,
    result: &empyrean::DetermineResult,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    // Flat state snapshot (orbit_x/y/z/vx/vy/vz, covariance, frame, origin).
    add_propagated_to_dict(&dict, "", "", &result.state(), "orbit_")?;
    // Fitted **absolute** non-grav so the Python orbit is re-feedable
    // (A1/A2/A3 + g(r) exponents + optional thermal-lag dt). Mirrors the
    // re-feedable `Orbit` the Rust wrapper now returns.
    {
        let o = &result.orbit;
        if o.a1 != 0.0 || o.a2 != 0.0 || o.a3 != 0.0 {
            dict.set_item("orbit_a1", o.a1)?;
            dict.set_item("orbit_a2", o.a2)?;
            dict.set_item("orbit_a3", o.a3)?;
            dict.set_item("orbit_ng_alpha", o.ng_alpha)?;
            dict.set_item("orbit_ng_r0", o.ng_r0)?;
            dict.set_item("orbit_ng_m", o.ng_m)?;
            dict.set_item("orbit_ng_n", o.ng_n)?;
            dict.set_item("orbit_ng_k", o.ng_k)?;
            if let Some(dt) = o.non_grav_dt {
                dict.set_item("orbit_non_grav_dt", dt)?;
            }
            // Fitted non-grav 3×3 covariance, row-major flat (9). Lets the
            // orbit re-feed into a StateAndNonGrav refine (empyrean-wo4n).
            if let Some(c) = o.ng_covariance {
                let flat: Vec<f64> = c.iter().flatten().copied().collect();
                dict.set_item("orbit_non_grav_cov", flat)?;
            }
        }
    }
    add_residuals_to_dict(&dict, &result.residuals)?;
    add_summary_to_dict(&dict, &result.summary, "summary_")?;
    add_acceptability_to_dict(&dict, &result.acceptability, "acceptability_")?;
    dict.set_item("iterations", result.iterations)?;
    dict.set_item("update_norm", result.update_norm)?;
    dict.set_item("converged", result.converged)?;

    // 6×6 fitted covariance, flat row-major.
    let mut cov_flat: Vec<f64> = Vec::with_capacity(36);
    for r in 0..6 {
        for c in 0..6 {
            cov_flat.push(result.covariance[r][c]);
        }
    }
    dict.set_item("covariance", cov_flat)?;
    let cov_rep = match result.covariance_representation {
        empyrean::CovarianceRepresentation::Cartesian => "cartesian",
        empyrean::CovarianceRepresentation::Keplerian => "keplerian",
        empyrean::CovarianceRepresentation::Cometary => "cometary",
        empyrean::CovarianceRepresentation::Spherical => "spherical",
    };
    dict.set_item("covariance_representation", cov_rep)?;

    if let Some(c9) = &result.covariance_9x9 {
        let mut flat = Vec::with_capacity(81);
        for r in 0..9 {
            for c in 0..9 {
                flat.push(c9[r][c]);
            }
        }
        dict.set_item("covariance_9x9", flat)?;
    }
    if let Some(d) = &result.non_grav_delta {
        dict.set_item("non_grav_delta", vec![d[0], d[1], d[2]])?;
    }

    dict.set_item("rejection_passes", result.rejection_passes)?;
    dict.set_item("num_oppositions_fit", result.num_oppositions_fit)?;
    let force_model_str = match result.force_model_used {
        empyrean::ForceModelTier::Approximate => "approximate",
        empyrean::ForceModelTier::Basic => "basic",
        empyrean::ForceModelTier::Standard => "standard",
    };
    dict.set_item("force_model_used", force_model_str)?;
    let solve_for_str = match result.solve_for_used {
        empyrean::SolveForParams::StateOnly => "state_only",
        empyrean::SolveForParams::StateAndNonGrav => "state_and_nongrav",
        empyrean::SolveForParams::Auto => "auto",
    };
    dict.set_item("solve_for_used", solve_for_str)?;

    add_station_biases_to_dict(&dict, &result.station_biases)?;

    Ok(dict)
}

// ══════════════════════════════════════════════════════════
//  Module
// ══════════════════════════════════════════════════════════

#[pymodule]
fn _empyrean_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_initialize, m)?)?;
    m.add_function(wrap_pyfunction!(_download_data, m)?)?;
    m.add_function(wrap_pyfunction!(_default_data_dir, m)?)?;
    m.add_function(wrap_pyfunction!(_version_string, m)?)?;
    m.add_function(wrap_pyfunction!(_versions, m)?)?;
    m.add_function(wrap_pyfunction!(_transform_coordinates, m)?)?;
    m.add_function(wrap_pyfunction!(_query_sbdb, m)?)?;
    m.add_function(wrap_pyfunction!(_query_horizons, m)?)?;
    m.add_function(wrap_pyfunction!(_query_horizons_vectors, m)?)?;
    m.add_function(wrap_pyfunction!(_query_observations, m)?)?;
    m.add_function(wrap_pyfunction!(_query_radar, m)?)?;
    m.add_function(wrap_pyfunction!(_propagate, m)?)?;
    m.add_function(wrap_pyfunction!(_compute_impact_probabilities, m)?)?;
    m.add_function(wrap_pyfunction!(_compute_b_planes, m)?)?;
    m.add_function(wrap_pyfunction!(_get_observers, m)?)?;
    m.add_function(wrap_pyfunction!(_generate_ephemeris, m)?)?;
    m.add_function(wrap_pyfunction!(_get_states, m)?)?;
    m.add_function(wrap_pyfunction!(_read_ades, m)?)?;
    m.add_function(wrap_pyfunction!(_determine, m)?)?;
    m.add_function(wrap_pyfunction!(_evaluate_single, m)?)?;
    m.add_function(wrap_pyfunction!(_refine_single, m)?)?;
    m.add_function(wrap_pyfunction!(_convert_epochs, m)?)?;
    m.add_function(wrap_pyfunction!(_iso_to_mjd, m)?)?;
    m.add_function(wrap_pyfunction!(_mjd_to_iso, m)?)?;
    m.add_function(wrap_pyfunction!(_split_gaussian, m)?)?;
    m.add_function(wrap_pyfunction!(_eigenvector_max_6x6, m)?)?;
    m.add_function(wrap_pyfunction!(_read_orbits_parquet, m)?)?;
    m.add_function(wrap_pyfunction!(_read_orbits_json, m)?)?;
    m.add_function(wrap_pyfunction!(_read_orbits_csv, m)?)?;
    m.add_function(wrap_pyfunction!(_write_orbits_parquet, m)?)?;
    m.add_function(wrap_pyfunction!(_write_orbits_json, m)?)?;
    m.add_function(wrap_pyfunction!(_write_orbits_csv, m)?)?;
    m.add_function(wrap_pyfunction!(_write_ephemeris_parquet, m)?)?;
    m.add_function(wrap_pyfunction!(_write_ephemeris_json, m)?)?;
    m.add_function(wrap_pyfunction!(_write_ephemeris_csv, m)?)?;
    m.add_function(wrap_pyfunction!(_write_events_parquet, m)?)?;
    m.add_function(wrap_pyfunction!(_write_events_json, m)?)?;
    m.add_function(wrap_pyfunction!(_write_events_csv, m)?)?;
    m.add_function(wrap_pyfunction!(_write_residuals_parquet, m)?)?;
    m.add_function(wrap_pyfunction!(_write_residuals_json, m)?)?;
    m.add_function(wrap_pyfunction!(_write_residuals_csv, m)?)?;
    m.add_class::<PySession>()?;
    Ok(())
}
