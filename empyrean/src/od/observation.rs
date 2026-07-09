//! ADES observations: a single [`Observation`] record and an owned
//! [`Observations`] set returned by [`Context::read_ades`](super::Context::read_ades).

use std::ffi::{CStr, CString};

use crate::error::{Error, Result};
use crate::observers::obs_code_from_bytes;

/// A single astrometric observation — full ADES schema.
///
/// In the safe Rust API, [`Observations`] (plural) is the RAII owner of
/// the underlying FFI array. Individual observations here are snapshots
/// materialized from that array — modifying them does not affect the
/// stored observations.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Observation {
    // ── Identification ────────────────────────────────────
    /// IAU permanent designation.
    pub perm_id: Option<String>,
    /// MPC provisional designation.
    pub prov_id: Option<String>,
    /// Observer-assigned tracklet identifier.
    pub trk_sub: Option<String>,
    /// MPC-assigned observation identifier (`obsid`).
    pub obs_id: Option<String>,
    /// Observer-assigned sub-identifier (`obsSubID`).
    pub obs_sub_id: Option<String>,
    /// Track identifier (`trkID`).
    pub trk_id: Option<String>,

    // ── Observer ──────────────────────────────────────────
    /// MPC observatory code.
    pub obs_code: String,
    /// Observation mode (CCD, CMOS, etc.).
    pub mode: Option<String>,
    /// MPC program code.
    pub prog: Option<String>,

    // ── Observer location (roving / spacecraft) ──────────
    /// Coordinate system for observer position (e.g. "WGS84", "ICRF_KM").
    pub sys: Option<String>,
    /// Center body NAIF ID.
    pub ctr: Option<f64>,
    /// Position component 1.
    pub pos1: Option<f64>,
    /// Position component 2.
    pub pos2: Option<f64>,
    /// Position component 3.
    pub pos3: Option<f64>,

    // ── Core astrometry ──────────────────────────────────
    /// Observation time (ISO 8601 UTC).
    pub obs_time: String,
    /// Right ascension (degrees).
    pub ra_deg: f64,
    /// Declination (degrees).
    pub dec_deg: f64,

    // ── Uncertainties ────────────────────────────────────
    /// RA·cos(Dec) uncertainty (arcseconds). NaN if unavailable.
    pub rms_ra_arcsec: f64,
    /// Dec uncertainty (arcseconds). NaN if unavailable.
    pub rms_dec_arcsec: f64,
    /// RA-Dec correlation coefficient.
    pub rms_corr: Option<f64>,

    // ── Astrometric catalog ──────────────────────────────
    /// Star catalog used for astrometric reduction.
    pub ast_cat: Option<String>,

    // ── Photometry ───────────────────────────────────────
    /// Apparent magnitude.
    pub mag: Option<f64>,
    /// Magnitude uncertainty.
    pub rms_mag: Option<f64>,
    /// Photometric passband.
    pub band: Option<String>,
    /// Photometric catalog.
    pub phot_cat: Option<String>,
    /// Photometric aperture (arcseconds).
    pub phot_ap: Option<f64>,

    // ── Supplementary diagnostics ────────────────────────
    /// log10(SNR) of the detection.
    pub log_snr: Option<f64>,
    /// Seeing FWHM (arcseconds).
    pub seeing: Option<f64>,
    /// Exposure time (seconds).
    pub exp: Option<f64>,
    /// RMS of astrometric fit (arcseconds).
    pub rms_fit: Option<f64>,
    /// Number of reference stars in astrometric fit.
    pub n_stars: Option<u32>,
    /// MPC note flags.
    pub notes: Option<String>,
    /// Free-text observer remarks.
    pub remarks: Option<String>,
}

impl Observation {
    pub(super) fn from_ffi(o: &empyrean_sys::EmpyreanObservation) -> Self {
        fn cstr_opt(p: *mut std::ffi::c_char) -> Option<String> {
            if p.is_null() {
                None
            } else {
                Some(unsafe { CStr::from_ptr(p).to_string_lossy().into_owned() })
            }
        }
        let nan_to_opt = |v: f64| if v.is_nan() { None } else { Some(v) };
        Self {
            perm_id: cstr_opt(o.perm_id),
            prov_id: cstr_opt(o.prov_id),
            trk_sub: cstr_opt(o.trk_sub),
            obs_id: cstr_opt(o.obs_id),
            obs_sub_id: cstr_opt(o.obs_sub_id),
            trk_id: cstr_opt(o.trk_id),
            obs_code: obs_code_from_bytes(&o.obs_code),
            mode: cstr_opt(o.mode),
            prog: cstr_opt(o.prog),
            sys: cstr_opt(o.sys),
            ctr: nan_to_opt(o.ctr),
            pos1: nan_to_opt(o.pos1),
            pos2: nan_to_opt(o.pos2),
            pos3: nan_to_opt(o.pos3),
            obs_time: cstr_opt(o.obs_time).unwrap_or_default(),
            ra_deg: o.ra_deg,
            dec_deg: o.dec_deg,
            rms_ra_arcsec: o.rms_ra_arcsec,
            rms_dec_arcsec: o.rms_dec_arcsec,
            rms_corr: nan_to_opt(o.rms_corr),
            ast_cat: cstr_opt(o.ast_cat),
            mag: nan_to_opt(o.mag),
            rms_mag: nan_to_opt(o.rms_mag),
            band: cstr_opt(o.band),
            phot_cat: cstr_opt(o.phot_cat),
            phot_ap: nan_to_opt(o.phot_ap),
            log_snr: nan_to_opt(o.log_snr),
            seeing: nan_to_opt(o.seeing),
            exp: nan_to_opt(o.exp),
            rms_fit: nan_to_opt(o.rms_fit),
            n_stars: if o.n_stars >= 0 {
                Some(o.n_stars as u32)
            } else {
                None
            },
            notes: cstr_opt(o.notes),
            remarks: cstr_opt(o.remarks),
        }
    }
}

/// The delay-or-Doppler measurement carried by a [`RadarObservation`]
/// (ADES `RadarValue` choice). All values are ADES-native — no unit
/// conversion is applied anywhere in the safe wrapper.
#[derive(Debug, Clone, PartialEq)]
pub enum RadarMeasurement {
    /// Round-trip time delay \\(\\tau\\): `delay_seconds` in **s**,
    /// `rms_delay_microseconds` (its \\(1\\sigma\\)) in **µs**.
    Delay {
        /// Round-trip time delay in seconds.
        delay_seconds: f64,
        /// 1σ uncertainty of the delay in microseconds.
        rms_delay_microseconds: f64,
    },
    /// Doppler shift \\(f_D\\): `doppler_hz` and `rms_doppler_hz` both in
    /// **Hz**, referred to the [`RadarObservation::frq_mhz`] carrier. The
    /// value is signed.
    Doppler {
        /// Doppler shift in Hz (signed).
        doppler_hz: f64,
        /// 1σ uncertainty of the Doppler shift in Hz.
        rms_doppler_hz: f64,
    },
}

/// A single radar observation — full ADES radar schema.
///
/// Mirrors [`Observation`] (the optical record): individual radar
/// observations are snapshots materialized from the FFI array owned by
/// [`Observations`]. All measurement values are ADES-native; the safe
/// wrapper performs no unit conversion.
#[derive(Debug, Clone, PartialEq)]
pub struct RadarObservation {
    // ── Identification ────────────────────────────────────
    /// IAU permanent designation.
    pub perm_id: Option<String>,
    /// MPC provisional designation.
    pub prov_id: Option<String>,
    /// Observer-assigned tracklet identifier.
    pub trk_sub: Option<String>,

    // ── Bistatic geometry ─────────────────────────────────
    /// MPC station code of the **transmitting** antenna (ADES `trx`).
    pub trx: String,
    /// MPC station code of the **receiving** antenna (ADES `rcv`).
    /// Equal to [`Self::trx`] for a monostatic observation.
    pub rcv: String,

    // ── Core measurement ──────────────────────────────────
    /// Observation epoch as an ISO 8601 UTC string. For radar this is the
    /// **receive** epoch (the time the returned signal is recorded).
    pub obs_time: String,
    /// The delay or Doppler measurement (ADES `RadarValue` choice).
    pub measurement: RadarMeasurement,

    // ── Reduction metadata ────────────────────────────────
    /// Transmit carrier reference frequency in MHz (ADES `frq`). Required;
    /// needed to relate a Doppler shift to a range rate.
    pub frq_mhz: f64,
    /// Center-of-mass flag (ADES `com`). `Some(true)` = reduced to the
    /// target center of mass; `Some(false)` = reduced to the peak-power
    /// (leading-edge) point. `None` when the column is absent — ADES treats
    /// a missing `com` as center-of-mass, but that default is applied
    /// explicitly downstream, not silently here.
    pub com: Option<bool>,
    /// \\(\\log_{10}(\\text{SNR})\\) of the echo, if reported.
    pub log_snr: Option<f64>,
    /// Free-text remarks from the observer.
    pub remarks: Option<String>,
}

impl RadarObservation {
    pub(super) fn from_ffi(o: &empyrean_sys::EmpyreanRadarObservation) -> Self {
        fn cstr_opt(p: *mut std::ffi::c_char) -> Option<String> {
            if p.is_null() {
                None
            } else {
                Some(unsafe { CStr::from_ptr(p).to_string_lossy().into_owned() })
            }
        }
        let nan_to_opt = |v: f64| if v.is_nan() { None } else { Some(v) };
        // `kind` is a `u8`; the bindgen constants are `u32`. Compare the
        // Doppler discriminant explicitly and treat everything else as
        // Delay. Our own C ABI only ever emits the two known kinds, so the
        // fall-through is defensive, not a silent default for live data.
        let measurement = if u32::from(o.kind) == empyrean_sys::EMPYREAN_RADAR_KIND_DOPPLER {
            RadarMeasurement::Doppler {
                doppler_hz: o.doppler_hz,
                rms_doppler_hz: o.rms_doppler_hz,
            }
        } else {
            RadarMeasurement::Delay {
                delay_seconds: o.delay_seconds,
                rms_delay_microseconds: o.rms_delay_microseconds,
            }
        };
        let com = match o.com {
            1 => Some(true),
            0 => Some(false),
            _ => None,
        };
        Self {
            perm_id: cstr_opt(o.perm_id),
            prov_id: cstr_opt(o.prov_id),
            trk_sub: cstr_opt(o.trk_sub),
            trx: obs_code_from_bytes(&o.trx),
            rcv: obs_code_from_bytes(&o.rcv),
            obs_time: cstr_opt(o.obs_time).unwrap_or_default(),
            measurement,
            frq_mhz: o.frq_mhz,
            com,
            log_snr: nan_to_opt(o.log_snr),
            remarks: cstr_opt(o.remarks),
        }
    }
}

/// Owned set of ADES observations returned by
/// [`Context::read_ades`](super::Context::read_ades).
///
/// Holds the FFI-allocated optical and radar arrays and frees both on
/// drop. Pass by reference to
/// [`Context::determine`](super::Context::determine) (optical + radar),
/// [`Context::evaluate`](super::Context::evaluate), or
/// [`Context::refine`](super::Context::refine) (the latter two are
/// optical-only).
pub struct Observations {
    ptr: *mut empyrean_sys::EmpyreanObservation,
    len: usize,
    radar_ptr: *mut empyrean_sys::EmpyreanRadarObservation,
    radar_len: usize,
}

unsafe impl Send for Observations {}
unsafe impl Sync for Observations {}

impl Observations {
    /// Construct an empty [`Observations`] set (no FFI allocation).
    pub(crate) fn default_empty() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            len: 0,
            radar_ptr: std::ptr::null_mut(),
            radar_len: 0,
        }
    }

    /// Wrap raw FFI pointers + lengths (optical + radar) as an owned
    /// [`Observations`]. The pointers must come from the C ABI's allocator
    /// so that `empyrean_observations_free` /
    /// `empyrean_radar_observations_free` can release them on drop.
    pub(crate) fn from_raw_parts(
        ptr: *mut empyrean_sys::EmpyreanObservation,
        len: usize,
        radar_ptr: *mut empyrean_sys::EmpyreanRadarObservation,
        radar_len: usize,
    ) -> Self {
        Self {
            ptr,
            len,
            radar_ptr,
            radar_len,
        }
    }

    /// Construct an [`Observations`] set from an array of in-memory
    /// [`Observation`] structs. The strings are copied into fresh
    /// FFI-owned allocations; the input slice can be dropped
    /// immediately after this returns.
    ///
    /// This is the entry point for callers that already have
    /// observations as an in-memory array and don't want to round-trip
    /// through ADES PSV. It is **optical-only**: the radar array is left
    /// empty. To carry both optical and radar observations, use
    /// [`Observations::from_arrays`].
    pub fn from_array(observations: &[Observation]) -> Result<Self> {
        Self::from_arrays(observations, &[])
    }

    /// Construct an [`Observations`] set from in-memory arrays of both
    /// [`Observation`] (optical) and [`RadarObservation`] (radar)
    /// structs. The strings are copied into fresh FFI-owned allocations;
    /// the input slices can be dropped immediately after this returns.
    ///
    /// This is the entry point for callers that already have
    /// observations as in-memory arrays and don't want to round-trip
    /// through ADES PSV — it parallels [`Context::read_ades`](super::Context::read_ades) in carrying
    /// both tables through to
    /// [`Context::determine`](super::Context::determine). Either slice
    /// may be empty (the corresponding FFI array is then null / length 0).
    pub fn from_arrays(observations: &[Observation], radar: &[RadarObservation]) -> Result<Self> {
        // ── Optical array ────────────────────────────────────────────
        let (opt_ptr, opt_len) = Self::build_optical_ffi(observations)?;
        // ── Radar array ──────────────────────────────────────────────
        let (radar_ptr, radar_len) = match Self::build_radar_ffi(radar) {
            Ok(parts) => parts,
            Err(e) => {
                // The optical array was already allocated by the C ABI;
                // free it so we don't leak when the radar pack fails.
                if !opt_ptr.is_null() {
                    unsafe { empyrean_sys::empyrean_observations_free(opt_ptr, opt_len) }
                }
                return Err(e);
            }
        };
        Ok(Self {
            ptr: opt_ptr,
            len: opt_len,
            radar_ptr,
            radar_len,
        })
    }

    /// Pack a slice of [`Observation`] into a C-ABI-owned optical array.
    /// Returns a null pointer / length 0 for an empty slice.
    fn build_optical_ffi(
        observations: &[Observation],
    ) -> Result<(*mut empyrean_sys::EmpyreanObservation, usize)> {
        if observations.is_empty() {
            return Ok((std::ptr::null_mut(), 0));
        }
        let mut keep_strings: Vec<CString> = Vec::with_capacity(observations.len() * 8);
        let mut input: Vec<empyrean_sys::EmpyreanObservation> =
            Vec::with_capacity(observations.len());
        // Encode an Option<String> into a (possibly null) c_char pointer
        // backed by `keep_strings` for lifetime extension.
        fn opt_str_ptr(
            value: &Option<String>,
            field: &str,
            keep: &mut Vec<CString>,
        ) -> Result<*mut std::ffi::c_char> {
            match value {
                Some(s) => {
                    let c = CString::new(s.as_bytes()).map_err(|_| {
                        Error::invalid_input(format!("{field} contains a NUL byte"))
                    })?;
                    let ptr = c.as_ptr() as *mut std::ffi::c_char;
                    keep.push(c);
                    Ok(ptr)
                }
                None => Ok(std::ptr::null_mut()),
            }
        }
        for obs in observations {
            let perm_ptr = opt_str_ptr(&obs.perm_id, "perm_id", &mut keep_strings)?;
            let prov_ptr = opt_str_ptr(&obs.prov_id, "prov_id", &mut keep_strings)?;
            let trk_sub_ptr = opt_str_ptr(&obs.trk_sub, "trk_sub", &mut keep_strings)?;
            let obs_id_ptr = opt_str_ptr(&obs.obs_id, "obs_id", &mut keep_strings)?;
            let obs_sub_id_ptr = opt_str_ptr(&obs.obs_sub_id, "obs_sub_id", &mut keep_strings)?;
            let trk_id_ptr = opt_str_ptr(&obs.trk_id, "trk_id", &mut keep_strings)?;
            let mode_ptr = opt_str_ptr(&obs.mode, "mode", &mut keep_strings)?;
            let prog_ptr = opt_str_ptr(&obs.prog, "prog", &mut keep_strings)?;
            let sys_ptr = opt_str_ptr(&obs.sys, "sys", &mut keep_strings)?;
            let ast_cat_ptr = opt_str_ptr(&obs.ast_cat, "ast_cat", &mut keep_strings)?;
            let band_ptr = opt_str_ptr(&obs.band, "band", &mut keep_strings)?;
            let phot_cat_ptr = opt_str_ptr(&obs.phot_cat, "phot_cat", &mut keep_strings)?;
            let notes_ptr = opt_str_ptr(&obs.notes, "notes", &mut keep_strings)?;
            let remarks_ptr = opt_str_ptr(&obs.remarks, "remarks", &mut keep_strings)?;
            let time_c = CString::new(obs.obs_time.as_bytes())
                .map_err(|_| Error::invalid_input("obs_time contains a NUL byte"))?;
            let time_ptr = time_c.as_ptr() as *mut std::ffi::c_char;
            keep_strings.push(time_c);
            let mut obs_code = [0u8; 4];
            let bytes = obs.obs_code.as_bytes();
            // Same contract as the ephemeris observer path: truncating a
            // 4-character MPC code to its 3-byte prefix silently aliases a
            // different observatory, so over-length codes are an error.
            if bytes.len() > 3 {
                return Err(Error::invalid_input(format!(
                    "observatory code \"{}\" is longer than 3 bytes; \
                     4-character MPC codes are not yet supported by the \
                     engine's observatory registry",
                    obs.obs_code
                )));
            }
            obs_code[..bytes.len()].copy_from_slice(bytes);
            input.push(empyrean_sys::EmpyreanObservation {
                perm_id: perm_ptr,
                prov_id: prov_ptr,
                trk_sub: trk_sub_ptr,
                obs_id: obs_id_ptr,
                obs_sub_id: obs_sub_id_ptr,
                trk_id: trk_id_ptr,
                obs_code,
                mode: mode_ptr,
                prog: prog_ptr,
                sys: sys_ptr,
                ctr: obs.ctr.unwrap_or(f64::NAN),
                pos1: obs.pos1.unwrap_or(f64::NAN),
                pos2: obs.pos2.unwrap_or(f64::NAN),
                pos3: obs.pos3.unwrap_or(f64::NAN),
                obs_time: time_ptr,
                ra_deg: obs.ra_deg,
                dec_deg: obs.dec_deg,
                rms_ra_arcsec: obs.rms_ra_arcsec,
                rms_dec_arcsec: obs.rms_dec_arcsec,
                rms_corr: obs.rms_corr.unwrap_or(f64::NAN),
                ast_cat: ast_cat_ptr,
                mag: obs.mag.unwrap_or(f64::NAN),
                rms_mag: obs.rms_mag.unwrap_or(f64::NAN),
                band: band_ptr,
                phot_cat: phot_cat_ptr,
                phot_ap: obs.phot_ap.unwrap_or(f64::NAN),
                log_snr: obs.log_snr.unwrap_or(f64::NAN),
                seeing: obs.seeing.unwrap_or(f64::NAN),
                exp: obs.exp.unwrap_or(f64::NAN),
                rms_fit: obs.rms_fit.unwrap_or(f64::NAN),
                n_stars: obs.n_stars.map(|v| v as i32).unwrap_or(-1),
                notes: notes_ptr,
                remarks: remarks_ptr,
            });
        }
        let mut out_ptr: *mut empyrean_sys::EmpyreanObservation = std::ptr::null_mut();
        let mut out_num: usize = 0;
        let code = unsafe {
            empyrean_sys::empyrean_observations_from_array(
                input.as_ptr(),
                input.len(),
                &mut out_ptr,
                &mut out_num,
            )
        };
        if code != 0 {
            return Err(Error::capture(code));
        }
        // `keep_strings` is dropped here — the C ABI duplicates the
        // strings into its own allocations.
        drop(keep_strings);
        Ok((out_ptr, out_num))
    }

    /// Pack a slice of [`RadarObservation`] into a C-ABI-owned radar
    /// array — the inverse of [`RadarObservation::from_ffi`]. Returns a
    /// null pointer / length 0 for an empty slice. All values stay
    /// ADES-native; no unit conversion is applied here.
    fn build_radar_ffi(
        radar: &[RadarObservation],
    ) -> Result<(*mut empyrean_sys::EmpyreanRadarObservation, usize)> {
        if radar.is_empty() {
            return Ok((std::ptr::null_mut(), 0));
        }
        let mut keep_strings: Vec<CString> = Vec::with_capacity(radar.len() * 5);
        let mut input: Vec<empyrean_sys::EmpyreanRadarObservation> =
            Vec::with_capacity(radar.len());
        // Encode an Option<String> into a (possibly null) c_char pointer
        // backed by `keep_strings` for lifetime extension.
        fn opt_str_ptr(
            value: &Option<String>,
            field: &str,
            keep: &mut Vec<CString>,
        ) -> Result<*mut std::ffi::c_char> {
            match value {
                Some(s) => {
                    let c = CString::new(s.as_bytes()).map_err(|_| {
                        Error::invalid_input(format!("{field} contains a NUL byte"))
                    })?;
                    let ptr = c.as_ptr() as *mut std::ffi::c_char;
                    keep.push(c);
                    Ok(ptr)
                }
                None => Ok(std::ptr::null_mut()),
            }
        }
        // Pack a station code String into a 4-byte null-padded array
        // (mirrors how `from_array` packs `obs_code`).
        fn station_bytes(code: &str) -> [u8; 4] {
            let mut buf = [0u8; 4];
            let bytes = code.as_bytes();
            let n = bytes.len().min(3);
            buf[..n].copy_from_slice(&bytes[..n]);
            buf
        }
        for obs in radar {
            let perm_ptr = opt_str_ptr(&obs.perm_id, "perm_id", &mut keep_strings)?;
            let prov_ptr = opt_str_ptr(&obs.prov_id, "prov_id", &mut keep_strings)?;
            let trk_sub_ptr = opt_str_ptr(&obs.trk_sub, "trk_sub", &mut keep_strings)?;
            let remarks_ptr = opt_str_ptr(&obs.remarks, "remarks", &mut keep_strings)?;
            let time_c = CString::new(obs.obs_time.as_bytes())
                .map_err(|_| Error::invalid_input("obs_time contains a NUL byte"))?;
            let time_ptr = time_c.as_ptr() as *mut std::ffi::c_char;
            keep_strings.push(time_c);
            // Select the live value pair by `kind`; NaN the inactive pair
            // so the discriminator — not a magic 0.0 — distinguishes a
            // zero-valued Doppler from an absent one.
            let (kind, delay_seconds, rms_delay_microseconds, doppler_hz, rms_doppler_hz) =
                match &obs.measurement {
                    RadarMeasurement::Delay {
                        delay_seconds,
                        rms_delay_microseconds,
                    } => (
                        empyrean_sys::EMPYREAN_RADAR_KIND_DELAY as u8,
                        *delay_seconds,
                        *rms_delay_microseconds,
                        f64::NAN,
                        f64::NAN,
                    ),
                    RadarMeasurement::Doppler {
                        doppler_hz,
                        rms_doppler_hz,
                    } => (
                        empyrean_sys::EMPYREAN_RADAR_KIND_DOPPLER as u8,
                        f64::NAN,
                        f64::NAN,
                        *doppler_hz,
                        *rms_doppler_hz,
                    ),
                };
            let com = match obs.com {
                Some(true) => 1,
                Some(false) => 0,
                None => -1,
            };
            input.push(empyrean_sys::EmpyreanRadarObservation {
                perm_id: perm_ptr,
                prov_id: prov_ptr,
                trk_sub: trk_sub_ptr,
                trx: station_bytes(&obs.trx),
                rcv: station_bytes(&obs.rcv),
                obs_time: time_ptr,
                kind,
                delay_seconds,
                rms_delay_microseconds,
                doppler_hz,
                rms_doppler_hz,
                frq_mhz: obs.frq_mhz,
                com,
                log_snr: obs.log_snr.unwrap_or(f64::NAN),
                remarks: remarks_ptr,
            });
        }
        let mut out_ptr: *mut empyrean_sys::EmpyreanRadarObservation = std::ptr::null_mut();
        let mut out_num: usize = 0;
        let code = unsafe {
            empyrean_sys::empyrean_radar_observations_from_array(
                input.as_ptr(),
                input.len(),
                &mut out_ptr,
                &mut out_num,
            )
        };
        if code != 0 {
            return Err(Error::capture(code));
        }
        // `keep_strings` is dropped here — the C ABI duplicates the
        // strings into its own allocations.
        drop(keep_strings);
        Ok((out_ptr, out_num))
    }

    /// Number of observations in the set.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Is the set empty?
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Iterate the observations as snapshots.
    pub fn iter(&self) -> impl Iterator<Item = Observation> + '_ {
        (0..self.len).map(move |i| unsafe { Observation::from_ffi(&*self.ptr.add(i)) })
    }

    /// Return a new [`Observations`] containing only the rows whose
    /// `obs_time` falls in the half-open interval `[start, end)`.
    ///
    /// Bounds are [`Epoch`](crate::Epoch) values — they carry their
    /// own [`TimeScale`](crate::TimeScale) so callers don't have to
    /// track "is this MJD UTC or TDB?" separately. Pass `None` to
    /// leave either bound unbounded.
    ///
    /// Internally each observation's `obs_time` (ISO 8601 UTC) is
    /// converted to MJD TDB and compared against the bounds in TDB.
    /// Observations whose `obs_time` fails to parse are dropped (the
    /// upstream FFI rejects them at OD time anyway).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use empyrean::{Epoch, query_observations};
    /// let obs = query_observations(&["2024 YR4"], None)?;
    /// // Strict 10-day discovery-arc cut: everything before MJD 60680 TDB.
    /// let early = obs.filter_by_epoch(None, Some(Epoch::from_mjd_tdb(60680.0)))?;
    /// # Ok::<(), empyrean::Error>(())
    /// ```
    ///
    /// The current implementation rebuilds the FFI-owned array via
    /// [`Observations::from_array`]; cost is one C-string allocation
    /// per retained observation.
    pub fn filter_by_epoch(
        &self,
        start: Option<crate::Epoch>,
        end: Option<crate::Epoch>,
    ) -> Result<Self> {
        use crate::time::{TimeScale, iso_to_mjd};
        let start_tdb = match start {
            Some(e) => e.mjd_tdb()?,
            None => f64::NEG_INFINITY,
        };
        let end_tdb = match end {
            Some(e) => e.mjd_tdb()?,
            None => f64::INFINITY,
        };
        let kept: Vec<Observation> = self
            .iter()
            .filter(|o| match iso_to_mjd(&o.obs_time, TimeScale::TDB) {
                Ok(t) => t >= start_tdb && t < end_tdb,
                Err(_) => false,
            })
            .collect();
        Self::from_array(&kept)
    }

    /// Number of radar observations in the set.
    pub fn radar_len(&self) -> usize {
        self.radar_len
    }

    /// Materialize the radar observations as snapshots.
    ///
    /// Mirrors [`Observations::iter`] for the optical side; radar
    /// observations only arrive via
    /// [`Context::read_ades`](super::Context::read_ades) and are carried
    /// through to [`Context::determine`](super::Context::determine).
    pub fn radar(&self) -> Vec<RadarObservation> {
        (0..self.radar_len)
            .map(|i| unsafe { RadarObservation::from_ffi(&*self.radar_ptr.add(i)) })
            .collect()
    }

    pub(crate) fn as_ffi_slice(&self) -> (*const empyrean_sys::EmpyreanObservation, usize) {
        (self.ptr as *const _, self.len)
    }

    pub(crate) fn as_radar_ffi_slice(
        &self,
    ) -> (*const empyrean_sys::EmpyreanRadarObservation, usize) {
        (self.radar_ptr as *const _, self.radar_len)
    }
}

impl Drop for Observations {
    fn drop(&mut self) {
        // Each array is freed exactly once with its matching allocator;
        // null guards keep `from_array` / `default_empty` (radar null) and
        // the empty-optical case safe.
        if !self.ptr.is_null() {
            unsafe { empyrean_sys::empyrean_observations_free(self.ptr, self.len) }
        }
        if !self.radar_ptr.is_null() {
            unsafe {
                empyrean_sys::empyrean_radar_observations_free(self.radar_ptr, self.radar_len)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 4-character observatory code must be a loud error at the FFI
    /// boundary: clipped to 3 bytes it would silently alias a different
    /// observatory (empyrean-agp9).
    #[test]
    fn four_char_obs_code_is_rejected() {
        let obs = Observation {
            obs_code: "W68a".to_string(),
            obs_time: "2026-01-01T00:00:00Z".to_string(),
            ..Observation::default()
        };
        let err = match Observations::from_array(&[obs]) {
            Ok(_) => panic!("4-character observatory code must not marshal"),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(msg.contains("W68a"), "error names the code: {msg}");
        assert!(
            msg.contains("longer than 3 bytes"),
            "error states the contract: {msg}"
        );
    }

    /// Round-trips radar observations through the FFI boundary the
    /// `query_radar` path relies on: build the C array (`build_radar_ffi`),
    /// snapshot it back (`from_ffi` via `radar()`), and free it on `Drop`.
    /// Every field — including the `com` tri-state's `None` and the
    /// delay-vs-Doppler measurement arm — must survive verbatim (ADES-native,
    /// no unit conversion), and the free must balance the alloc.
    #[test]
    fn radar_round_trips_through_ffi_and_frees_on_drop() {
        let input = vec![
            RadarObservation {
                perm_id: Some("99942".to_string()),
                prov_id: None,
                trk_sub: None,
                trx: "253".to_string(),
                rcv: "253".to_string(),
                obs_time: "2021-03-11T08:20:00Z".to_string(),
                measurement: RadarMeasurement::Delay {
                    delay_seconds: 120.5,
                    rms_delay_microseconds: 0.25,
                },
                frq_mhz: 8560.0,
                com: Some(true),
                log_snr: Some(2.5),
                remarks: Some("note".to_string()),
            },
            RadarObservation {
                perm_id: None,
                prov_id: Some("2004 MN4".to_string()),
                trk_sub: None,
                trx: "253".to_string(),
                rcv: "257".to_string(),
                obs_time: "2021-03-08T02:50:00Z".to_string(),
                measurement: RadarMeasurement::Doppler {
                    doppler_hz: -5000.0,
                    rms_doppler_hz: 0.2,
                },
                frq_mhz: 2380.0,
                com: None, // absent stays None, never silently false
                log_snr: None,
                remarks: None,
            },
        ];

        // No optical records; radar drives the round-trip.
        let obs = Observations::from_arrays(&[], &input).expect("from_arrays");
        assert_eq!(
            obs.radar(),
            input,
            "radar round-trip through the FFI must preserve every field"
        );
        // Dropping `obs` frees the FFI radar array via the C ABI — a leak or
        // double-free in the wrapper Drop / C free would surface here under
        // miri/ASAN.
        drop(obs);
    }
}
