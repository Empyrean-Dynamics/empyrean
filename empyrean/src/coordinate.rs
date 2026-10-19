//! Coordinate systems, frames, and state representations.
//!
//! All angular values cross the FFI boundary in **degrees**. Distances are
//! in AU, velocities in AU/day.

/// Orbital state representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Representation {
    /// Cartesian: x, y, z, vx, vy, vz (AU, AU/day)
    Cartesian = 0,
    /// Keplerian: a, e, i, raan, ap, ma (AU, -, deg, deg, deg, deg)
    Keplerian = 1,
    /// Cometary: q, e, i, raan, ap, tp (AU, -, deg, deg, deg, MJD)
    Cometary = 2,
    /// Spherical: r, lon, lat, vr, vlon, vlat (AU, deg, deg, AU/day, deg/day, deg/day)
    Spherical = 3,
}

/// Reference frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum Frame {
    /// International Celestial Reference Frame (inertial, J2000).
    ICRF = 0,
    /// Ecliptic J2000 (inertial).
    EclipticJ2000 = 1,
    /// Earth body-fixed rotating frame (ITRF93). Requires a BPC kernel
    /// loaded into the [`Context`](crate::context::Context).
    ITRF93 = 2,
}

impl std::fmt::Display for Frame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Frame::ICRF => write!(f, "icrf"),
            Frame::EclipticJ2000 => write!(f, "eclipticj2000"),
            Frame::ITRF93 => write!(f, "itrf93"),
        }
    }
}

impl std::str::FromStr for Frame {
    type Err = crate::error::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "icrf" => Ok(Frame::ICRF),
            "eclipticj2000" | "ecliptic_j2000" | "ecliptic j2000" => Ok(Frame::EclipticJ2000),
            "itrf93" | "itrf_93" => Ok(Frame::ITRF93),
            _ => Err(crate::error::Error::invalid_input(format!(
                "unknown frame: {s:?}"
            ))),
        }
    }
}

/// Origin (center body) for a coordinate state.
///
/// Use the named variants for solar-system bodies. For numbered
/// asteroids, use [`Origin::asteroid`].
///
/// DE440 ships planet body-center segments only for Mercury, Venus,
/// Earth, and the Moon. Mars through Pluto are exposed as
/// system-barycenter variants because that's what the kernel provides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Origin {
    /// Solar System Barycenter.
    SolarSystemBarycenter,
    /// Sun center.
    Sun,
    /// Mercury center.
    Mercury,
    /// Venus center.
    Venus,
    /// Earth center.
    Earth,
    /// Moon center.
    Moon,
    /// Mars system barycenter.
    MarsBarycenter,
    /// Jupiter system barycenter.
    JupiterBarycenter,
    /// Saturn system barycenter.
    SaturnBarycenter,
    /// Uranus system barycenter.
    UranusBarycenter,
    /// Neptune system barycenter.
    NeptuneBarycenter,
    /// Pluto system barycenter.
    PlutoBarycenter,
    /// Numbered asteroid (IAU number, e.g. `1` for Ceres, `4` for Vesta).
    Asteroid(i32),
}

impl Origin {
    /// Solar System Barycenter — alias for [`Origin::SolarSystemBarycenter`].
    pub const SSB: Origin = Origin::SolarSystemBarycenter;
    /// Sun — alias for [`Origin::Sun`].
    pub const SUN: Origin = Origin::Sun;
    /// Mercury — alias for [`Origin::Mercury`].
    pub const MERCURY: Origin = Origin::Mercury;
    /// Venus — alias for [`Origin::Venus`].
    pub const VENUS: Origin = Origin::Venus;
    /// Earth — alias for [`Origin::Earth`].
    pub const EARTH: Origin = Origin::Earth;
    /// Moon — alias for [`Origin::Moon`].
    pub const MOON: Origin = Origin::Moon;
    /// Mars system barycenter — alias for [`Origin::MarsBarycenter`].
    pub const MARS_BARYCENTER: Origin = Origin::MarsBarycenter;
    /// Jupiter system barycenter — alias for [`Origin::JupiterBarycenter`].
    pub const JUPITER_BARYCENTER: Origin = Origin::JupiterBarycenter;
    /// Saturn system barycenter — alias for [`Origin::SaturnBarycenter`].
    pub const SATURN_BARYCENTER: Origin = Origin::SaturnBarycenter;
    /// Uranus system barycenter — alias for [`Origin::UranusBarycenter`].
    pub const URANUS_BARYCENTER: Origin = Origin::UranusBarycenter;
    /// Neptune system barycenter — alias for [`Origin::NeptuneBarycenter`].
    pub const NEPTUNE_BARYCENTER: Origin = Origin::NeptuneBarycenter;
    /// Pluto system barycenter — alias for [`Origin::PlutoBarycenter`].
    pub const PLUTO_BARYCENTER: Origin = Origin::PlutoBarycenter;

    /// Construct a numbered-asteroid origin from an IAU number.
    pub fn asteroid(number: i32) -> Self {
        Origin::Asteroid(number)
    }

    /// NAIF integer body code that the C ABI uses to identify this origin.
    ///
    /// Crate-internal — surfaced for tests and the FFI bridge. The user-facing
    /// API never asks the caller to type a NAIF integer; if you need to
    /// interoperate with a system that does, this is the escape hatch.
    pub fn naif_id(self) -> i32 {
        match self {
            Origin::SolarSystemBarycenter => 0,
            Origin::Sun => 10,
            Origin::Mercury => 199,
            Origin::Venus => 299,
            Origin::Earth => 399,
            Origin::Moon => 301,
            Origin::MarsBarycenter => 4,
            Origin::JupiterBarycenter => 5,
            Origin::SaturnBarycenter => 6,
            Origin::UranusBarycenter => 7,
            Origin::NeptuneBarycenter => 8,
            Origin::PlutoBarycenter => 9,
            Origin::Asteroid(n) => 2_000_000 + n,
        }
    }

    /// Construct an [`Origin`] from its NAIF integer body code.
    ///
    /// Returns `None` for codes that don't correspond to any supported
    /// origin variant. Used by the FFI bridge to re-hydrate typed
    /// origins from C-ABI events / states.
    pub fn from_naif_id(id: i32) -> Option<Self> {
        match id {
            0 => Some(Origin::SolarSystemBarycenter),
            10 => Some(Origin::Sun),
            199 => Some(Origin::Mercury),
            299 => Some(Origin::Venus),
            399 => Some(Origin::Earth),
            301 => Some(Origin::Moon),
            4 => Some(Origin::MarsBarycenter),
            5 => Some(Origin::JupiterBarycenter),
            6 => Some(Origin::SaturnBarycenter),
            7 => Some(Origin::UranusBarycenter),
            8 => Some(Origin::NeptuneBarycenter),
            9 => Some(Origin::PlutoBarycenter),
            id if (2_000_001..3_000_000).contains(&id) => Some(Origin::Asteroid(id - 2_000_000)),
            _ => None,
        }
    }
}

impl std::fmt::Display for Origin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Origin::SolarSystemBarycenter => write!(f, "SSB"),
            Origin::Sun => write!(f, "Sun"),
            Origin::Mercury => write!(f, "Mercury"),
            Origin::Venus => write!(f, "Venus"),
            Origin::Earth => write!(f, "Earth"),
            Origin::Moon => write!(f, "Moon"),
            Origin::MarsBarycenter => write!(f, "Mars Barycenter"),
            Origin::JupiterBarycenter => write!(f, "Jupiter Barycenter"),
            Origin::SaturnBarycenter => write!(f, "Saturn Barycenter"),
            Origin::UranusBarycenter => write!(f, "Uranus Barycenter"),
            Origin::NeptuneBarycenter => write!(f, "Neptune Barycenter"),
            Origin::PlutoBarycenter => write!(f, "Pluto Barycenter"),
            Origin::Asteroid(n) => write!(f, "asteroid_{n}"),
        }
    }
}

impl std::str::FromStr for Origin {
    type Err = crate::error::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.trim().to_lowercase();
        if let Some(rest) = lower.strip_prefix("asteroid_") {
            let number: i32 = rest.parse().map_err(|_| {
                crate::error::Error::invalid_input(format!(
                    "asteroid number must be a positive integer: {s:?}"
                ))
            })?;
            if number <= 0 {
                return Err(crate::error::Error::invalid_input(format!(
                    "asteroid number must be positive: {s:?}"
                )));
            }
            return Ok(Origin::Asteroid(number));
        }
        match lower.as_str() {
            "ssb" | "solar system barycenter" | "solar_system_barycenter" => {
                Ok(Origin::SolarSystemBarycenter)
            }
            "sun" => Ok(Origin::Sun),
            "mercury" => Ok(Origin::Mercury),
            "venus" => Ok(Origin::Venus),
            "earth" => Ok(Origin::Earth),
            "moon" => Ok(Origin::Moon),
            "mars" | "mars barycenter" | "mars_barycenter" => Ok(Origin::MarsBarycenter),
            "jupiter" | "jupiter barycenter" | "jupiter_barycenter" => {
                Ok(Origin::JupiterBarycenter)
            }
            "saturn" | "saturn barycenter" | "saturn_barycenter" => Ok(Origin::SaturnBarycenter),
            "uranus" | "uranus barycenter" | "uranus_barycenter" => Ok(Origin::UranusBarycenter),
            "neptune" | "neptune barycenter" | "neptune_barycenter" => {
                Ok(Origin::NeptuneBarycenter)
            }
            "pluto" | "pluto barycenter" | "pluto_barycenter" => Ok(Origin::PlutoBarycenter),
            _ => Err(crate::error::Error::invalid_input(format!(
                "unknown origin: {s:?}"
            ))),
        }
    }
}

/// Convert an integer code to a [`Frame`].
///
/// Codes match the C ABI: 0 = ICRF, 1 = EclipticJ2000, 2 = ITRF93.
pub fn int_to_frame(val: i32) -> Result<Frame, crate::error::Error> {
    match val {
        0 => Ok(Frame::ICRF),
        1 => Ok(Frame::EclipticJ2000),
        2 => Ok(Frame::ITRF93),
        _ => Err(crate::error::Error::invalid_input(format!(
            "unknown frame code: {val}"
        ))),
    }
}

/// Convert a [`Frame`] to its integer code (matches the C ABI).
pub fn frame_to_int(frame: Frame) -> i32 {
    frame as i32
}

/// Convert an integer code to a [`Representation`].
///
/// Codes match the C ABI: 0 = Cartesian, 1 = Keplerian, 2 = Cometary, 3 = Spherical.
pub fn int_to_rep(val: i32) -> Result<Representation, crate::error::Error> {
    match val {
        0 => Ok(Representation::Cartesian),
        1 => Ok(Representation::Keplerian),
        2 => Ok(Representation::Cometary),
        3 => Ok(Representation::Spherical),
        _ => Err(crate::error::Error::invalid_input(format!(
            "unknown representation code: {val}"
        ))),
    }
}

/// Convert a [`Representation`] to its integer code (matches the C ABI).
pub fn rep_to_int(rep: Representation) -> i32 {
    rep as i32
}

/// Coordinate state: epoch + six-element state vector + optional covariance.
///
/// Element ordering depends on [`Representation`]:
///
/// | Representation | `elements[0..6]` | Angular fields (degrees) |
/// |---|---|---|
/// | Cartesian | x, y, z, vx, vy, vz | none |
/// | Keplerian | a, e, i, raan, ap, ma | i, raan, ap, ma |
/// | Cometary | q, e, i, raan, ap, tp | i, raan, ap |
/// | Spherical | r, lon, lat, vr, vlon, vlat | lon, lat, vlon, vlat |
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoordinateState {
    /// Epoch.
    pub epoch: crate::Epoch,
    /// Six-element state vector in the chosen representation.
    pub elements: [f64; 6],
    /// Optional 6x6 covariance matrix in native representation units
    /// (degrees for angular fields).
    pub covariance: Option<[[f64; 6]; 6]>,
    /// State representation.
    pub representation: Representation,
    /// Reference frame.
    pub frame: Frame,
    /// NAIF body ID of the coordinate origin.
    pub origin: Origin,
}

impl CoordinateState {
    /// Construct a Cartesian state.
    pub fn cartesian(
        epoch: crate::Epoch,
        elements: [f64; 6],
        frame: Frame,
        origin: Origin,
    ) -> Self {
        Self {
            epoch,
            elements,
            covariance: None,
            representation: Representation::Cartesian,
            frame,
            origin,
        }
    }

    /// Construct a Keplerian state. Angular elements must be in degrees.
    pub fn keplerian(
        epoch: crate::Epoch,
        elements: [f64; 6],
        frame: Frame,
        origin: Origin,
    ) -> Self {
        Self {
            epoch,
            elements,
            covariance: None,
            representation: Representation::Keplerian,
            frame,
            origin,
        }
    }

    /// Construct a Cometary state. Angular elements must be in degrees.
    pub fn cometary(epoch: crate::Epoch, elements: [f64; 6], frame: Frame, origin: Origin) -> Self {
        Self {
            epoch,
            elements,
            covariance: None,
            representation: Representation::Cometary,
            frame,
            origin,
        }
    }

    /// Construct a Spherical state. Angular elements must be in degrees.
    pub fn spherical(
        epoch: crate::Epoch,
        elements: [f64; 6],
        frame: Frame,
        origin: Origin,
    ) -> Self {
        Self {
            epoch,
            elements,
            covariance: None,
            representation: Representation::Spherical,
            frame,
            origin,
        }
    }

    /// Attach a covariance matrix.
    pub fn with_covariance(mut self, covariance: [[f64; 6]; 6]) -> Self {
        self.covariance = Some(covariance);
        self
    }

    /// Convert to the FFI layout.
    pub(crate) fn to_ffi(self) -> crate::error::Result<empyrean_sys::CoordinateState> {
        Ok(empyrean_sys::CoordinateState {
            epoch_mjd_tdb: self.epoch.mjd_tdb()?,
            elements: self.elements,
            covariance: self.covariance.unwrap_or([[0.0; 6]; 6]),
            has_covariance: self.covariance.is_some() as u8,
            representation: self.representation as i32,
            frame: self.frame as i32,
            origin: self.origin.naif_id(),
        })
    }

    /// Convert from the FFI layout.
    ///
    /// Returns an error if the FFI struct carries a NAIF code that
    /// doesn't correspond to a supported [`Origin`] variant — the C ABI
    /// shouldn't emit those, but unrecognised codes are surfaced rather
    /// than silently coerced.
    pub(crate) fn from_ffi(state: &empyrean_sys::CoordinateState) -> crate::error::Result<Self> {
        let representation = int_to_rep(state.representation)?;
        let frame = int_to_frame(state.frame)?;
        let origin = Origin::from_naif_id(state.origin).ok_or_else(|| {
            crate::error::Error::invalid_input(format!(
                "C ABI returned unknown NAIF id for origin: {}",
                state.origin
            ))
        })?;
        Ok(Self {
            epoch: crate::Epoch::from_mjd_tdb(state.epoch_mjd_tdb),
            elements: state.elements,
            covariance: (state.has_covariance != 0).then_some(state.covariance),
            representation,
            frame,
            origin,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn naif_id_round_trip_named_bodies() {
        let bodies = [
            Origin::SolarSystemBarycenter,
            Origin::Sun,
            Origin::Mercury,
            Origin::Venus,
            Origin::Earth,
            Origin::Moon,
            Origin::MarsBarycenter,
            Origin::JupiterBarycenter,
            Origin::SaturnBarycenter,
            Origin::UranusBarycenter,
            Origin::NeptuneBarycenter,
            Origin::PlutoBarycenter,
        ];
        for o in bodies {
            assert_eq!(Origin::from_naif_id(o.naif_id()), Some(o));
        }
    }

    #[test]
    fn naif_id_uses_de440_segments_for_outer_planets() {
        // Mars-Pluto are exposed as system barycenters because that's
        // what DE440 ships. Earlier versions of the wrapper had broken
        // body-center constants (499, 599, 699, 799, 899) that
        // villeneuve rejected — guard against the regression.
        assert_eq!(Origin::MarsBarycenter.naif_id(), 4);
        assert_eq!(Origin::JupiterBarycenter.naif_id(), 5);
        assert_eq!(Origin::SaturnBarycenter.naif_id(), 6);
        assert_eq!(Origin::UranusBarycenter.naif_id(), 7);
        assert_eq!(Origin::NeptuneBarycenter.naif_id(), 8);
        assert_eq!(Origin::PlutoBarycenter.naif_id(), 9);
    }

    #[test]
    fn asteroid_round_trips_via_sb441_offset() {
        let ceres = Origin::asteroid(1);
        assert_eq!(ceres.naif_id(), 2_000_001);
        assert_eq!(Origin::from_naif_id(2_000_001), Some(Origin::Asteroid(1)));

        let vesta = Origin::asteroid(4);
        assert_eq!(vesta.naif_id(), 2_000_004);
        assert_eq!(Origin::from_naif_id(2_000_004), Some(Origin::Asteroid(4)));
    }

    #[test]
    fn const_aliases_match_variants() {
        assert_eq!(Origin::SSB, Origin::SolarSystemBarycenter);
        assert_eq!(Origin::SUN, Origin::Sun);
        assert_eq!(Origin::EARTH, Origin::Earth);
        assert_eq!(Origin::MOON, Origin::Moon);
        assert_eq!(Origin::MARS_BARYCENTER, Origin::MarsBarycenter);
    }

    #[test]
    fn display_uses_canonical_strings() {
        assert_eq!(Origin::Earth.to_string(), "Earth");
        assert_eq!(Origin::SolarSystemBarycenter.to_string(), "SSB");
        assert_eq!(Origin::JupiterBarycenter.to_string(), "Jupiter Barycenter");
        assert_eq!(Origin::Asteroid(99942).to_string(), "asteroid_99942");
    }

    #[test]
    fn from_str_parses_canonical_and_lowercase() {
        assert_eq!(Origin::from_str("Earth").unwrap(), Origin::Earth);
        assert_eq!(Origin::from_str("earth").unwrap(), Origin::Earth);
        assert_eq!(
            Origin::from_str("SSB").unwrap(),
            Origin::SolarSystemBarycenter
        );
        assert_eq!(
            Origin::from_str("Mars Barycenter").unwrap(),
            Origin::MarsBarycenter
        );
        assert_eq!(Origin::from_str("mars").unwrap(), Origin::MarsBarycenter);
        assert_eq!(Origin::from_str("asteroid_1").unwrap(), Origin::Asteroid(1));
    }

    #[test]
    fn from_str_rejects_unknown_and_zero_asteroid() {
        assert!(Origin::from_str("Pluto Center").is_err());
        assert!(Origin::from_str("asteroid_0").is_err());
        assert!(Origin::from_str("asteroid_-1").is_err());
        assert!(Origin::from_str("asteroid_abc").is_err());
    }

    #[test]
    fn frame_round_trips_via_int() {
        for f in [Frame::ICRF, Frame::EclipticJ2000, Frame::ITRF93] {
            assert_eq!(int_to_frame(f as i32).unwrap(), f);
        }
    }

    #[test]
    fn frame_from_str_accepts_canonical_and_variants() {
        assert_eq!(Frame::from_str("icrf").unwrap(), Frame::ICRF);
        assert_eq!(Frame::from_str("ICRF").unwrap(), Frame::ICRF);
        assert_eq!(
            Frame::from_str("eclipticj2000").unwrap(),
            Frame::EclipticJ2000
        );
        assert_eq!(
            Frame::from_str("ecliptic_j2000").unwrap(),
            Frame::EclipticJ2000
        );
        assert_eq!(Frame::from_str("itrf93").unwrap(), Frame::ITRF93);
        assert!(Frame::from_str("galactic").is_err());
    }

    #[test]
    fn frame_display_uses_canonical_lowercase() {
        assert_eq!(Frame::ICRF.to_string(), "icrf");
        assert_eq!(Frame::EclipticJ2000.to_string(), "eclipticj2000");
        assert_eq!(Frame::ITRF93.to_string(), "itrf93");
    }

    #[test]
    fn from_naif_id_rejects_planet_body_centers() {
        // 499/599/699/... are body-center NAIF IDs that DE440 doesn't
        // ship segments for. These should NOT round-trip — anyone
        // synthesizing them by hand should hit a clear failure.
        assert_eq!(Origin::from_naif_id(499), None);
        assert_eq!(Origin::from_naif_id(599), None);
        assert_eq!(Origin::from_naif_id(699), None);
        assert_eq!(Origin::from_naif_id(799), None);
        assert_eq!(Origin::from_naif_id(899), None);
    }
}
