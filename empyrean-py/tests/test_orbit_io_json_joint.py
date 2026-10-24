"""The JSON orbit format carries the full packed joint (0.11.0).

Parquet and JSON both round-trip the orbit's whole ``(6+P)x(6+P)`` covariance
as one packed joint; CSV carries the state ``6x6`` only (the joint's leading
block) and refuses a joint carrying solved-parameter axes by name
(``ParameterCovarianceNotWritable``). A JSON file written before this change —
carrying the removed flat ``6x6`` ``covariance`` field — is refused by name on
read, with no migration shim.
"""

import math
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from empyrean import CartesianCoordinates, CartesianOrbits, NonGravParams, Origin
from empyrean.coordinates.epoch import Epochs
from empyrean.io.orbits import (
    read_orbits_json,
    read_orbits_parquet,
    write_orbits_csv,
    write_orbits_json,
)
from empyrean.orbits.joint import (
    CovarianceKind,
    JointFunctional,
    JointProvenance,
    JointQuality,
    PackedJoint,
    PackedJoints,
)

LEGACY_FIXTURE = (
    Path(__file__).resolve().parents[1].parent
    / "empyrean-c"
    / "tests"
    / "fixtures"
    / "orbit_legacy_flat_covariance.json"
)


def _width9_joint() -> PackedJoint:
    """A width-9 Cartesian joint: the state 6x6 plus a diagonal Marsden
    (A1, A2, A3) prior on bits 0/1/2, with a distinctive strictly increasing
    triangle so a transposed or truncated round trip is visible."""
    tri = np.asarray([1.0 + i * 0.25 for i in range(9 * 10 // 2)], dtype=np.float64)
    return PackedJoint(
        layout_present=0b111,
        layout_considered=0,
        layout_marginalized=0,
        width=9,
        tri=tri,
        kind=CovarianceKind.LINEAR,
        quality=JointQuality.POSITIVE_DEFINITE,
        cov_rank=0,
        cov_null_axes=0,
        cov_min_eig_corr=float("nan"),
        functional=JointFunctional.STATE,
        provenance=JointProvenance.DERIVED,
        asserted_source=None,
    )


def _wide_orbit() -> CartesianOrbits:
    """One Cartesian orbit carrying the width-9 Marsden joint, with the
    matching (zero-coefficient) Marsden model attached so the engine's
    CSV/parquet path accepts it."""
    coords = CartesianCoordinates.from_kwargs(
        epochs=Epochs.from_kwargs(mjd=[60320.0], scale="tdb"),
        x=[-7.85e-02],
        y=[-8.20e-01],
        z=[4.19e-02],
        vx=[1.99e-02],
        vy=[1.32e-03],
        vz=[3.99e-04],
        frame="ecliptic_j2000",
        origin=[str(Origin.SUN)],
    )
    non_grav = NonGravParams.from_kwargs(a1=[0.0], a2=[0.0], a3=[0.0], model=["inverse_square"])
    return CartesianOrbits.from_kwargs(
        orbit_id=["WIDE_JOINT"],
        object_id=["wide_joint_obj"],
        coordinates=coords,
        non_grav=non_grav,
        covariance=PackedJoints.from_joints([_width9_joint()]),
    )


def _assert_joints_equal(got: PackedJoint, want: PackedJoint) -> None:
    assert got.width == want.width
    assert got.layout_present == want.layout_present
    assert got.layout_considered == want.layout_considered
    assert got.layout_marginalized == want.layout_marginalized
    assert got.kind == want.kind
    assert got.quality == want.quality
    assert got.functional == want.functional
    assert got.provenance == want.provenance
    assert got.asserted_source == want.asserted_source
    # cov_min_eig_corr is the NaN not-available sentinel for a PD joint; it
    # round-trips as NaN, not 0 (NaN != NaN, so test it explicitly).
    assert math.isnan(got.cov_min_eig_corr) and math.isnan(want.cov_min_eig_corr)
    np.testing.assert_array_equal(
        np.asarray(got.tri, dtype=np.float64),
        np.asarray(want.tri, dtype=np.float64),
    )


def test_json_round_trips_a_width9_joint(tmp_path: Path) -> None:
    orbits = _wide_orbit()
    path = str(tmp_path / "wide.json")
    write_orbits_json(path, orbits)
    back = read_orbits_json(path)

    assert len(back) == 1
    got = back.covariance.joint(0)
    assert got is not None, "the JSON round trip must restore the packed joint"
    _assert_joints_equal(got, _width9_joint())


def test_json_round_trip_matches_parquet(tmp_path: Path) -> None:
    """JSON and parquet carry the SAME joint for the same orbit — Cartesian,
    so bit-for-bit (no angular scaling at any boundary)."""
    orbits = _wide_orbit()
    jpath = str(tmp_path / "wide.json")
    ppath = str(tmp_path / "wide.parquet")
    write_orbits_json(jpath, orbits)
    from empyrean.io.orbits import write_orbits_parquet

    write_orbits_parquet(ppath, orbits)
    jj = read_orbits_json(jpath).covariance.joint(0)
    pj = read_orbits_parquet(ppath).covariance.joint(0)
    assert jj is not None and pj is not None
    assert jj.width == pj.width
    np.testing.assert_array_equal(
        np.asarray(jj.tri, dtype=np.float64),
        np.asarray(pj.tri, dtype=np.float64),
    )


def test_csv_refuses_a_width9_joint_by_name(tmp_path: Path) -> None:
    orbits = _wide_orbit()
    path = str(tmp_path / "wide.csv")
    with pytest.raises(Exception) as excinfo:
        write_orbits_csv(path, orbits)
    assert "ParameterCovarianceNotWritable" in str(excinfo.value), (
        "CSV must refuse a joint carrying solved-parameter axes by name"
    )


def test_read_refuses_the_legacy_flat_covariance_field(tmp_path: Path) -> None:
    assert LEGACY_FIXTURE.exists(), f"missing fixture {LEGACY_FIXTURE}"
    with pytest.raises(Exception) as excinfo:
        read_orbits_json(str(LEGACY_FIXTURE))
    msg = str(excinfo.value)
    assert "flat" in msg and "covariance" in msg, (
        f"the refusal must name the removed flat covariance shape: {msg}"
    )


def _resolve_cli() -> Path | None:
    """The `empyrean` CLI binary: `EMPYREAN_CLI` if set, else the worktree's
    built binary. `None` => the determine-chain test skips."""
    env = os.environ.get("EMPYREAN_CLI")
    if env and Path(env).exists():
        return Path(env)
    root = Path(__file__).resolve().parents[1].parent
    for rel in ("target/release/empyrean", "target/debug/empyrean"):
        cand = root / rel
        if cand.exists():
            return cand
    return None


def test_cli_determine_json_joint_read_back_with_python_equals_parquet(
    tmp_path: Path,
) -> None:
    """End-to-end: the CLI `determine` writes `--format json`, and Python
    reads it back with a joint equal to the parquet the same run wrote.

    Shells the CLI, so it needs the built binary (`EMPYREAN_CLI` or the
    worktree target) and a populated data directory; skips otherwise.
    """
    cli = _resolve_cli()
    if cli is None:
        pytest.skip("no empyrean CLI binary (set EMPYREAN_CLI)")
    data_dir = os.environ.get("EMPYREAN_DATA_DIR")
    if not data_dir or not (Path(data_dir) / "de440.bsp").exists():
        pytest.skip("no data directory with de440.bsp")
    fixture = (
        Path(__file__).resolve().parents[1].parent
        / "empyrean-c"
        / "tests"
        / "fixtures"
        / "433_eros.psv"
    )
    env = dict(os.environ)
    env["DYLD_LIBRARY_PATH"] = str(cli.parent)
    env["LD_LIBRARY_PATH"] = str(cli.parent)
    env["EMPYREAN_OFFLINE"] = "1"

    jdir = tmp_path / "j"
    pdir = tmp_path / "p"
    for out_dir, fmt in ((jdir, "json"), (pdir, "parquet")):
        proc = subprocess.run(
            [
                str(cli),
                "--data-dir",
                data_dir,
                "determine",
                str(fixture),
                "--out-dir",
                str(out_dir),
                "--format",
                fmt,
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"determine --format {fmt} failed:\n{proc.stderr}"

    jj = read_orbits_json(str(jdir / "fitted_orbits.json")).covariance.joint(0)
    pj = read_orbits_parquet(str(pdir / "fitted_orbits.parquet")).covariance.joint(0)
    assert jj is not None, "the fitted JSON must carry its packed joint"
    assert pj is not None
    assert jj.width == pj.width
    np.testing.assert_allclose(
        np.asarray(jj.tri, dtype=np.float64),
        np.asarray(pj.tri, dtype=np.float64),
        rtol=1e-9,
        atol=1e-18,
    )
