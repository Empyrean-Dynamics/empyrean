"""One-posterior OD contract at the distribution surface.

scott v1.19.0 ("Publish one posterior: read the flat covariance from the
joint") makes the published flat ``DetermineResult.covariance`` a read of
the delivered packed joint's state block — one object, two reads — rather
than a separate matrix from the retired inversion. This test proves, at the
Python surface, that a real short-arc fit:

* delivers a covariance-bearing orbit through ``determine``,
* publishes a flat ``6x6`` state covariance that equals the orbit joint's
  own state block bit-for-bit (the joint is the authority; the flat is a
  view of it), and
* re-feeds that fitted orbit through ``propagate`` without the covariance
  state-block refusal — its joint transports as a LINEAR covariance.

The fixture ``2014_AA.psv`` is a byte-identical copy of the validation
catalog's ``2014 AA`` optical arc, a seven-observation short arc whose
ill-conditioned posterior is exactly the case where two inversions of one
posterior used to disagree beyond round-off. The arc lives beside the
other PSV fixtures, as ``test_od_output_wiring.py`` reads its own.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from empyrean import CovarianceKind, Epochs, determine, propagate, read_ades

DATA_DIR = Path(__file__).parent / "fixtures"
FIXTURE_2014AA = DATA_DIR / "2014_AA.psv"


@pytest.fixture(scope="module")
def aa_observations():
    if not FIXTURE_2014AA.exists():
        pytest.skip(f"missing fixture: {FIXTURE_2014AA}")
    optical, _radar = read_ades(FIXTURE_2014AA)
    return optical


def test_published_flat_is_the_joint_state_block_and_refeeds(aa_observations):
    """The published flat covariance IS the delivered joint's state block,
    and the covariance-bearing fit re-propagates without refusal."""
    fit = determine(aa_observations).single()
    assert fit.converged, f"2014 AA short-arc fit did not converge: {fit.termination!r}"

    # The published flat 6x6 state covariance is real and finite.
    flat = np.asarray(fit.covariance, dtype=float)
    assert flat.shape == (6, 6)
    assert np.isfinite(flat).all(), "published flat covariance carries a non-finite element"

    # One posterior: the published flat IS the delivered joint's state block.
    # Cartesian output (the ODConfig default), so no angular congruence — the
    # two reads of the one object must agree bit-for-bit.
    joint = fit.orbit.covariance.joint(0)
    assert joint is not None, "fitted orbit carries no packed joint"
    state_block = np.asarray(joint.state_block(), dtype=float)
    assert np.array_equal(flat, state_block), (
        "published DetermineResult.covariance must equal the orbit joint's "
        "state block bit-for-bit (scott v1.19.0 one-posterior contract)"
    )

    # The covariance-bearing fitted orbit re-feeds through propagate without
    # the state-block refusal a mismatched posterior would raise: its joint
    # transports as a LINEAR covariance at every delivered epoch.
    epoch0 = fit.orbit.coordinates.epoch.to_pylist()[0]
    epochs = Epochs.from_mjd([epoch0, epoch0 + 1.0], scale="tdb")
    result = propagate(fit.orbit, epochs, tagged_covariance=True)
    assert len(result.states) == 2, "the fitted orbit did not deliver both propagated epochs"

    series = result.tagged_covariance_series(0)
    assert len(series) == 2
    for tagged in series:
        matrix = np.asarray(tagged.matrix, dtype=float)
        assert matrix.shape == (6, 6)
        assert np.isfinite(matrix).all(), "transported covariance carries a non-finite element"
        assert tagged.kind is CovarianceKind.LINEAR
