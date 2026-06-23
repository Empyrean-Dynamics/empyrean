"""Advanced integrator config (integrator backend, origin switching) and
math primitives (eigenvector, Gaussian split)."""

import numpy as np
import pytest
from empyrean import (
    AdvancedIntegratorConfig,
    IntegratorChoice,
    MixtureComponent,
    OriginSwitchingConfig,
    PropagationConfig,
    eigenvector_max_6x6,
    split_gaussian,
)

# ── AdvancedIntegratorConfig defaults ──────────────────────────


def test_advanced_config_defaults_match_engine():
    adv = AdvancedIntegratorConfig()
    assert adv.integrator == IntegratorChoice.GR15
    assert adv.origin_switching.enabled is True
    assert adv.origin_switching.hysteresis == pytest.approx(0.2)


def test_advanced_config_wire_dict_includes_new_fields():
    cfg = PropagationConfig(
        advanced=AdvancedIntegratorConfig(
            integrator=IntegratorChoice.DOP853,
            origin_switching=OriginSwitchingConfig(enabled=True, hysteresis=0.15),
        ),
    )
    wire = cfg._to_wire_dict()
    assert wire["advanced"]["integrator"] == "dop853"
    assert wire["advanced"]["origin_switching"] == {
        "enabled": True,
        "hysteresis": 0.15,
    }


# ── Math primitives ────────────────────────────────────────────


def test_eigenvector_max_6x6_diagonal():
    # Diagonal covariance — dominant axis is the largest variance.
    diag = np.diag([1.0, 4.0, 2.0, 0.5, 9.0, 3.0])
    eigenvalue, eigenvector = eigenvector_max_6x6(diag)
    assert eigenvalue == pytest.approx(9.0)
    expected = np.zeros(6)
    expected[4] = 1.0
    assert np.allclose(np.abs(eigenvector), expected)


def test_split_gaussian_preserves_total_weight_and_mean():
    mean = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    cov = np.diag([4.0, 1.0, 1.0, 0.01, 0.01, 0.01])
    components = split_gaussian(mean, cov, k=3)

    assert len(components) == 3
    assert all(isinstance(c, MixtureComponent) for c in components)

    # Weights sum to 1.
    total_weight = sum(c.weight for c in components)
    assert total_weight == pytest.approx(1.0)

    # Mixture mean equals the original mean (centered split).
    mixture_mean = sum(c.weight * c.mean for c in components)
    assert np.allclose(mixture_mean, mean)


def test_eigenvector_max_6x6_rejects_wrong_shape():
    with pytest.raises(ValueError, match="6x6"):
        eigenvector_max_6x6(np.eye(5))


def test_split_gaussian_rejects_wrong_k():
    with pytest.raises(ValueError, match="k must be"):
        split_gaussian(np.zeros(6), np.eye(6), k=0)
