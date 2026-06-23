# Working with sensitivities

The {class}`~empyrean.StateSensitivities` and
{class}`~empyrean.ObservationSensitivities` tables hold per-epoch state
transition matrices (and their second-order tensors) for use in
covariance propagation, nonlinearity diagnostics, and Bayesian updates.

## Filter to one chain

Sensitivity tables hold every `(orbit, epoch)` row across all
propagated orbits — group with the standard quivr `select` pattern
before pulling matrices:

```python
result = empyrean.propagate(orbits, epochs)
chain  = result.sensitivity.select("orbit_id", "99942")

epoch  = empyrean.Epochs.from_mjd([61500.0], scale="tdb")
phi    = chain.stms_array()[chain.index_at(epoch)]     # (6, 6) Φ matrix
```

### STM convention

`stms_array()` returns **cumulative** STMs:
$\Phi(t_i,\, t_0)$ at each output epoch $t_i$, referenced to
the orbit's *initial* epoch $t_0$ (the epoch on the input
``Orbits`` row). To get the segment STM between two non-initial
epochs, compose:

$$
\Phi(t_b,\, t_a) = \Phi(t_b,\, t_0)\, \Phi(t_a,\, t_0)^{-1}
$$

```python
i_a, i_b = chain.index_at(t_a), chain.index_at(t_b)
phi_ba = chain.stms_array()[i_b] @ np.linalg.inv(chain.stms_array()[i_a])
```

Multi-chain methods raise with a hint to filter — calling
`stms_array()` on a multi-orbit table raises:

```
ValueError: stms_array() requires a single chain but got 2 unique
orbit_ids: 'A', 'B'. Filter to one chain first:
sens.select("orbit_id", "<orbit_id>").stms_array(...)
```

## Propagating a covariance

Map an initial 6×6 covariance through the chain to the output epoch:

```python
epoch_t = empyrean.Epochs.from_mjd([60750.0], scale="tdb")
cov_t, dmu_t = chain.propagate_covariance(cov_0, i=chain.index_at(epoch_t))
```

`order="auto"` uses Park-Scheeres second-order Gaussian (Park &
Scheeres 2006) when STTs are available, else the linear map:

$$
\Sigma(t) = \Phi\, \Sigma_0\, \Phi^\top
$$

The order-2 correction adds the $O(\Sigma^2)$ Isserlis term plus
the mean shift $\Delta\mu = \tfrac{1}{2}\, \Psi : \Sigma_0$. See
Park &amp; Scheeres (2006), *Nonlinear Mapping of Gaussian Statistics:
Theory and Applications to Spacecraft Trajectory Design*, J. Guidance
Control Dyn. 29(6).

## Nonlinearity diagnostic

For Jet2-propagated chains, `chain.kappa(cov_0)` returns the per-epoch
local nonlinearity diagnostic — a scalar measure of how much the
second-order STT contribution would shift the propagated mean
relative to the linear map:

```python
kappas = chain.kappa(cov_0)        # array of length n_t
```

The diagnostic is per-row, not a single number — the same orbit can
be linear at the last-observation epoch and strongly non-linear over
a multi-year propagation through a planetary close approach. Use it
qualitatively: large local values flag epochs where the first-order
covariance map is going to disagree with sample-based estimates.
Select the uncertainty method up front in
:class:`~empyrean.PropagationConfig` for the regime you expect.

## Observation-side partials

After {func}`~empyrean.generate_ephemeris`, the
{class}`~empyrean.ObservationSensitivities` table carries observation
Jacobians $\partial h / \partial x_0$ (and Hessians for
SECOND_ORDER):

```python
chain = eph.sensitivity \
    .select("orbit_id", "2020 CD3") \
    .select("obs_code", "F51")

H             = chain.jacobians_array()       # (n_t, 6, n_params)
sigma_h, dmu_h = chain.propagate_covariance(cov_0)   # returns (cov, mean-shift)
```

`n_params` is 6 for state-only DC, 9 when fitting non-grav A1/A2/A3.
