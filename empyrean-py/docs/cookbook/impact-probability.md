# Close approach, B-plane, and impact probability

This is the planetary-science headline workflow: take an orbit with
covariance, propagate it through a planetary close approach, and
report $(B\cdot R,\, B\cdot T)$ geometry plus impact probability
under multiple uncertainty-mapping methods.

The examples below use the four canonical close-approach scenarios
shared with the empyrean 3D viewer — Apophis (2029 deep Earth flyby),
2024 YR4 (the current attention-getter), 2020 CD3 (a mini-moon
temporary capture), and 2008 TC3 (the first asteroid predicted to
impact Earth, before the Almahata Sitta airburst over Sudan in 2008).
The MJDs called out alongside each `query_sbdb` example match the
default jump-to epochs in the empyrean viewer so you can cross-walk
between this page and a rendered trajectory.

## Apophis through 2029

```python
import empyrean
from empyrean import UncertaintyMethod, MonteCarlo

empyrean.initialize()

orbits = empyrean.query_sbdb(["99942"])           # Apophis, with covariance

end_epoch = empyrean.Epochs.from_mjd([63000.0], scale="tdb")   # 2031-05, past the 2029-04 CA

# One full propagation per method — the result tables tag every row
# with which method produced it, so you can compare in one query.
ips = empyrean.compute_impact_probabilities(
    orbits,
    end_epoch,
    methods=[
        UncertaintyMethod.FIRST_ORDER,             # Φ Σ Φᵀ floor
        UncertaintyMethod.SECOND_ORDER,            # Park-Scheeres NL correction
        MonteCarlo(n_samples=100_000, seed=42),    # tail probability via VA sample
    ],
    body_filter=["Earth", "Moon"],                  # Apophis 2029 grazes the Moon's
                                                    # gravitational influence — include it
                                                    # for accurate IP.
)

bps = empyrean.compute_b_planes(
    orbits,
    end_epoch,
    methods=[UncertaintyMethod.FIRST_ORDER, UncertaintyMethod.SECOND_ORDER],
    body_filter=["Earth"],
)
```

Both calls dispatch one full propagation per method internally — the
cost scales linearly with `len(methods)`. For risk-list operations
across thousands of orbits, run `FirstOrder` only and reach for the
non-linear methods on the screened-in subset.

## Reading the impact-probability table

{class}`~empyrean.ImpactProbabilities` is a quivr table with one row
per `(method × orbit × body)` close approach:

```python
# Drop into the Apophis 2029 row, second-order method
ap_2029 = (
    ips.select("orbit_id", "(99942) Apophis").select("method", "second_order")
)

print(f"closest approach     {ap_2029.epochs.to_iso()[0]}")
print(f"miss distance        {ap_2029.miss_distance_km[0].as_py():,.0f} km")
print(f"σ along miss vector  {ap_2029.sigma_distance_km[0].as_py():,.0f} km")
print(f"linear IP            {ap_2029.ip_linear[0].as_py():.3e}")
print(f"second-order IP      {ap_2029.ip_second_order[0].as_py():.3e}")
print(f"non-linearity κ      {ap_2029.nonlinearity[0].as_py():.3f}")
```

| Column                | Meaning                                                                    |
|-----------------------|----------------------------------------------------------------------------|
| `miss_distance_km`    | Closest-approach geocentric distance at the nominal trajectory             |
| `sigma_distance_km`   | 1σ uncertainty along the miss-distance direction (linearised)              |
| `effective_radius_km` | Body radius inflated for atmospheric capture — what IP is computed against |
| `ip_linear`           | Linear (Φ Σ Φᵀ-mapped) impact probability                                  |
| `ip_second_order`     | Park-Scheeres second-order Gaussian IP (populated when method ≥ Jet2)      |
| `ip_mc`               | Monte-Carlo fraction `mc_n_impacts / mc_n_samples`                         |
| `nonlinearity`        | Local nonlinearity diagnostic at the close-approach epoch — included for completeness; do not use for method selection |

Compare the IP estimates across methods on the same row — divergence
between `ip_linear` and `ip_second_order` is the cleanest signal that
the linear approximation is breaking down.

## B-plane geometry

The B-plane is the encounter plane perpendicular to the hyperbolic
excess velocity vector. {class}`~empyrean.BPlanes` reports
$(B\cdot T,\, B\cdot R)$ coordinates plus the projected
covariance, in the canonical Kizner frame:

```python
ap_bp = bps.select("method", "second_order")

print(f"B·T               {ap_bp.b_dot_t_km[0].as_py():,.0f} km")
print(f"B·R               {ap_bp.b_dot_r_km[0].as_py():,.0f} km")
print(f"|B|               {ap_bp.b_mag_km[0].as_py():,.0f} km")
print(f"v_∞               {ap_bp.v_inf_km_s[0].as_py():.3f} km/s")
print(f"Earth radius      {ap_bp.body_radius_km[0].as_py():,.0f} km")
print(f"effective radius  {ap_bp.effective_radius_km[0].as_py():,.0f} km")

# 3σ uncertainty ellipse on the B-plane (semi-major / semi-minor in km,
# rotation angle in radians from the +T axis).
print(f"3σ ellipse: {ap_bp.semi_major_3sig_km[0].as_py():,.0f} × "
      f"{ap_bp.semi_minor_3sig_km[0].as_py():,.0f} km @ "
      f"{ap_bp.ellipse_angle_rad[0].as_py():.2f} rad")
```

In the Kizner frame the **T axis** is
$\hat T = (\hat S \times \hat k)/|\hat S \times \hat k|$ — the inbound
asymptote $\hat S$ crossed with the ecliptic pole $\hat k$ — so it lies
in the ecliptic plane, running approximately along the planet's orbital
motion. The **R axis** completes the right-handed triad,
$\hat R = \hat S \times \hat T$, pointing out of the ecliptic plane.
$B \cdot T$ is the *along-track / timing* coordinate and carries the
resonant-return / keyhole structure; $B \cdot R$ is the *cross-track*
coordinate and sets the minimum-distance miss.

Impact requires $|B| < R_\oplus^{\rm eff}$, where the effective
radius is gravitationally focused:

$$
R_\mathrm{eff}^2 = R_\oplus^2 \left( 1 + \frac{v_\mathrm{esc}^2}{v_\infty^2} \right)
$$

with $v_\mathrm{esc}$ Earth's escape velocity at the surface
(11.2 km/s) and $v_\infty$ the encounter's hyperbolic-excess velocity
(`v_inf_km_s`). For typical NEA encounters with $v_\infty \sim 10$ km/s
the effective radius is ≈ 1.5 × the body radius; for slow encounters
($v_\infty \sim 5$ km/s) it grows to ≈ 2.4–2.5 × the body radius.

## Method comparison in one query

```python
# Side-by-side IP per method on the Apophis 2029 row.
ap = ips.select("orbit_id", "(99942) Apophis")
for row_method, row_linear, row_so, row_mc in zip(
    ap.method.to_pylist(),
    ap.ip_linear.to_pylist(),
    ap.ip_second_order.to_pylist(),
    ap.ip_mc.to_pylist(),
):
    populated = row_so or row_mc or row_linear
    print(f"{row_method:14s}  IP = {populated:.3e}")
```

For a screened-in object, the typical sanity-check pattern is:

- All three IP estimates **within an order of magnitude** of each
  other → linear gate is trustworthy
- `ip_second_order` ≪ `ip_linear` → linear is over-estimating; the
  second-order correction shrinks the encounter ellipse
- `ip_mc` diverges from both linear and second-order → tail
  probabilities matter; report the MC value

## Driving a virtual-asteroid sample with Monte Carlo

The `MonteCarlo(n_samples=...)` method handles VA sampling
internally — draws from the input covariance, propagates each sample,
counts impacts:

```python
mc = empyrean.compute_impact_probabilities(
    orbits,
    end_epoch,
    methods=[MonteCarlo(n_samples=1_000_000, seed=42)],
    body_filter=["Earth"],
)

ap_mc = mc.select("orbit_id", "(99942) Apophis")
print(f"sampled {ap_mc.mc_n_samples[0].as_py():,d} virtual asteroids")
print(f"  {ap_mc.mc_n_impacts[0].as_py():,d} impacted Earth")
print(f"  IP_MC = {ap_mc.ip_mc[0].as_py():.3e}")
```

For finer-grained control — running propagation directly on the VA
sample and inspecting per-sample states — request `MonteCarlo` on
{func}`empyrean.propagate` instead of `compute_impact_probabilities`;
the propagated sample comes back as one orbit per VA in
`result.states`, indexable by `orbit_id`.

## Cost guidance

Approximate per-encounter cost on a single thread, Standard force
model, ~5-year propagation:

| Method              | Cost vs FirstOrder                                                       |
|---------------------|--------------------------------------------------------------------------|
| `FirstOrder`        | 1×                                                                       |
| `SecondOrder`       | ~5×                                                                      |
| `MonteCarlo(n=10⁵)` | ~10⁵× (purely sample-driven, embarrassingly parallel — set `num_threads` on the inner `PropagationConfig`) |

For large risk lists: `FirstOrder` everywhere as the screening pass,
then `SecondOrder` on the filtered subset, then `MonteCarlo` only on
the tail of objects with non-trivial linear IP.
