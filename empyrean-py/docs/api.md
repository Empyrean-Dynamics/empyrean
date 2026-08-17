# API reference

## Pipeline entry points

```{eval-rst}
.. autosummary::
   :toctree: _generated
   :nosignatures:

   empyrean.propagate
   empyrean.generate_ephemeris
   empyrean.determine
   empyrean.evaluate
   empyrean.refine
   empyrean.transform_coordinates
   empyrean.get_states
   empyrean.get_observer_states
   empyrean.compute_impact_probabilities
   empyrean.compute_b_planes
```

## I/O helpers

```{eval-rst}
.. autosummary::
   :toctree: _generated
   :nosignatures:

   empyrean.query_sbdb
   empyrean.query_horizons
   empyrean.query_horizons_vectors
   empyrean.query_observations
   empyrean.query_radar
   empyrean.read_ades
   empyrean.initialize
   empyrean.download_data
   empyrean.default_data_dir
   empyrean.versions
   empyrean.version_string
```

## Coordinates & orbits

```{eval-rst}
.. autosummary::
   :toctree: _generated
   :nosignatures:

   empyrean.CartesianCoordinates
   empyrean.KeplerianCoordinates
   empyrean.CometaryCoordinates
   empyrean.SphericalCoordinates
   empyrean.CartesianCovariance
   empyrean.KeplerianCovariance
   empyrean.CometaryCovariance
   empyrean.SphericalCovariance
   empyrean.CartesianOrbits
   empyrean.KeplerianOrbits
   empyrean.CometaryOrbits
   empyrean.SphericalOrbits
   empyrean.NonGravParams
   empyrean.PhotometricParams
   empyrean.SRPParams
   empyrean.WideCross
   empyrean.Epochs
   empyrean.TimeScale
   empyrean.Frame
   empyrean.Origin
```

### Continuous thrust

```{eval-rst}
.. autosummary::
   :toctree: _generated
   :nosignatures:

   empyrean.ThrustParams
   empyrean.ThrustArc
   empyrean.SteeringLaw
   empyrean.ConstantRTN
   empyrean.VelocityTangent
   empyrean.InertialFixed
```

## Observers

```{eval-rst}
.. autosummary::
   :toctree: _generated
   :nosignatures:

   empyrean.Observers
```

## Propagation

```{eval-rst}
.. autosummary::
   :toctree: _generated
   :nosignatures:

   empyrean.PropagationConfig
   empyrean.PropagationResult
   empyrean.DiagnosticsConfig
   empyrean.AdvancedIntegratorConfig
   empyrean.IntegratorChoice
   empyrean.OriginSwitchingConfig
   empyrean.EphemerisOverlapPolicy
   empyrean.ForceModelTier
   empyrean.UncertaintyMethod
   empyrean.Auto
   empyrean.SigmaPoint
   empyrean.MonteCarlo
   empyrean.EventConfig
   empyrean.Events
   empyrean.EventSummary
   empyrean.CloseApproachStarts
   empyrean.CloseApproachEnds
   empyrean.Periapses
   empyrean.Impacts
   empyrean.PossibleImpacts
   empyrean.AtmosphericEntries
   empyrean.AtmosphericExits
   empyrean.CaptureStarts
   empyrean.CaptureEnds
   empyrean.ShadowEntries
   empyrean.ShadowExits
   empyrean.CovarianceRegimeChanges
```

### Tagged covariance

The per-epoch covariance readback, carrying the joint's off-diagonal
blocks alongside the 6×6 state block.

```{eval-rst}
.. autosummary::
   :toctree: _generated
   :nosignatures:

   empyrean.TaggedCovariance
   empyrean.TaggedCovariances
   empyrean.CovarianceKind
   empyrean.CovarianceQuality
   empyrean.TargetFunctional
   empyrean.GaussianMixture
```

### System handles

```{eval-rst}
.. autosummary::
   :toctree: _generated
   :nosignatures:

   empyrean.build_system
   empyrean.od_system
   empyrean.BuiltSystem
   empyrean.SystemDescription
   empyrean.KernelRecord
   empyrean.KernelKind
   empyrean.KernelProvenance
```

## Ephemeris

```{eval-rst}
.. autosummary::
   :toctree: _generated
   :nosignatures:

   empyrean.Ephemeris
   empyrean.EphemerisConfig
   empyrean.EphemerisResult
```

## Impact probability and B-plane

```{eval-rst}
.. autosummary::
   :toctree: _generated
   :nosignatures:

   empyrean.ImpactProbabilities
   empyrean.BPlanes
```

## Observation planning

```{eval-rst}
.. autosummary::
   :toctree: _generated
   :nosignatures:

   empyrean.evaluate_plan
   empyrean.PlanResult
   empyrean.PlanMetrics
   empyrean.PlanCandidates
   empyrean.PlanEphemeris
```

### Planning configuration

```{eval-rst}
.. autosummary::
   :toctree: _generated
   :nosignatures:

   empyrean.PlanningConfig
   empyrean.ObservatoryConfig
   empyrean.STAGE_PRIOR
   empyrean.STAGE_POSTERIOR
   empyrean.PlannedObservation
   empyrean.PlannedObservationKind
   empyrean.RadarMode
   empyrean.RadarStation
```

## Math primitives

```{eval-rst}
.. autosummary::
   :toctree: _generated
   :nosignatures:

   empyrean.eigenvector_max_6x6
   empyrean.split_gaussian
   empyrean.MixtureComponent
```

## Sensitivity

```{eval-rst}
.. autosummary::
   :toctree: _generated
   :nosignatures:

   empyrean.StateSensitivities
   empyrean.ObservationSensitivities
   empyrean.SENSITIVITY_ROW_RANGE
   empyrean.SENSITIVITY_ROW_RA
   empyrean.SENSITIVITY_ROW_DEC
   empyrean.SENSITIVITY_ROW_VRANGE
   empyrean.SENSITIVITY_ROW_VRA
   empyrean.SENSITIVITY_ROW_VDEC
```

## Orbit determination

```{eval-rst}
.. autosummary::
   :toctree: _generated
   :nosignatures:

   empyrean.ADESObservations
   empyrean.ADESRadarObservations
   empyrean.ODConfig
   empyrean.DetermineResult
   empyrean.DetermineResults
   empyrean.DetermineFailure
   empyrean.EvaluateResult
   empyrean.ObservationResults
   empyrean.ResidualSummary
   empyrean.FitSummary
   empyrean.BandStat
   empyrean.AcceptabilityReport
   empyrean.GateRecord
   empyrean.SolvedCovariance
   empyrean.CovarianceTrust
   empyrean.TrustGateEvent
   empyrean.StationBiases
   empyrean.OutputEpoch
   empyrean.OutputEpochMode
   empyrean.Session
   empyrean.SessionDiff
```

### Post-OD photometry

```{eval-rst}
.. autosummary::
   :toctree: _generated
   :nosignatures:

   empyrean.PhotometryConfig
   empyrean.PhotometryModel
   empyrean.PhotometryResult
```

### OD configuration

```{eval-rst}
.. autosummary::
   :toctree: _generated
   :nosignatures:

   empyrean.WeightingConfig
   empyrean.WeightingLayer
   empyrean.WeightingLayerKind
   empyrean.WeightingPreset
   empyrean.SigmaPolicy
   empyrean.DebiasingConfig
   empyrean.DebiasingResolution
   empyrean.StationRaDecConfig
   empyrean.RejectionConfig
   empyrean.RejectionKind
   empyrean.SolveForParams
   empyrean.SolveFor
   empyrean.ParamDisposition
   empyrean.IODConfig
   empyrean.OriginPolicy
   empyrean.OriginPolicyMode
   empyrean.AutoEscalationPolicy
   empyrean.AcceptabilityThresholds
   empyrean.CovarianceRepresentation
```
