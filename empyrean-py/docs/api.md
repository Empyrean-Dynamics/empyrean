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
   empyrean.query_observations
   empyrean.read_ades
   empyrean.initialize
   empyrean.download_data
   empyrean.default_data_dir
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
   empyrean.Epochs
   empyrean.TimeScale
   empyrean.Frame
   empyrean.Origin
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
   empyrean.ForceModelTier
   empyrean.UncertaintyMethod
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
```

## Orbit determination

```{eval-rst}
.. autosummary::
   :toctree: _generated
   :nosignatures:

   empyrean.ADESObservations
   empyrean.ODConfig
   empyrean.DetermineResult
   empyrean.EvaluateResult
   empyrean.ObservationResults
   empyrean.ResidualSummary
   empyrean.AcceptabilityReport
   empyrean.StationBiases
   empyrean.OutputEpoch
   empyrean.OutputEpochMode
   empyrean.Session
   empyrean.SessionDiff
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
   empyrean.IODConfig
   empyrean.AutoEscalationPolicy
   empyrean.AcceptabilityThresholds
   empyrean.CovarianceRepresentation
```
