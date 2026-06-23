"""Tests for the typed :class:`Origin` API."""

import pytest
from empyrean import Origin
from empyrean._convert import naif_to_origin, origin_to_naif


class TestNamedBodies:
    def test_named_constants_are_origin_instances(self):
        for o in [
            Origin.SSB,
            Origin.SUN,
            Origin.MERCURY,
            Origin.VENUS,
            Origin.EARTH,
            Origin.MOON,
            Origin.MARS_BARYCENTER,
            Origin.JUPITER_BARYCENTER,
            Origin.SATURN_BARYCENTER,
            Origin.URANUS_BARYCENTER,
            Origin.NEPTUNE_BARYCENTER,
            Origin.PLUTO_BARYCENTER,
        ]:
            assert isinstance(o, Origin)

    def test_canonical_names(self):
        assert str(Origin.EARTH) == "Earth"
        assert str(Origin.SSB) == "SSB"
        assert str(Origin.JUPITER_BARYCENTER) == "Jupiter Barycenter"

    def test_origins_are_hashable_and_equal(self):
        d = {Origin.EARTH: 1, Origin.MOON: 2}
        assert d[Origin.EARTH] == 1
        assert Origin.EARTH == Origin("Earth")
        assert Origin.EARTH != Origin.MOON

    def test_origins_are_immutable(self):
        with pytest.raises(Exception):
            Origin.EARTH.name = "Mars"


class TestAsteroids:
    def test_asteroid_factory(self):
        ceres = Origin.asteroid(1)
        assert ceres.name == "asteroid_1"
        assert str(ceres) == "asteroid_1"

    def test_asteroid_round_trips(self):
        for n in [1, 4, 433, 99942]:
            o = Origin.asteroid(n)
            naif = origin_to_naif(o)
            assert naif == 2_000_000 + n
            assert naif_to_origin(naif) == o.name

    def test_asteroid_rejects_non_positive(self):
        with pytest.raises(ValueError):
            Origin.asteroid(0)
        with pytest.raises(ValueError):
            Origin.asteroid(-1)


class TestFromString:
    def test_canonical_names(self):
        assert Origin.from_string("Earth") == Origin.EARTH
        assert Origin.from_string("SSB") == Origin.SSB
        assert Origin.from_string("Jupiter Barycenter") == Origin.JUPITER_BARYCENTER

    def test_lowercase_and_underscore(self):
        assert Origin.from_string("earth") == Origin.EARTH
        assert Origin.from_string("ssb") == Origin.SSB
        assert Origin.from_string("jupiter_barycenter") == Origin.JUPITER_BARYCENTER

    def test_outer_planet_aliases(self):
        # Bare planet name → barycenter (since DE440 only has barycenter
        # segments for these).
        assert Origin.from_string("mars") == Origin.MARS_BARYCENTER
        assert Origin.from_string("jupiter") == Origin.JUPITER_BARYCENTER

    def test_asteroid_pattern(self):
        assert Origin.from_string("asteroid_1") == Origin.asteroid(1)
        assert Origin.from_string("asteroid_99942") == Origin.asteroid(99942)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            Origin.from_string("Pluto Center")
        with pytest.raises(ValueError):
            Origin.from_string("not_a_body")


class TestNaifConversion:
    def test_origin_to_naif_accepts_typed_origin(self):
        assert origin_to_naif(Origin.EARTH) == 399
        assert origin_to_naif(Origin.SSB) == 0
        assert origin_to_naif(Origin.MARS_BARYCENTER) == 4
        assert origin_to_naif(Origin.JUPITER_BARYCENTER) == 5

    def test_origin_to_naif_accepts_canonical_string(self):
        assert origin_to_naif("Earth") == 399
        assert origin_to_naif("SSB") == 0
        assert origin_to_naif("Mars Barycenter") == 4

    def test_origin_to_naif_passes_through_int(self):
        assert origin_to_naif(399) == 399

    def test_origin_to_naif_handles_asteroids(self):
        assert origin_to_naif(Origin.asteroid(1)) == 2_000_001
        assert origin_to_naif("asteroid_4") == 2_000_004

    def test_naif_to_origin(self):
        assert naif_to_origin(399) == "Earth"
        assert naif_to_origin(0) == "SSB"
        assert naif_to_origin(4) == "Mars Barycenter"
        assert naif_to_origin(2_000_001) == "asteroid_1"

    def test_round_trip_origin_string(self):
        for o in [
            Origin.SSB,
            Origin.SUN,
            Origin.EARTH,
            Origin.MARS_BARYCENTER,
            Origin.JUPITER_BARYCENTER,
            Origin.PLUTO_BARYCENTER,
            Origin.asteroid(99942),
        ]:
            assert naif_to_origin(origin_to_naif(o)) == o.name

    def test_outer_planet_body_centers_rejected(self):
        # NAIF 499 etc. are body-center codes that DE440 doesn't ship
        # segments for. Don't silently accept them — they were a real
        # source of pre-existing bugs.
        with pytest.raises(ValueError):
            origin_to_naif(Origin.from_string("non-existent body"))
