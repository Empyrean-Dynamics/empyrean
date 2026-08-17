"""Photometric parameters for apparent magnitude computation."""

import quivr as qv


class PhotometricParams(qv.Table):
    """Photometric parameters for apparent magnitude computation.

    Phase function models:
        "hg"    -- Classical HG model (Bowell et al., 1989): H, G
        "hg1g2" -- Three-parameter model (Muinonen et al., 2010): H, G1, G2
        "hg12"  -- Two-parameter model (Muinonen et al., 2010): H, G12

    The apparent V-band magnitude is:
        V(alpha) = H + 5*log10(r*Delta) + phi(alpha)
    """

    model = qv.LargeStringColumn()  # "hg", "hg1g2", "hg12"
    # Absolute magnitude. A null H is "this row has no photometry" — the
    # model tag alone predicts nothing — and is carried through as such,
    # never coerced to 0.0. H = 0.0 itself is refused on the way to the
    # engine: it is the C ABI's "no absolute magnitude supplied" value
    # and no minor body has it.
    h = qv.Float64Column()
    g = qv.Float64Column(nullable=True)  # G (HG model)
    g1 = qv.Float64Column(nullable=True)  # G1 (HG1G2 model)
    g2 = qv.Float64Column(nullable=True)  # G2 (HG1G2 model)
    g12 = qv.Float64Column(nullable=True)  # G12 (HG12 model)

    # 3x3 covariance over (H, slope1, slope2), row-major flattened (9
    # values). Which slope a row/column names follows `model`: slope1 is
    # G for "hg", G1 for "hg1g2", G12 for "hg12"; slope2 is G2 for
    # "hg1g2" and unused otherwise.
    #
    # Populated by the post-OD photometric fit, by SBDB ingest (a
    # diagonal built from the published H/G sigmas), and by the orbit
    # file readers. With it attached, `generate_ephemeris` reports
    # mag_sigma = sqrt(sigma_photo^2 + sigma_state^2) instead of the
    # state contribution alone.
    #
    # Rows and columns of parameters the producing fit held fixed are
    # zero, which is what an H-only fit emits. Null for a hand-built
    # orbit with no photometric uncertainty — never a block of zeros,
    # which would read as a supplied zero uncertainty.
    #
    # Must be symmetric, and is validated as such on the way to the
    # engine: the parquet / CSV / JSONL orbit formats carry only the six
    # lower-triangle cells (phot_cov_00/10/11/20/21/22) and mirror them
    # on read, so an asymmetric matrix has no file representation and
    # would come back truncated rather than mirrored.
    covariance = qv.LargeListColumn(qv.Float64Column(), nullable=True)
