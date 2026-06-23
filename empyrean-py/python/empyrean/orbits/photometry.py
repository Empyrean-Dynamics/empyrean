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
    h = qv.Float64Column()  # absolute magnitude
    g = qv.Float64Column(nullable=True)  # G (HG model)
    g1 = qv.Float64Column(nullable=True)  # G1 (HG1G2 model)
    g2 = qv.Float64Column(nullable=True)  # G2 (HG1G2 model)
    g12 = qv.Float64Column(nullable=True)  # G12 (HG12 model)
