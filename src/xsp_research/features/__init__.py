"""Feature framework: registry + families.

Importing this package registers every daily feature family. Surface features
(family F) live in `surface.py` with a snapshot-based API and are not part of
the daily registry build.
"""

from xsp_research.features import (  # noqa: F401  (imports populate REGISTRY)
    breadth,
    cross_asset,
    pca_regime,
    trend,
    volatility,
)
from xsp_research.features.registry import (  # noqa: F401
    REGISTRY,
    BuildResult,
    FeatureFamily,
    FeatureSpec,
    MarketDataBundle,
    build_features,
)
