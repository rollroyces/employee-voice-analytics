"""Exception types shared across layer modules.

Kept in their own module so analyzers.py and layers.py can import these
without creating a circular dependency (analyzers._detect_backends raises
LayerDependencyError, and layers imports from analyzers).
"""


class LayerError(RuntimeError):
    """Base class for layer-enforcement failures."""


class LayerDependencyError(LayerError):
    """Raised when a layer's required Python module is missing and
    ALLOW_FALLBACK is False. The message includes the pip-install hint."""


class LayerContractError(LayerError):
    """Raised when a layer's required output columns are missing or entirely
    null after the run completes."""


class LayerOrderError(LayerError):
    """Raised when a downstream layer is invoked before an upstream layer
    has populated the columns it depends on."""
