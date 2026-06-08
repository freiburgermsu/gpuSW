"""Exception hierarchy for gpusw."""
from __future__ import annotations


class GpuSWError(Exception):
    """Base class for every error raised by gpusw."""


class NoGpuError(GpuSWError):
    """CuPy or a usable NVIDIA GPU is not available.

    gpusw keeps CuPy a *soft* dependency: importing the package and using the
    pure-NumPy reference (:func:`gpusw.cpu_reference_score`) never needs a GPU.
    Only the on-GPU code paths raise this, with an actionable message.
    """


class SchemeError(GpuSWError):
    """A :class:`~gpusw.Scheme` / :class:`~gpusw.Matrix` is invalid or inconsistent."""


class OverflowRiskError(GpuSWError):
    """A forced ``dtype="int16"`` cannot represent the worst-case score range.

    Raised at scheme/encode time (never a silent wrap). Use ``dtype="auto"`` to
    let gpusw promote to int32 automatically, or ``dtype="int32"`` explicitly.
    """


class EncodeError(GpuSWError):
    """Inputs could not be funnelled into the discrete encoded form."""
