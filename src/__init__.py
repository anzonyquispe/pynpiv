"""
npivlib: Nonparametric IV estimation tools.

This package exposes:

- npiv      : main estimator
- plot_npiv : plotting helper

and also makes internal modules importable as:

    import npivlib.util_npiv
    from npivlib import util_npiv
"""

# Top-level API
from .npiv import npiv
from .plotting import plot_npiv

# Re-export submodules so users can do:
#   from npivlib import util_npiv
# or:
#   import npivlib.util_npiv
from . import util_npiv
from . import glp_model_matrix
from . import gsl_bspline
from . import mgcv_tensor
from . import prodspline

__all__ = [
    "npiv",
    "plot_npiv",
    "util_npiv",
    "glp_model_matrix",
    "gsl_bspline",
    "mgcv_tensor",
    "prodspline",
]
