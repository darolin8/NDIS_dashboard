# utils/__init__.py
# Make this a proper Python package and (optionally) re-export helpers.

from .factor_labels import shorten_factor  # re-export for convenience

__all__ = ["shorten_factor"]


