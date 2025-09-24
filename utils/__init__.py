# factor_labels import that works whether run as a package or a script
try:
    from .factor_labels import shorten_factor  # package-relative
except Exception:
    try:
        from factor_labels import shorten_factor  # absolute
    except Exception:
        # Safe fallback: no-op (prevents NameError if factor_labels is missing)
        def shorten_factor(x): 
            return x

__all__ = ["shorten_factor"]



