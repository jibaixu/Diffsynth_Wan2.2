__all__ = ["BCViLTPolicy"]


def __getattr__(name):
    if name == "BCViLTPolicy":
        from .vilt import BCViLTPolicy

        return BCViLTPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
