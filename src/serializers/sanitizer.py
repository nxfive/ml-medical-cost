from typing import Any


def sanitize_params(params: dict[str, Any]) -> dict[str, Any]:
    """
    Converts callables in params dict to their __name__ for
    serialization.
    """
    sanitized = {}
    for k, v in params.items():
        if callable(v):
            sanitized[k] = v.__name__
        else:
            sanitized[k] = v
    return sanitized
