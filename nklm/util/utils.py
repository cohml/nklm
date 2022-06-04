from pathlib import Path


def full_path(p: str) -> Path:
    """Given relative path, return associated full path."""
    return Path(p).resolve()
