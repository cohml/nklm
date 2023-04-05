from pathlib import Path


def full_path(p: str) -> Path:
    """Given relative path, return associated absolute path."""
    return Path(p).resolve()
