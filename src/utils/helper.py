from pathlib import Path


def create_directory(directory: Path) -> None:
    """
    Create directory if it does not already exist.
    """

    directory.mkdir(parents=True, exist_ok=True)