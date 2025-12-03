from pathlib import Path


class PathManager:
    @staticmethod
    def ensure_dir(path: Path, parents: bool = True, exist_ok: bool = True) -> Path:
        """
        Ensures that a directory exists. Creates it if needed.
        """
        path.mkdir(parents=parents, exist_ok=exist_ok)
        return path

    @staticmethod
    def remove_file(path: Path):
        """
        Removes file if it exists.
        """
        path.unlink(missing_ok=True)

    @staticmethod
    def exists(path: Path) -> bool:
        """
        Returns True if the given path exists, False otherwise.
        """
        return path.exists()
