import shutil
from abc import ABC, abstractmethod
from pathlib import Path

import kagglehub


class BaseDownloader(ABC):
    @abstractmethod
    def download(self, path: Path) -> Path: ...


class KaggleDownloader(BaseDownloader):
    def __init__(self, handle: str, filename: str):
        self.handle = handle
        self.filename = filename

    def download(self, path: Path) -> Path:
        """
        Downloads dataset from Kaggle, copies it to the target directory, and cleans up temporary files.
        """
        temp_dir = kagglehub.dataset_download(self.handle)
        self.copy_dir(temp_dir, path)
        self.cleanup_temp(temp_dir)
        return path / self.filename

    @staticmethod
    def copy_dir(src: Path, dst: Path):
        """
        Copies all files from src directory to dst directory.
        """
        shutil.copytree(src, dst, dirs_exist_ok=True)

    @staticmethod
    def cleanup_temp(path: Path):
        """
        Deletes the directory at the given path, ignoring errors.
        """
        shutil.rmtree(path, ignore_errors=True)


class DatasetDownloader:
    def __init__(self, raw_dir: Path, downloader: BaseDownloader):
        self.raw_dir = raw_dir
        self.downloader = downloader

    def download(self) -> Path:
        """
        Downloads the dataset using the configured downloader and returns the path to the downloaded file.
        """
        return self.downloader.download(self.raw_dir)
