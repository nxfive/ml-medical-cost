from dataclasses import dataclass

from src.data.core import DataLoader, DataSaver


@dataclass
class BuildResult:
    loader: DataLoader
    saver: DataSaver
