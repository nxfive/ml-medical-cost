from pathlib import Path

from src.factories.data_factory import DataFactory
from src.features.core import convert_features_type
from src.io.file_ops import PathManager
from src.patterns.base_pipeline import BasePipeline


class DataPipeline(BasePipeline[None]):
    def __init__(
        self, raw_dir: str, processed_dir: str, kaggle_handle: str, kaggle_filename: str
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)

        self.data = DataFactory.create(
            raw_dir=self.raw_dir,
            processed_dir=self.processed_dir,
            kaggle_handle=kaggle_handle,
            kaggle_filename=kaggle_filename,
        )

    def build(self) -> None:
        """
        Ensures that the required directories exist.
        """
        PathManager.ensure_dir(self.raw_dir)
        PathManager.ensure_dir(self.processed_dir)

    def run(self) -> None:
        """
        Executes a full data pipeline.
        """
        self.build()
        df = self.data.fetch()
        df = convert_features_type(df)
        self.data.split(df)
