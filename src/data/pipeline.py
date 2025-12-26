import time

from src.conf.schema import DataStageConfig
from src.factories.data_factory import DataFactory
from src.features.core import convert_features_type
from src.io.file_ops import PathManager
from src.logger.setup import logger
from src.patterns.base_pipeline import BasePipeline


class DataPipeline(BasePipeline[None, None]):
    def __init__(self, cfg: DataStageConfig):
        self.cfg = cfg
        self.data = DataFactory.create(cfg=self.cfg)

    def build(self) -> None:
        """
        Ensures that the required directories exist.
        """
        PathManager.ensure_dir(self.cfg.data_dir.raw_dir)
        PathManager.ensure_dir(self.cfg.data_dir.processed_dir)

    def run(self) -> None:
        """
        Executes a full data pipeline.
        """
        logger.info("Running data stage")
        start_data = time.perf_counter()

        logger.info("Initializing data pipeline environment")
        self.build()

        logger.info("Fetching dataset")
        df = self.data.fetch()

        logger.info("Converting features to numeric types [int/float]")
        df = convert_features_type(df)

        logger.info("Preparing train/test splits")
        self.data.split(df)

        end_data = time.perf_counter()
        logger.info(f"Data stage completed in {end_data - start_data:.2f}s")
