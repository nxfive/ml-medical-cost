from src.conf.schema import DataStageConfig
from src.factories.data_factory import DataFactory
from src.features.core import convert_features_type
from src.io.file_ops import PathManager
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
        self.build()
        df = self.data.fetch()
        df = convert_features_type(df)
        self.data.split(df)
