from src.conf.schema import DataStageConfig

from .pipeline import DataPipeline


def run(cfg: DataStageConfig):
    pipeline = DataPipeline(cfg)
    pipeline.run()
