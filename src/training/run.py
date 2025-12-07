from src.conf.schema import TrainingStageConfig

from .pipeline import TrainingPipeline


def run(cfg: TrainingStageConfig):
    pipeline = TrainingPipeline(cfg)
    pipeline.run()
