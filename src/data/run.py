from omegaconf import DictConfig

from .pipeline import DataPipeline


def run(cfg: DictConfig):
    pipeline = DataPipeline(
        cfg.data.raw_dir, cfg.data.processed_dir, cfg.kaggle.handle, cfg.kaggle.filename
    )
    pipeline.run()
