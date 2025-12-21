from omegaconf import DictConfig

from .base import OptunaBasePipeline
from .pipeline import OptunaPipeline


def run(cfg: DictConfig):
    base = OptunaBasePipeline(dynamic_cfg=cfg)
    static_config = base.run()

    pipeline = OptunaPipeline(cfg=static_config)
    pipeline.run()
