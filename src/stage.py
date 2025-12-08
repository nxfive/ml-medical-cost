from importlib import import_module

from omegaconf import DictConfig

from .config_loader import load_stage_configs

STAGE_MODULES = {
    "data": "src.data.run",
    "training": "src.training.run",
    "optuna": "src.optuna.run",
}


def run(cfg: DictConfig):
    data_stage_cfg, training_stage_cfg = load_stage_configs(cfg)

    stage_cfg_map = {
        "data": data_stage_cfg,
        "training": training_stage_cfg,
        "optuna": cfg,
    }
    stage = cfg.stage
    
    if stage not in STAGE_MODULES:
        raise ValueError(
            f"Stage '{stage}' unknown. Available: {list(STAGE_MODULES.keys())}"
        )

    module = import_module(STAGE_MODULES[stage])
    module.run(stage_cfg_map[stage])
