import hydra
from omegaconf import DictConfig


STAGE_MODULES = {
    "data": "src.data.pipeline",
    "training": "src.training.pipeline",
    "optuna": "src.optuna.pipeline",
}

def run_stage(cfg: DictConfig):
    stage = cfg.stage
    if stage not in STAGE_MODULES:
        raise ValueError(f"Stage '{stage}' unknown. Available: {list(STAGE_MODULES.keys())}")

    module = __import__(STAGE_MODULES[stage], fromlist=["run"])
    module.run(cfg)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    run_stage(cfg)


if __name__ == "__main__":
    main()
