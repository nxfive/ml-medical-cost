from src.conf.schema import OptunaStageConfig
from src.containers.data import SplitData
from src.containers.experiment import ExperimentContext
from src.dto.config import OptunaExperimentConfig


class ExperimentSerializer:
    @staticmethod
    def from_optuna_stage(
        cfg: OptunaStageConfig, split_data: SplitData
    ) -> ExperimentContext:
        return ExperimentContext(
            model_cfg=cfg.model,
            features_cfg=cfg.features,
            optuna_model_cfg=cfg.optuna_model_config,
            transformers_cfg=cfg.transformers,
            X_train=split_data.X_train,
            X_test=split_data.X_test,
            y_train=split_data.y_train,
        )

    @staticmethod
    def to_experiment_config(context: ExperimentContext) -> OptunaExperimentConfig:
        return OptunaExperimentConfig(
            model=context.model_cfg,
            features=context.features_cfg,
            transformers=context.transformers_cfg,
            optuna_model_config=context.optuna_model_cfg,
        )
