from omegaconf import DictConfig
from sklearn.model_selection import KFold


def get_cv(cfg: DictConfig) -> KFold:
    """
    Returns a KFold cross-validator configured based on the values from config.
    """
    return KFold(
        n_splits=cfg.cv.n_splits,
        shuffle=cfg.cv.shuffle,
        random_state=cfg.cv.random_state,
    )
