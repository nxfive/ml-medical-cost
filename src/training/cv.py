from sklearn.model_selection import KFold

from src.conf.schema import CVConfig


def get_cv(cfg: CVConfig) -> KFold:
    """
    Returns a KFold cross-validator configured based on the values from config.
    """
    return KFold(
        n_splits=cfg.n_splits,
        shuffle=cfg.shuffle,
        random_state=cfg.random_state,
    )
