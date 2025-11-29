from omegaconf import OmegaConf
from sklearn.model_selection import KFold

from src.utils.cv import get_cv


def test_get_cv():
    cfg_cv = OmegaConf.create(
        {"cv": {"n_splits": 10, "shuffle": True, "random_state": 42}}
    )
    kfold_cv = get_cv(cfg_cv)

    assert isinstance(kfold_cv, KFold)
    assert kfold_cv.get_n_splits() == 10
    assert kfold_cv.shuffle is True
    assert kfold_cv.random_state == 42
