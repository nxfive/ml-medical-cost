from omegaconf import DictConfig

from src.data.data import fetch_data, split_data
from src.features.features import convert_features_type


def run(cfg: DictConfig):
    """
    Loads and preprocesses the dataset, then splits it into training and test sets.
    """
    df = fetch_data(cfg)
    df = convert_features_type(df)
    split_data(df, cfg)
