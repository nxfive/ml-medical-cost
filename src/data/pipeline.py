import pandas as pd

from src.data.data import fetch_data, split_data
from src.features.features import convert_features_type


def data_pipeline() -> pd.DataFrame:
    """
    Loads and preprocesses the dataset, then splits it into training and test sets.
    """
    df = fetch_data()
    df = convert_features_type(df)
    
    return split_data(df)
