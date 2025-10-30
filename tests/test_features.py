import pandas as pd

from src.features.features import convert_features_type


def test_convert_features():
    sample_df = pd.DataFrame(
        {
            "age": [10, 20],
            "children": [1, 3],
            "sex": ["male", "female"],
            "smoker": ["yes", "no"],
        }
    )
    df = convert_features_type(sample_df)

    assert all(df[column].dtype == float for column in sample_df.columns)
