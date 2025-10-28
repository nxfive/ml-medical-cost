import pandas as pd


def convert_features_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts specific columns in the DataFrame to float and encodes categorical features.
    """
    df = df.copy()

    df["age"] = df["age"].astype(float)
    df["children"] = df["children"].astype(float)
    df["sex"] = (df["sex"] == "female").astype(float)
    df["smoker"] = (df["smoker"] == "yes").astype(float)
    
    return df
