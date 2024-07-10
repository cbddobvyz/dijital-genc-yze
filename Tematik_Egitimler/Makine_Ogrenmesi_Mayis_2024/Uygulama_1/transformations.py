from typing import List

from pandas import DataFrame
from sklearn.preprocessing import PowerTransformer, StandardScaler


def standardize_dataset(df: DataFrame):
    scaler = StandardScaler().fit(df)
    df2 = DataFrame(data=scaler.transform(df), columns=df.columns)
    return df2


def normalize_dataset(df: DataFrame,
                      all_pos=False,
                      min_val: float = 0.000001,
                      exclude: List[str] = []):
    df2 = df.copy()
    for c in df.columns:
        if c not in exclude:
            if all_pos:
                df2[c] = min_val + (df[c] - df[c].min()) / (df[c].max() - df[c].min() - min_val)
            else:
                df2[c] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())
    return df2


def box_cox(data,
            normalize=False,
            standardize=False):
    fitter = PowerTransformer(method="box-cox",
                              standardize=standardize)
    dt = fitter.fit_transform(data)
    if normalize:
        dt = (dt - dt.min()) / (dt.max() - dt.min())
    return dt


def yeo_johnson(data,
                normalize=False,
                standardize=False):
    fitter = PowerTransformer(method="yeo-johnson",
                              standardize=standardize)
    dt = fitter.fit_transform(data)
    if normalize:
        dt = (dt - dt.min()) / (dt.max() - dt.min())
    return dt
