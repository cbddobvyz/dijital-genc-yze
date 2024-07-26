from pandas import DataFrame
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def encode_one_hot(df: DataFrame,
                   column: str) -> DataFrame:
    encoder = OneHotEncoder(handle_unknown='ignore',
                            sparse_output=False)
    df2 = DataFrame(data=df[column],
                    columns=[column])
    encoder.fit(df2)
    transformed = encoder.transform(df2)
    for column_counter in range(transformed.shape[1]):
        column_name = column + "_T" + str(column_counter)
        df[column_name] = transformed[:, column_counter]
    return df


def ordinal_encoding(df: DataFrame,
                     column: str) -> DataFrame:
    encoder = OrdinalEncoder()
    df2 = DataFrame(data=df[column],
                    columns=[column])
    encoder.fit(df2)
    df[column + "OrdT"] = encoder.transform(df2)
    return df
