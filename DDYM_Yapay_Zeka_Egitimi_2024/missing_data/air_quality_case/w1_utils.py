import pandas, numpy, random, math, copy

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Basic Functionalities

# Show Feature Completeness
# Show Line Completeness


# Show Overall Completeness

def completeness_overall(df: pandas.DataFrame):
    if df.size > 0:
        return 100 - (df.isna().sum().sum() / df.size * 100)
    return 0


def completeness_feature(df: pandas.DataFrame):
    report = {}
    for colum_name in df.columns:
        report[colum_name] = completeness_overall(df[colum_name])
    return report

def completeness_data_point(df: pandas.DataFrame):
    return df.notna().all(axis=1).sum(), df.notna().all(axis=1).sum() / df.shape[0] * 100


def remove_columns(df: pandas.DataFrame, completeness_limit: float):
    comp_report = completeness_feature(df)

    to_drop = []
    for feature_name in comp_report:
        if comp_report[feature_name] < completeness_limit:
            to_drop.append(feature_name)
    
    print(str(len(to_drop)) + " columns dropped")

    return df.drop(columns=to_drop)

def parse_date(df: pandas.DataFrame, one_hot_encoded = False):
    # Date Format: 27.01.2023 05:00:56
    df["month"] = pandas.Series(dtype=int)
    df["hour"] = pandas.Series(dtype=int)
    for index, row in df.iterrows():
        time_stamp = row['DATE']
        parsed_ts = time_stamp.split(" ")
        date = parsed_ts[0]
        time = parsed_ts[1]
        parsed_date = date.split(".")
        M = parsed_date[1]
        hour = time.split(":")[0]
        df.at[index, "month"] = M
        df.at[index, "hour"] = hour
    if not one_hot_encoded:
        return df
    df = pandas.get_dummies(df, columns = ['month', 'hour']) 
    return df


def introduce_nans(df: pandas.DataFrame, row_ratio: float, ratio: float, border: float=None):
    # Add NaN values to dataframe
    # Ratio_rows: Ratio of rows that will be nanned
    # Ratio: Ratio of the number of nans that will be added to each row
    # Borders: The variance between min and max number of nan for each row

    if border is None:
        border = 0
    for index, row in df.iterrows():
        if random.random() < row_ratio:
            no_of_nans = int((border + ratio) * df.shape[1])
            nan_locs = random.sample(range(0, df.shape[1]), no_of_nans)
            for nan_loc in nan_locs:
                df.at[index, df.columns[nan_loc]] = numpy.nan
    return df

def report_completeness(df: pandas.DataFrame, detail = False):
    comp_row, compp_row_perc = completeness_data_point(df)
    if detail:
        print("Overall completeness: " + str(completeness_overall(df)))
        comps = completeness_feature(df)

        for d_key in comps:
            print(d_key + " - " + str(comps[d_key]))

        print("COMP ROW COUNT: " + str(comp_row))
        print("COMP ROW PERC%: " + str(compp_row_perc))
    else: 
        
        print("Overall: " + str(completeness_overall(df)) + " - COMP ROW: " + str(comp_row) + " - COMP ROW PERC%: " + str(compp_row_perc))


def rmse_report(df_real: pandas.DataFrame, df_imputed: pandas.DataFrame):
    for col in df_real:
        print("lol")

def sub_dataframe(df: pandas.DataFrame, pollutant:str=None, station:str=None, data_completeness: float=None, year: int=None):
    if pollutant is None and station is None and data_completeness is None and year is None:
        print("No selection criteria!")
        return df
    
    neo_df = copy.deepcopy(df)

    # Filter Pollutants
    if pollutant is not None:
        for column_name in neo_df.columns:
            if pollutant not in column_name and column_name != 'DATE':
                neo_df = neo_df.drop([column_name], axis=1)

    # Filter Stations
    if station is not None:
        for column_name in neo_df.columns:
            if str(station + "_") not in column_name and column_name != 'DATE':
                neo_df = neo_df.drop([column_name], axis=1)

    # Filter Data Completeness
    if data_completeness is not None:
        for column_name in neo_df.columns:
            if completeness_feature(neo_df[column_name]) < data_completeness and column_name != 'DATE':
                neo_df = neo_df.drop([column_name], axis=1)

    # Filter Year
    if year is not None:
        neo_df = neo_df[neo_df['DATE'].str.contains(str(year))]
        
    return neo_df

def AQ(df:pandas.DataFrame):
    for col in df.columns:
        print(col)

# Air Quality 

pols = ["NO2", "PM10", "O3", "PM25"]
pol_levels = {"NO2": [50, 100, 200, 400], "PM10": [25, 50, 90, 180], "O3": [60, 120, 180, 240], "PM25": [15, 30, 55, 110]}

def polcon2AQI(pol_name: str, val:float):
    global pols
    registered_pollutant = False
    pol_id = None
    for pol in pols:
        if pol in pol_name:
            registered_pollutant = True
            pol_id = pol
    if not registered_pollutant:
        return 0
    if val < pol_levels[pol_id][0]:
        return 1
    elif val < pol_levels[pol_id][1]:
        return 2
    elif val < pol_levels[pol_id][2]:
        return 3
    elif val < pol_levels[pol_id][3]:
        return 4
    return 5

def generate_class_label(row: pandas.Series):
    vals = []
    for key in row.keys():
        vals.append(polcon2AQI(pol_name=key, val=row[key]))
    return max(vals)

def generate_class_labels(df: pandas.DataFrame):
    AQIs = []
    for index, row in df.iterrows():
        AQIs.append(generate_class_label(row=dict(row)))
    AQIs.append(0)
    df["OUT"] = AQIs[1:]
    return df

# Classification

def test_data_classification_(df:pandas.DataFrame): # Yedek Çalışan Version
    data_out = df["OUT"].to_numpy()
    data_in = df.iloc[:, :-1].to_numpy()

    data_in_train, data_in_test, data_out_train, data_out_test = train_test_split(data_in, data_out, test_size=0.2, random_state=42)

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(data_in_train, data_out_train)
    data_out_pred = clf.predict(data_in_test)

    return accuracy_score(data_out_test, data_out_pred)


from sklearn.metrics import accuracy_score, f1_score


def test_data_classification(df:pandas.DataFrame, df_org:pandas.DataFrame, k:int = 5, data_out=False):
    # df        -> Imputed Dataset
    # df_org    -> Orginal Dataset
    # k         -> Cross Fold K

    if data_out:
        _data_out(df, df_org)

    test_size = 1.0 / k

    accuracies = []

    for cv_iter in range(k):
        imp_data_out = df["OUT"].to_numpy()
        imp_data_in = df.iloc[:, :-1].to_numpy()    
        org_data_out = df_org["OUT"].to_numpy()
        org_data_in = df_org.iloc[:, :-1].to_numpy()    

        test_indexs = [int((cv_iter * test_size) * df.shape[0]), int(((cv_iter + 1) * test_size) * df.shape[0])]
        data_in_test = org_data_in[test_indexs[0]: test_indexs[1],:]
        data_out_test = org_data_out[test_indexs[0]: test_indexs[1]]
        data_in_train = numpy.delete(imp_data_in, range(test_indexs[0], test_indexs[1]), 0)
        data_out_train = numpy.delete(imp_data_out, range(test_indexs[0], test_indexs[1]), 0)

        clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        clf.fit(data_in_train, data_out_train)
        data_out_pred = clf.predict(data_in_test)

        #print("pred:")
        #print(data_out_pred)
        #print("real:")
        #print(data_out_test)
        #print("---------------------------------------------------")

        accuracies.append(accuracy_score(data_out_test, data_out_pred))

    return sum(accuracies) / len(accuracies)


from sklearn.svm import SVR
from sklearn.metrics import r2_score


def test_data_regression(df:pandas.DataFrame, df_org:pandas.DataFrame, k:int = 5, kernel:str = "linear", data_out=False):
    # df        -> Imputed Dataset
    # df_org    -> Orginal Dataset
    # k         -> Cross Fold K

    if data_out:
        _data_out(df, df_org)

    test_size = 1.0 / k

    accuracies = []

    for cv_iter in range(k):
        imp_data_out = df["OUT"].to_numpy()
        imp_data_in = df.iloc[:, :-1].to_numpy()    
        org_data_out = df_org["OUT"].to_numpy()
        org_data_in = df_org.iloc[:, :-1].to_numpy()    

        test_indexs = [int((cv_iter * test_size) * df.shape[0]), int(((cv_iter + 1) * test_size) * df.shape[0])]
        data_in_test = org_data_in[test_indexs[0]: test_indexs[1],:]
        data_out_test = org_data_out[test_indexs[0]: test_indexs[1]]
        data_in_train = numpy.delete(imp_data_in, range(test_indexs[0], test_indexs[1]), 0)
        data_out_train = numpy.delete(imp_data_out, range(test_indexs[0], test_indexs[1]), 0)

        regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
        regr.fit(data_in_train, data_out_train)
        data_out_pred = regr.predict(data_in_test)

        #print("pred:")
        #print(data_out_pred)
        #print("real:")
        #print(data_out_test)
        #print("---------------------------------------------------")

        accuracies.append(r2_score(data_out_test, data_out_pred))

    return sum(accuracies) / len(accuracies)

_data_out_counter = 0

def _data_out(df:pandas.DataFrame, df_org:pandas.DataFrame):
    global _data_out_counter
    df.to_csv("temp/nanned" + str(_data_out_counter)+".csv")
    df_org.to_csv("temp/org" + str(_data_out_counter)+".csv")