import pandas as pd
import numpy as np
import datetime
import math
from random import randrange
import numpy as np

''' This function takes the data to be preprocessed and returns a deep copy of the preprocessed data.'''
def preprare_data(data, training_data):
    tempData = data.copy(deep=True)
    tempTrain = training_data.copy(deep=True)

    # converting all unique blood_types/symptoms to OHE features and removing blood_type feature
    encoded_blood_types = pd.get_dummies(tempTrain['blood_type'])
    encoded_symptoms = tempTrain['symptoms'].str.get_dummies(sep=';')
    tempTrain = pd.concat([tempTrain.drop(['blood_type', 'symptoms'], axis=1), encoded_blood_types, encoded_symptoms],
                          axis=1)

    encoded_blood_types = pd.get_dummies(tempData['blood_type'])
    encoded_symptoms = tempData['symptoms'].str.get_dummies(sep=';')
    tempData = pd.concat([tempData.drop(['blood_type', 'symptoms'], axis=1), encoded_blood_types, encoded_symptoms],
                         axis=1)

    tempTrain = cleanDates(tempTrain)
    tempData = cleanDates(tempData)

    tempTrain.drop(columns=[
        "patient_id", "conversations_per_day", "B-", "AB-",
        "AB+", "address", "low_appetite", "headache", "PCR_06",
        "sex", "B+", "O-", "pcr_date", "happiness_score",
        "current_location", "job"], inplace=True)

    tempData.drop(columns=[
        "patient_id", "conversations_per_day", "B-",
        "AB-", "AB+", "address", "low_appetite", "headache", "PCR_06",
        "sex", "B+", "O-", "pcr_date", "happiness_score",
        "current_location", "job"], inplace=True)

    tempTrain = cleanOutliers(tempTrain)
    tempTrain = commitImputation(tempTrain, tempTrain)

    tempData = cleanOutliers(tempData)
    tempData = commitImputation(tempData, training_data)

    tempData["risk"].replace("Low", np.float64(0), inplace=True)
    tempData["risk"].replace("High", np.float64(1), inplace=True)
    tempData["spread"].replace("High", np.float64(1), inplace=True)
    tempData["spread"].replace("Low", np.float64(0), inplace=True)
    tempData["covid"] = tempData["covid"].astype(int)

    return tempData

def cleanOutliers(df):
    # Replaces all negative values in the discrete_features columns of df with nan
    discrete_features = ["sugar_levels", "sport_activity", "weight", "num_of_siblings", ]
    df[discrete_features] = df[discrete_features].where(df[discrete_features] >= 0, np.nan)

    # Setting weight outliers for each age group under 10
    df.loc[df["age"] <= 9, "weight"] = df.loc[df["age"] <= 9, "weight"].apply(lambda x: 0.65 * x if x > 5 else 0.3 * x)

    # Outlier pairs cleaning implementation, graphs printing are for self-analysis
    relations = [["age", ['weight', 'sugar_levels'], 1.5], ['sugar_levels', ['weight'], 1.5]]

    for relation in relations:
        feature = relation[0]
        related_features = relation[1]
        threshold = relation[2]

        for related_feature in related_features:
            q1 = df[feature].quantile(0.25)
            q3 = df[feature].quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr

            df.loc[(df[feature] < lower_bound) | (df[feature] > upper_bound), related_feature] = np.nan
    return df


def cleanDates(df):
    mask = (df.pcr_date >= '2020-05-21') & (df.pcr_date <= '2020-06-08')
    indexes_to_remove = df.index[mask]
    df.loc[indexes_to_remove, ['PCR_01', 'PCR_02', 'PCR_03',
                               'PCR_04', 'PCR_05', 'PCR_06', 'PCR_07', 'PCR_08', 'PCR_09', 'PCR_10']] = np.nan
    return df


def imputePerAgeGroup(df1, df2, string):
    df1[string].fillna(df2.groupby("age")[string].transform("median"), inplace=True)

    # Replace remaining NaN values with overall median
    overall_median = np.nanmedian(df1[string])
    df1[string].fillna(overall_median, inplace=True)
    return df1


def imputeMedian(df1, df2, string):
    df1[string].fillna(df2[string].median(), inplace=True)
    return df1


def commitImputation(df1, df2):
    # Replace all nan values with random num in (0,9)
    df1["num_of_siblings"] = df1["num_of_siblings"].fillna(randrange(0, 9))
    df1["age"].fillna(df2["age"].median(), inplace=True)
    df1 = imputePerAgeGroup(df1, df2, "sugar_levels")
    df1 = imputePerAgeGroup(df1, df2, "weight")
    df1 = imputeMedian(df1, df2, "household_income")
    df1 = imputeMedian(df1, df2, "sport_activity")

    # missingImputation of All pcr's  by median sample method
    PCRS = ['PCR_01', 'PCR_02', 'PCR_03', 'PCR_04', 'PCR_05',
            'PCR_07', 'PCR_08', 'PCR_09', 'PCR_10']
    for i in PCRS:
        df1[i].fillna(df1[i].median(), inplace=True)

    return df1


