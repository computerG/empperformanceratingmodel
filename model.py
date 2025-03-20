import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scipy.stats.mstats import winsorize
from imblearn.over_sampling import (
    SMOTE,
)  # SMOTE(synthetic minority oversampling techinque)
import joblib

import os
import logging

# Define a path to save the trained model and encoders
MODEL_PATH = "randomforest_model.pkl"
ENCODER_PATH = "performace_rating_encoder.pkl"
logging.basicConfig(level=logging.INFO)
# List of categorical columns explicitly provided
cat_column = [
    "EmpEducationLevel",
    "EmpEnvironmentSatisfaction",
    "EmpJobInvolvement",
    "EmpJobSatisfaction",
    "PerformanceRating",
    "EmpRelationshipSatisfaction",
    "EmpWorkLifeBalance",
]

# Automatically identify numerical columns

def load_data():
    df_employee = pd.read_excel(
        "data/INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xls"
    )
    logging.info("Data loaded successfully.")
    return df_employee  # `df_employee` is a DataFrame variable that is used to store the data loaded from a CSV
    # file named 'INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xls'. It is used
    # within the functions defined in the code snippet for data processing, encoding, and
    # training the machine learning model.


df_employee = load_data()

numerical_columns = [
    col
    for col in df_employee.select_dtypes(include=["int64", "float64"]).columns
    if col not in cat_column
]

# Identify categorical columns
categorical_columns = (
    list(df_employee.select_dtypes(include=["object"]).columns) + cat_column
)

# Initialize encoders
severity_encoder = LabelEncoder()

def remove_outliers(dataframe, columns):
    for column in columns:
        winsorized_data = winsorize(dataframe[column], (0, 0.06))
        dataframe[column] = winsorized_data
    return dataframe

def preprocess_data(df):
    if "EmpNumber" in df.columns:
        df.drop("EmpNumber", axis=1)
    df = remove_outliers(df, numerical_columns)
    return df


def encode_data(df):
    df = preprocess_data(df)
    label_encoders = {col: LabelEncoder() for col in categorical_columns}
    for col in categorical_columns:
        df[col] = label_encoders[col].fit_transform(df[col])
    if "PerformanceRating" in df.columns:
        df["PerformanceRating"] = severity_encoder.fit_transform(
            df["PerformanceRating"]
        )
    logging.info("Data encoded successfully.")
    return df


def train_model():
    # check the train model and train the model
    df = load_data()
    df = encode_data(df)
    X = df.drop(columns=["PerformanceRating"], axis=1)
    y = df["PerformanceRating"]
    sm = SMOTE()  # obeject creation
    X, y = sm.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(severity_encoder, ENCODER_PATH)
    logging.info("Model and encoder saved successfully.")

    y_pred = model.predict(X_test)
    unique_classes = sorted(list(set(y_test) | set(y_pred)))
    target_names = [
        severity_encoder.classes_[i]
        for i in unique_classes
        if i < len(severity_encoder.classes_)
    ]
    print(f"Target names {target_names}")
    """ report = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        labels=unique_classes,
        zero_division=1,
    )
    print(report) """

    return model


def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        return train_model()
    model = joblib.load(MODEL_PATH)
    global severity_encoder
    severity_encoder = joblib.load(ENCODER_PATH)
    logging.info("Model and encoder loaded successfully.")
    print(severity_encoder)
    return model


def predict_model(model, X):
    prediction = model.predict(X)

    return severity_encoder.inverse_transform(prediction)
