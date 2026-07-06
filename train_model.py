import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder

# =====================================
# LOAD DATASETS
# =====================================

print("Loading datasets...")

app_df = pd.read_csv("dataset/application_record.csv")
credit_df = pd.read_csv("dataset/credit_record.csv")

print("Application Dataset Shape :", app_df.shape)
print("Credit Dataset Shape :", credit_df.shape)

# =====================================
# CREATE TARGET COLUMN
# =====================================

credit_df["TARGET"] = credit_df["STATUS"].apply(
    lambda x: 0 if x in ["X", "C", "0"] else 1
)

credit_target = credit_df.groupby("ID")["TARGET"].max().reset_index()

print("\nTarget Counts:")
print(credit_target["TARGET"].value_counts())

# =====================================
# MERGE DATASETS
# =====================================

df = pd.merge(app_df, credit_target, on="ID", how="inner")

print("\nMerged Dataset Shape:", df.shape)

# =====================================
# HANDLE MISSING VALUES
# =====================================

print("\nMissing Values Before Filling:")
print(df.isnull().sum())

df["OCCUPATION_TYPE"] = df["OCCUPATION_TYPE"].fillna("Unknown")

print("\nMissing Values After Filling:")
print(df.isnull().sum())

# =====================================
# REMOVE DUPLICATES
# =====================================

print("\nShape Before Removing Duplicates:", df.shape)

df = df.drop_duplicates()

print("Shape After Removing Duplicates:", df.shape)

# =====================================
# CONVERT STRING COLUMNS TO OBJECT
# =====================================

string_columns = [
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE"
]

for col in string_columns:
    df[col] = df[col].astype("object")

# =====================================
# LABEL ENCODING
# =====================================

encoders = {}

for col in string_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

print("\nDataset After Encoding:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nFirst Five Rows:")
print(df.head())
# =====================================
# SPLIT FEATURES AND TARGET
# =====================================

from sklearn.model_selection import train_test_split

X = df.drop("TARGET", axis=1)
y = df["TARGET"]

print("\nFeature Shape :", X.shape)
print("Target Shape :", y.shape)

# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape :", X_train.shape)
print("Testing Shape :", X_test.shape)

# =====================================
# FEATURE SCALING
# =====================================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFeature Scaling Completed")

# =====================================
# LOGISTIC REGRESSION
# =====================================

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
print("Logistic Regression :", accuracy_score(y_test, y_pred))

print("\n==============================")
print("LOGISTIC REGRESSION")
print("==============================")

print("Accuracy :", accuracy_score(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))
# =====================================
# DECISION TREE
# =====================================

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)

print("\n==============================")
print("DECISION TREE")
print("==============================")
print("Accuracy :", dt_accuracy)

# =====================================
# RANDOM FOREST
# =====================================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("\n==============================")
print("RANDOM FOREST")
print("==============================")
print("Accuracy :", rf_accuracy)

# =====================================
# XGBOOST
# =====================================

xgb = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)

xgb_accuracy = accuracy_score(y_test, xgb_pred)

print("\n==============================")
print("XGBOOST")
print("==============================")
print("Accuracy :", xgb_accuracy)

# =====================================
# MODEL COMPARISON
# =====================================

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print("Logistic Regression :", accuracy_score(y_test, y_pred))
print("Decision Tree       :", dt_accuracy)
print("Random Forest       :", rf_accuracy)
print("XGBoost             :", xgb_accuracy)
# =====================================
# SAVE BEST MODEL
# =====================================

import joblib

joblib.dump(xgb, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nBest Model Saved Successfully!")
print("Saved as : models/model.pkl")
print("Scaler Saved as : models/scaler.pkl")