import os
import kagglehub
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_experiment("Hotel Booking Demand - CI")

# Load dataset (runtime)
path = kagglehub.dataset_download("jessemostipak/hotel-booking-demand")
df = pd.read_csv(os.path.join(path, "hotel_bookings.csv"))

# Basic preprocessing
df["children"] = df["children"].fillna(0)
df["agent"] = df["agent"].fillna(0)
df["company"] = df["company"].fillna(0)
df["country"] = df["country"].fillna("Unknown")
df = df.drop_duplicates()

X = df.drop(columns=["is_canceled"])
y = df["is_canceled"]

# Pisahkan kolom
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
with mlflow.start_run(run_name="ci_training"):

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 20)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # Save model
    mlflow.sklearn.log_model(sk_model=clf, name="model")

    # Confusion matrix artifact
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")

    os.makedirs("artifacts", exist_ok=True)
    cm_path = "artifacts/cm.png"
    plt.savefig(cm_path)
    plt.close()

    mlflow.log_artifact(cm_path)

    print(f"CI Training finished | accuracy={acc:.4f}, f1={f1:.4f}")
