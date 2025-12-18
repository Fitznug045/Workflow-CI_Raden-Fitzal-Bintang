import pandas as pd
import kagglehub
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def main():
    # 1. Download Dataset
    path = kagglehub.dataset_download("jessemostipak/hotel-booking-demand")
    csv_path = os.path.join(path, "hotel_bookings.csv")
    df = pd.read_csv(csv_path)

    # 2. Preprocessing Ringan
    df = df.drop_duplicates()
    df["children"] = df["children"].fillna(0)
    df["agent"] = df["agent"].fillna(0)
    df["company"] = df["company"].fillna(0)
    df["country"] = df["country"].fillna("Unknown")

    X = df.drop(columns=["is_canceled"])
    y = df["is_canceled"]

    # 3. Column Split
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns
    
    # 4. Preprocessing
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # 5. Model Pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # 6. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 7. Training + MLflow
    with mlflow.start_run(nested=True):
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, "model")

        print(f"Training selesai | accuracy={acc:.4f}, f1={f1:.4f}")


if __name__ == "__main__":
    main()
