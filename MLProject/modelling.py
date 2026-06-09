import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)
import os

# Setup MLflow - ambil dari environment variable jika ada
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000/")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Titanic-Classification-CI")

# Load data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

X_train = train_df.drop(columns=["Survived"])
y_train = train_df["Survived"]
X_test = test_df.drop(columns=["Survived"])
y_test = test_df["Survived"]

# Training dengan autolog
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="RandomForest-CI"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC  : {roc_auc_score(y_test, y_proba):.4f}")

    # Simpan run_id untuk diambil workflow
    run_id = mlflow.last_active_run().info.run_id
    print(f"MLflow Run ID: {run_id}")

print("Training selesai!")
