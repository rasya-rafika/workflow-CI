import os
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default="../adult_preprocessing.csv")
    p.add_argument("--target_col", type=str, default="income")
    return p.parse_args()

def main():
    args = parse_args()

    # =============================
    # 0) BIKIN TRACKING NYATU
    # =============================
    script_dir = Path(__file__).resolve().parent          # .../workflow-CI/MLProject
    project_root = script_dir.parent.parent               # .../SMSML_Rasya Rafika Widalala
    shared_mlruns = project_root / "membangun_model" / "mlruns"

    # kalau folder belum ada, bikin (biar aman)
    shared_mlruns.mkdir(parents=True, exist_ok=True)

    # set tracking uri ke MLRUNS YANG SAMA
    mlflow.set_tracking_uri(shared_mlruns.as_uri())

    # samain experiment biar masuk list yang sama
    mlflow.set_experiment("Adult Income Classification")

    # WAJIB: autolog
    mlflow.autolog(log_models=True)

    # =============================
    # 1) Load dataset hasil preprocessing
    # =============================
    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = (script_dir / data_path).resolve()    # relatif dari MLProject/

    if not data_path.exists():
        raise FileNotFoundError(f"Data tidak ditemukan: {data_path}")

    df = pd.read_csv(data_path)

    if args.target_col not in df.columns:
        raise ValueError(
            f"Kolom target '{args.target_col}' tidak ada. Kolom tersedia: {df.columns.tolist()}"
        )

    # =============================
    # 2) Split X/y
    # =============================
    X = df.drop(columns=[args.target_col])
    y_raw = df[args.target_col].astype(str).str.strip()
    y = (y_raw == ">50K").astype(int)

    # Validasi: modelling.py tidak preprocessing
    non_numeric = X.select_dtypes(include=["object", "bool"]).columns.tolist()
    if len(non_numeric) > 0:
        raise ValueError(
            "modelling.py tidak boleh melakukan preprocessing.\n"
            f"Tapi dataset kamu masih punya fitur non-numerik: {non_numeric}\n"
            "Solusi: lakukan encoding di preprocessing lalu simpan ulang adult_preprocessing.csv."
        )

    split_test_size = 0.2
    split_random_state = 42

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_test_size, random_state=split_random_state, stratify=y
    )

    # =============================
    # 3) Train + log
    # =============================
    with mlflow.start_run(run_name="ci-retrain"):
        model = LogisticRegression(max_iter=2500, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # metrics 
        metrics = {
            "accuracy_manual": float(accuracy_score(y_test, y_pred)),
            "precision_manual": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall_manual": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_manual": float(f1_score(y_test, y_pred, zero_division=0)),
        }
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # param split
        mlflow.log_param("split_test_size", split_test_size)
        mlflow.log_param("split_random_state", split_random_state)

        # ===== minimal 2 artifacts manual =====
        # Artifact 1: confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure()
        plt.imshow(cm)
        plt.title("Confusion Matrix (CI)")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.tight_layout()
        cm_path = script_dir / "confusion_matrix_ci.png"
        plt.savefig(cm_path, dpi=150)
        plt.close()
        mlflow.log_artifact(str(cm_path))

        # Artifact 2: classification report
        report_path = script_dir / "classification_report_ci.txt"
        report_path.write_text(classification_report(y_test, y_pred, digits=4), encoding="utf-8")
        mlflow.log_artifact(str(report_path))

        # Artifact 3: metrics json (opsional tapi bagus)
        metrics_path = script_dir / "metrics_ci.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(metrics_path))

        print("MLflow UI")
        print(f"Shared mlruns: {shared_mlruns}")

if __name__ == "__main__":
    main()
