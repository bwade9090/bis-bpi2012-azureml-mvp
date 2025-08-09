#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
03 — Train → (Optional) Register → (Optional) Deploy on Azure ML

Steps:
1) Read labeled features: artifacts/processed_case_features_labeled.csv
2) Train XGBoost baseline, print F1 / PR-AUC
3) Save model + feature names under artifacts/model/
4) (Optional) Register to Azure ML (requires env: AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_ML_WORKSPACE)
5) (Optional) Create/Update Managed Online Endpoint + Deployment

Usage examples:
  # Train only
  python src/03_train_register_deploy.py

  # Train + Register
  python src/03_train_register_deploy.py --register

  # Train + Register + Deploy (blue)
  python src/03_train_register_deploy.py --register --deploy --endpoint-name bpi2012-risk-endpoint
"""

import os
import json
import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, average_precision_score, classification_report
from xgboost import XGBClassifier


def train_and_save(labeled_path: str, model_dir: str, threshold: float = 0.5):
    os.makedirs(model_dir, exist_ok=True)
    df = pd.read_csv(labeled_path)

    target = "label_late"
    X = df.drop(columns=[target, "case_id"], errors="ignore")
    y = df[target].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=0
    )
    clf.fit(X_tr, y_tr)

    proba = clf.predict_proba(X_te)[:, 1]
    pred = (proba >= threshold).astype(int)

    f1 = f1_score(y_te, pred)
    pr_auc = average_precision_score(y_te, proba)
    print("[METRIC]", {"F1": round(float(f1), 4), "PR_AUC": round(float(pr_auc), 4)})
    print(classification_report(y_te, pred))

    # Save model + feature names
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(clf, model_path)
    with open(os.path.join(model_dir, "feature_names.json"), "w") as f:
        json.dump(list(X.columns), f)

    print(f"[OK] Saved model -> {model_path}")
    return model_path


def register_model(model_dir: str, model_name: str):
    print("[INFO] Registering model to Azure ML...")
    from azure.identity import DefaultAzureCredential
    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import Model

    sub = os.getenv("AZURE_SUBSCRIPTION_ID")
    rg = os.getenv("AZURE_RESOURCE_GROUP")
    ws = os.getenv("AZURE_ML_WORKSPACE")

    if not all([sub, rg, ws]):
        raise RuntimeError(
            "Missing Azure env vars. Set AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_ML_WORKSPACE."
        )

    cred = DefaultAzureCredential(exclude_interactive_browser_credential=False)
    ml_client = MLClient(cred, sub, rg, ws)

    registered = ml_client.models.create_or_update(
        Model(
            name=model_name,
            version=None,
            path=model_dir,  # folder with model.joblib + feature_names.json
            description="Baseline XGBoost classifier for BPI2012 late completion risk"
        )
    )
    print(f"[OK] Registered model: {registered.name} v{registered.version}")
    return registered


def deploy_endpoint(model_id: str, endpoint_name: str):
    print(f"[INFO] Deploying Managed Online Endpoint '{endpoint_name}'...")
    from azure.identity import DefaultAzureCredential
    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, Environment, CodeConfiguration

    sub = os.getenv("AZURE_SUBSCRIPTION_ID")
    rg = os.getenv("AZURE_RESOURCE_GROUP")
    ws = os.getenv("AZURE_ML_WORKSPACE")

    cred = DefaultAzureCredential(exclude_interactive_browser_credential=False)
    ml_client = MLClient(cred, sub, rg, ws)

    # Create endpoint (idempotent)
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        auth_mode="key",
        tags={"project": "bpi2012", "stage": "mvp"}
    )
    try:
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        print("[OK] Endpoint ready:", endpoint.name)
    except Exception as e:
        print("[WARN] Endpoint create/update:", e)

    # Environment
    env = Environment(
        name="bpi2012-aml-env",
        conda_file="infra/environment-conda.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:20240516.v1",
    )

    # Deployment (blue)
    deployment = ManagedOnlineDeployment(
        name="blue",
        endpoint_name=endpoint_name,
        model=model_id,
        environment=env,
        code_configuration=CodeConfiguration(code="infra/inference", scoring_script="score.py"),
        instance_type="Standard_DS2_v2",
        instance_count=1
    )
    poller = ml_client.online_deployments.begin_create_or_update(deployment)
    result = poller.result()
    print("[OK] Deployment state:", result.provisioning_state)

    # Route traffic
    ep = ml_client.online_endpoints.get(endpoint_name)
    ep.traffic = {"blue": 100}
    ml_client.online_endpoints.begin_create_or_update(ep).result()
    print("[OK] Traffic routed to 'blue'")

    ep = ml_client.online_endpoints.get(endpoint_name)
    print("[INFO] Scoring URI:", ep.scoring_uri)
    print("[NOTE] Fetch key via CLI: az ml online-endpoint get-credentials -n", endpoint_name)


def parse_args():
    p = argparse.ArgumentParser(description="Train baseline and optionally register/deploy on Azure ML.")
    p.add_argument("--labeled-path", default="artifacts/processed_case_features_labeled.csv")
    p.add_argument("--model-dir", default="artifacts/model")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--model-name", default="bpi2012_late_risk_xgb")
    p.add_argument("--endpoint-name", default="bpi2012-risk-endpoint")
    p.add_argument("--register", action="store_true", help="Register model to Azure ML")
    p.add_argument("--deploy", action="store_true", help="Deploy Managed Online Endpoint (requires --register or existing model)")
    return p.parse_args()


def main():
    args = parse_args()
    model_path = train_and_save(args.labeled_path, args.model_dir, args.threshold)

    if not args.register and not args.deploy:
        return

    registered = None
    if args.register:
        registered = register_model(args.model_dir, args.model_name)

    if args.deploy:
        # Use the just-registered model if available; otherwise assume user passes an AML model id
        if registered is None:
            raise RuntimeError("Deploy requires a registered model. Run with --register first.")
        deploy_endpoint(registered.id, args.endpoint_name)


if __name__ == "__main__":
    main()