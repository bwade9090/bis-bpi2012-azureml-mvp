import os
import json
import joblib
import pandas as pd

model = None
feature_names = None
threshold = 0.5

def init():
    global model, feature_names, threshold
    model_dir = os.getenv("AZUREML_MODEL_DIR", ".")
    model_path = os.path.join(model_dir, "model.joblib")
    feat_path = os.path.join(model_dir, "feature_names.json")

    model = joblib.load(model_path)
    if os.path.exists(feat_path):
        with open(feat_path, "r") as f:
            feature_names = json.load(f)
    th = os.getenv("THRESHOLD")
    if th:
        try:
            threshold = float(th)
        except Exception:
            pass

def run(raw_data):
    try:
        data = json.loads(raw_data)
        if isinstance(data, dict):
            records = [data]
        elif isinstance(data, list):
            records = data
        else:
            return json.dumps({"error": "Unsupported payload type"})

        df = pd.DataFrame.from_records(records)

        # Ensure feature order
        if feature_names:
            for c in feature_names:
                if c not in df.columns:
                    df[c] = 0
            df = df[feature_names]

        proba = model.predict_proba(df)[:, 1].tolist()
        pred = [1 if p >= threshold else 0 for p in proba]
        return json.dumps({"prob": proba, "pred": pred})
    except Exception as e:
        return json.dumps({"error": str(e)})