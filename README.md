# End‑to‑End Process Mining & Predictive Risk (BPI 2012) — Azure ML + Power BI

**TL;DR.** A compact, reproducible template that mines process bottlenecks and predicts late‑completion risk from an ERP‑style event log.  
Built to mirror the **BIS Data & AI Engineer** remit: data engineering, process mining, Azure ML deployment, and Power BI reporting.

---

## Why this project matters (role alignment)

- **Process analysis & optimisation:** PM4Py EDA (variants, DFG, bottlenecks) from a real event log (BPI 2012).  
- **AI & automation:** Baseline XGBoost model trained and exposed via **Azure ML Managed Online Endpoint** (REST inference).  
- **Tooling & dashboards:** **3‑page Power BI** report (web) showing process health, bottlenecks, and a prediction lens with a **threshold slider** and **confusion‑matrix metrics** (Precision/Recall/F1).  
- **Collaboration & documentation:** Public, reproducible repo with step‑by‑step notebooks/scripts and lightweight architecture.

---

## What’s included

- **Notebooks**  
  `01_prepare_bpi2012.ipynb` — clean, features, label (q75 late)  
  `02_process_mining_eda.ipynb` — process map + stats  
  `03_train_register_deploy.ipynb` — train → register → deploy (Azure ML)  
  `04_batch_scoring_and_export.ipynb` — batch scoring → CSVs for Power BI

- **Power BI (web, mac‑friendly)**  
  Pages: **Process Health**, **Bottleneck Map**, **Prediction Lens**  
  Data model from `cases.csv`, `events.csv`, `edges.csv` (exported by notebook 04).  
  DAX includes a **what‑if Threshold** parameter and **Precision/Recall/F1** measures.

- **Inference**  
  Azure ML **Managed Online Endpoint** with key auth and JSON payloads.

---

## Live assets (replace with your links)

- **Report (.pbix):** `[link to PBIX download]`  
- **Embedded report (iframe):**
  ```html
  <iframe
    width="100%" height="720"
    src="[YOUR_POWER_BI_EMBED_URL]"
    frameborder="0" allowFullScreen="true">
  </iframe>
  ```
- **Endpoint (REST, optional):** `https://bpi2012-risk-endpoint.koreacentral.inference.ml.azure.com/score`  
- **Repo:** `https://github.com/bwade9090/bis-bpi2012-azureml-mvp.git`

---

## Quick start (reproduce in ~15–30 min)

1) **Install**
```bash
pip install -r requirements.txt
```

2) **Data**  
Place `BPI_Challenge_2012.xes` in `data/`.

3) **Run Day 1**
- 01 → 02 → 03 (set `AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP`, `AZURE_ML_WORKSPACE` to register/deploy).

4) **Run Day 2**
- 04 to export Power BI data: `artifacts/powerbi/{cases,events,edges}.csv`.

5) **Power BI (web)**  
Create a semantic model from `cases.csv`, then add `events.csv` and `edges.csv` to the same model.  
Set relationship `Cases[case_id] 1—* Events[case_id]`.  
Build the 3 pages using the provided DAX (threshold slider + F1/Precision/Recall).

---

## Modeling summary

- **Label:** late case if total cycle time > **75th percentile** (configurable in code).  
- **Features (case‑level):** counts (events/resources/activities), inter‑event stats, weekend/working‑hour flags, etc.  
- **Model:** XGBoost baseline; probabilities exposed as `risk_score` (used in Power BI).  
- **Evaluation:** F1 / PR‑AUC reported in notebook 03; confusion matrix recomputed in Power BI via DAX as the **threshold changes**.  
- **Batch scoring:** notebook 04 supports **local model** or **endpoint** mode and writes CSVs for BI.

> Note: Azure Monitor/Model Monitor is intentionally **out of scope for this MVP**; left as a next‑step.

---

## Dashboard pages (web)

- **Process Health:** KPIs (Total Cases, Late Rate, Avg/P95 Cycle Time), distributions, slice by resources/risk bins.  
- **Bottleneck Map:** Edges matrix (from→to with counts/avg gap) or a process‑mining custom visual.  
- **Prediction Lens:** Threshold slider, **Precision/Recall/F1** cards, and a per‑case table (`risk_score`, `y_late`, key drivers).

---

## Architecture (thin slice)

```
BPI2012 (XES/CSV)
      └── 01 Prepare → features + label
            └── 02 EDA (PM4Py)
                  └── 03 Train→Register→Deploy (Azure ML endpoint)
                        └── 04 Batch Score → cases/events/edges CSV
                              └── Power BI (web) dashboards
```

---

## What I contributed / skills demonstrated

- Designed and shipped an **end‑to‑end** pipeline (data → EDA → ML → serving → BI) in a constrained timeframe.  
- Hands‑on with **Azure ML (SDK v2)**: model registration, **Managed Online Endpoint**, key‑auth REST.  
- Built a **browser‑only** Power BI workflow on macOS (semantic model, relationships, DAX, embedded report).  
- Process‑mining analysis with **PM4Py**, translating insights into **actionable BI**.

---

## Next steps (if extended)
 
- Automate Power BI refresh via OneDrive/SharePoint or Dataflow; optional Fabric pipelines.  
- Revisit label (SLA‑based), expand features, and calibrate threshold for business targets.

---

## Data & license

- Dataset: **BPI Challenge 2012** (public process‑mining benchmark).  
- Code: **MIT License**. See `LICENSE`.

---

### Contact

If you’d like a quick walkthrough or access to the endpoint/report, I’m happy to provide a short demo and discuss how this template can be adapted to BIS corporate processes.
