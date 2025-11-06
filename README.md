# Health Dashboard – Heart Disease Risk Prediction

An AI-based web app that predicts heart disease risk using the UCI Heart Disease dataset. It includes data preprocessing, model training, saved artifacts, a Streamlit UI, and a Vercel-ready FastAPI backend for serverless deployments. The app reports prediction probability, risk category, global feature importances, optional SHAP explanations, and test-set metrics (ROC AUC, Accuracy, etc.).

## Tech Stack
- Python 3.8+
- pandas, numpy, scikit-learn, joblib
- streamlit
- shap (optional in app)

## Project Structure
```
health-dashboard/
├─ data/
│  └─ heart.csv            # place dataset here
├─ notebooks/
│  └─ 01_eda_modeling.ipynb
├─ src/
│  ├─ preprocess.py
│  ├─ train.py
│  └─ utils.py
├─ api/                     # FastAPI serverless API for Vercel
│  ├─ index.py
│  └─ _shared.py
├─ app/
│  └─ streamlit_app.py
├─ model/
│  ├─ model.joblib         # produced by train.py
│  ├─ scaler.joblib        # produced by train.py
│  └─ metrics.json         # produced by train.py
├─ frontend/                # Minimal static page to exercise the API
│  └─ index.html
├─ vercel.json              # Vercel routes & runtime config
├─ requirements.txt
└─ README.md
```

## Setup
1) Create and activate a virtual environment
```
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
```

2) Install dependencies
```
pip install -r requirements.txt
```

3) Place dataset
- Put the UCI Heart dataset CSV at `data/heart.csv`.
- Column names are normalized automatically; common variants are handled.

## Train the Model
```
python src/train.py
```
This will train a RandomForest model, print metrics, and save artifacts to `model/`:
- `model.joblib`
- `scaler.joblib`
- `metrics.json` (includes ROC AUC)

## Run locally
You have two options locally:

1) Streamlit UI (full interactive app)
```
streamlit run app/streamlit_app.py
```
Open the provided URL in your browser. Enter patient features and click Predict to see probability, risk level, ROC AUC (from test set), and explanations.

2) FastAPI server (Vercel-style API)
```
uvicorn api.index:app --host 127.0.0.1 --port 8000
```
- Health check: GET http://127.0.0.1:8000/health → {"ok": true}
- Metrics: GET http://127.0.0.1:8000/metrics → metrics JSON (if available)
- Predict: POST http://127.0.0.1:8000/predict
	Body (JSON):
	{
		"instances": [
			{"age":54, "sex":1, "trestbps":130, "chol":246, "fbs":0, "thalach":150, "exang":0, "oldpeak":1.0}
		]
	}

Minimal static page to exercise the API:
```
open frontend/index.html
```

## Deploy to Streamlit Cloud
1. Push this project to GitHub.
2. In Streamlit Cloud, create a new app from your repository.
3. Set app path to `app/streamlit_app.py` and deploy.

## Deploy to Vercel (API + static)
This repo includes a FastAPI serverless function and a minimal static frontend.

1) Ensure artifacts exist and are committed (simplest):
	 - `model/model.joblib`
	 - `model/scaler.joblib`
	 - `model/metrics.json` (optional)
2) Push to GitHub and import the repo in Vercel.
3) Vercel reads `vercel.json`:
	 - `/api/*` → Python function `api/index.py` (runtime python3.11)
	 - `/` → serves `frontend/index.html`
4) After deploy, test endpoints:
	 - `GET https://<your-app>.vercel.app/api/health`
	 - `POST https://<your-app>.vercel.app/api/predict`

Note: Streamlit is not hosted on Vercel. For the full UI, deploy Streamlit on Streamlit Cloud or as a container elsewhere.

## API reference
- POST /api/predict
	- Request JSON: {"instances": [Patient, ...]}
	- Patient fields (all optional, missing values will be imputed): age, sex, trestbps, chol, fbs, thalach, exang, oldpeak
	- Response JSON: {"predictions": [0|1,...], "probabilities": [0..1,...], "features": [...]}
- GET /api/health → {"ok": bool}
- GET /api/metrics → metrics.json contents or {}

## Notes
- The app degrades gracefully if SHAP cannot run (still shows global feature importances).
- This is an educational tool, not a medical device.

## Testing
Two convenience test scripts are provided:
- `python test_qa.py`
- `python test_comprehensive.py`

## Git workflow
Typical steps to commit your changes:
```
git init                # if this folder is not a git repo yet
git remote add origin https://github.com/<your-username>/<your-repo>.git
git checkout -b main    # or use existing main branch
git add .
git commit -m "feat: add Vercel API, static frontend, and docs"
git push -u origin main
```

## Dataset Citation
UCI Machine Learning Repository: Heart Disease Data Set.


