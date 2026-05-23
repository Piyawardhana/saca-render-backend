# SACA Render Backend

This is the cleaned backend folder for hosting the SACA FastAPI demo on Render.

## Render settings

- Language: Python 3
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## Test endpoints

- `/` - API running message
- `/health` - confirms model loading
- `/predict` - POST endpoint for triage prediction

Example request body for `/predict`:

```json
{
  "text": "I have severe chest pain and difficulty breathing for 2 days",
  "age": 45,
  "gender": "male",
  "pain_score": 8,
  "body_part": "chest"
}
```

## Files intentionally included

- `main.py`: FastAPI app and prediction endpoints
- `nlp/`: text normalisation, symptom extraction, rules/recommendations
- `ml/saved/linear_svc.joblib`: severity model
- `ml/saved/disease/disease_model.joblib`: disease prediction model
- `requirements.txt`: Python dependencies for Render
- `.python-version`: Python version hint for Render

## Files intentionally removed

- raw/processed datasets
- training scripts
- model comparison scripts
- result images and reports
- unused model files
- `__pycache__` folders
