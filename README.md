# BRAC Household Aid Eligibility Prediction System

AI-powered system to predict household eligibility for Income Generating Activity (IGA) support.

## Features
- **Eligibility Prediction**: Predicts probability (0-100%) of household qualifying for aid
- **Aid Type Recommendation**: Recommends suitable aid type (Enterprise Development, Skill Development, Both)
- **Explainable AI**: Shows top 3 factors driving each prediction

## Model Performance
| Task | Model | Metric | Score |
|------|-------|--------|-------|
| Eligibility | Random Forest | ROC-AUC | 0.78 |
| Aid Type | Random Forest | Accuracy | 91% |

## Fairness
✅ Passed all fairness tests across protected attributes

## Tech Stack
- Python, Scikit-learn, SHAP
- Streamlit for Web Interface

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Author
Nasim Mahmud - ML Assignment for BRAC Technology Division