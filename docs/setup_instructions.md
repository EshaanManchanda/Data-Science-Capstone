# 🚀 Setup Instructions
## SpaceX Falcon 9 Landing Prediction

---

## Prerequisites

- Python 3.9 or higher
- pip package manager
- 4GB RAM minimum

---

## Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/spacex-landing-prediction.git
cd spacex-landing-prediction
```

---

## Step 2: Create Virtual Environment

### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### On Mac/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 4: Verify Installation

```bash
python -c "import pandas; import sklearn; import xgboost; print('All packages installed!')"
```

---

## Step 5: Run Notebooks

### Option A: Jupyter Lab
```bash
jupyter lab notebooks/
```

### Option B: Jupyter Notebook
```bash
jupyter notebook notebooks/
```

---

## Step 6: Run Web Application

### Streamlit App:
```bash
streamlit run deployment/streamlit/app.py
```

The app will open at: http://localhost:8501

### Flask API:
```bash
python deployment/flask/app.py
```

The API will run at: http://localhost:5000

---

## Project Structure

```
spacex-landing-prediction/
├── data/
│   ├── raw/spacex_data.csv
│   └── processed/spacex_cleaned.csv
├── notebooks/
│   ├── 01_eda_analysis.ipynb
│   └── 02_model_training.ipynb
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── models/
│   ├── visualization/
│   └── deployment/
├── models/
│   ├── best_model.pkl
│   └── scaler.pkl
├── deployment/
│   ├── streamlit/app.py
│   └── flask/app.py
├── docs/
├── requirements.txt
└── README.md
```

---

## Quick Start Guide

### 1. Data Analysis
```python
import pandas as pd
df = pd.read_csv('data/raw/spacex_data.csv')
print(df.head())
```

### 2. Train Model
```python
from src.models.trainer import ModelTrainer
trainer = ModelTrainer()
# Follow notebooks for complete training
```

### 3. Make Predictions
```python
from src.models.predictor import Predictor
predictor = Predictor()
predictor.load_model('models/best_model.pkl')
result = predictor.predict_single(features)
```

---

## Common Issues

### Issue 1: ModuleNotFoundError
**Solution:** Ensure you're in the project root and have activated the virtual environment

### Issue 2: Port Already in Use
**Solution:** Use a different port
```bash
streamlit run deployment/streamlit/app.py --server.port 8502
```

### Issue 3: Model Not Found
**Solution:** Run the model training notebook first to create the model file

---

## Testing

### Run Tests:
```bash
pytest tests/
```

### Run Specific Test:
```bash
pytest tests/test_predictor.py -v
```

---

## Deployment

### Local Deployment
Already covered in Step 6

### Cloud Deployment (Heroku)
```bash
# Create Procfile
echo "web: gunicorn deployment.flask.app:app" > Procfile
git add .
git commit -m "Deploy"
git push heroku main
```

---

## Support

For issues or questions:
- Email: eshaan@example.com
- GitHub Issues: [Link]

---

**Happy Learning! 🚀**