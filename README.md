# 🚀 SpaceX Falcon 9 First Stage Landing Prediction

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Project Overview

This comprehensive data science capstone project predicts whether SpaceX's Falcon 9 first stage will successfully land after launch. The project integrates features from all Coursera IBM Data Science notebooks into a production-ready system.

### 🎯 Key Features

- ✅ **Data Collection** - SpaceX API integration
- ✅ **Data Wrangling** - Complete cleaning & preprocessing  
- ✅ **EDA** - Comprehensive statistical analysis
- ✅ **SQL Analysis** - Database queries & exploration
- ✅ **Geographic Mapping** - Interactive Folium maps
- ✅ **Machine Learning** - 5 models with unified pipeline
- ✅ **Predictions** - Real-time inference
- ✅ **Web Dashboard** - Streamlit analytics

---

## 📁 Project Structure

```
spacex-landing-prediction/
├── data/                    # Dataset files
│   ├── raw/                # Raw data
│   └── processed/          # Cleaned data
├── notebooks/              # Jupyter notebooks
│   ├── 01_eda_analysis.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_complete_pipeline.ipynb
│   └── 04_time_series_analysis.ipynb
├── src/                    # Source code
│   ├── data/              # Data modules
│   │   ├── api_client.py      # SpaceX API
│   │   ├── loader.py          # Data loading
│   │   ├── validator.py        # Validation
│   │   └── wrangler.py         # Data cleaning
│   ├── analysis/          # Analytics
│   │   └── sql_analyzer.py     # SQL queries
│   ├── models/            # ML models
│   │   ├── unified_pipeline.py # Complete ML
│   │   ├── trainer.py          # Model training
│   │   ├── predictor.py        # Predictions
│   │   └── evaluator.py        # Evaluation
│   ├── preprocessing/     # Preprocessing
│   │   ├── pipeline.py
│   │   └── transformer.py
│   └── visualization/      # Visualizations
│       ├── eda_analysis.py    # EDA
│       ├── mapping.py         # Folium maps
│       └── plots.py            # Charts
├── models/                # Trained models
├── deployment/           # Deployment
│   ├── streamlit/        # Streamlit app
│   ├── flask/            # Flask API
│   └── dashboard/        # Complete dashboard
├── docs/                 # Documentation
├── tests/                # Unit tests
├── requirements.txt      # Dependencies
├── demo.py              # Quick demo
├── run_complete.py       # Complete run
└── README.md            # This file
```

---

## 🚀 Quick Start

### 1. Installation

```bash
cd spacex-landing-prediction
pip install -r requirements.txt
```

### 2. Run Demo

```bash
python demo.py
```

### 3. Run Dashboard

```bash
streamlit run dashboard/complete_dashboard.py
```

### 4. Run API

```bash
python deployment/flask/app.py
```

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** | **92.1%** | **0.93** | **0.92** | **0.92** | **0.96** |
| Gradient Boosting | 91.5% | 0.92 | 0.91 | 0.91 | 0.95 |
| XGBoost | 94.3% | 0.95 | 0.94 | 0.94 | 0.98 |
| Decision Tree | 89.2% | 0.91 | 0.89 | 0.90 | 0.92 |
| Logistic Regression | 87.5% | 0.89 | 0.88 | 0.88 | 0.94 |

---

## 🔧 Technology Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.9+ |
| Data Processing | Pandas, NumPy |
| ML Framework | Scikit-learn, XGBoost |
| Visualization | Plotly, Matplotlib, Seaborn, Folium |
| Web App | Streamlit |
| API | Flask |
| Database | SQLite |

---

## 📈 Key Insights

1. **Booster Version**: Block 5 (B5) has 100% success rate in recent years
2. **Launch Site**: KSC LC-39A has the highest success rate (~85%)
3. **Year Trend**: Success rate improved from 20% (v1.0) to 100% (B5)
4. **Payload Mass**: Higher payloads correlate with successful landings

---

## 📱 Web Application Features

- ✅ Real-time prediction with probability
- ✅ Interactive data visualizations
- ✅ Model performance metrics
- ✅ Launch history analysis
- ✅ Filter and search data
- ✅ Export data to CSV
- ✅ Dark/Light mode UI

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/predict` | POST | Single prediction |
| `/predict_batch` | POST | Batch predictions |
| `/model_info` | GET | Model details |
| `/data_statistics` | GET | Dataset statistics |

---

## 📚 Documentation

- [Project Report](docs/project_report.md) - Full technical report
- [Presentation](docs/presentation.md) - 20-slide presentation
- [Interview Q&A](docs/interview_qa.md) - 35+ viva questions
- [Setup Guide](docs/setup_instructions.md) - Installation instructions
- [API Docs](docs/api_docs.md) - API reference

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
python -m pytest tests/test_model.py -v
```

---

## 🐳 Docker

```bash
# Build image
docker build -t spacex-prediction .

# Run container
docker run -p 8501:8501 spacex-prediction
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

---

## 📝 License

MIT License - See LICENSE file for details

---

## 👨‍💻 Author

**Eshaan Manchanda**  
Data Science Capstone Project  
Coursera IBM Data Science Professional Certificate

---

## 🙏 Acknowledgments

- SpaceX for the public launch data
- IBM/Coursera for the Data Science Professional Certificate
- Open source community for the tools and libraries

---

**🚀 Ready for deployment!**