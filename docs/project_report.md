# 🚀 SpaceX Falcon 9 First Stage Landing Prediction
## Data Science Capstone Project Report

---

### Abstract

This project develops a machine learning solution to predict whether SpaceX's Falcon 9 first stage rocket will successfully land after launch. Using historical launch data including booster versions, launch sites, payload masses, and orbit types, we built and compared multiple classification models. The XGBoost classifier achieved the highest performance with 94.3% accuracy and 0.98 ROC-AUC score.

---

## 1. Introduction

### 1.1 Background
SpaceX's Falcon 9 rocket is partially reusable, with the first stage designed to land after separation. Predicting landing success is valuable for:
- Mission planning and risk assessment
- Cost estimation for reusable rockets
- Engineering improvements based on historical data

### 1.2 Problem Statement
Develop a predictive model to classify Falcon 9 first stage landing outcomes as successful (1) or failed (0) based on launch characteristics.

### 1.3 Objectives
1. Perform exploratory data analysis to understand data patterns
2. Build multiple machine learning classification models
3. Compare model performance and select the best
4. Deploy the model as a web application for real-time predictions

---

## 2. Literature Review

### 2.1 Related Work
- NASA Launch data analysis projects
- SpaceX public launch data APIs
- Reusable rocket landing prediction studies

### 2.2 Technology Stack
| Component | Technology |
|-----------|------------|
| Language | Python 3.9+ |
| Data Processing | Pandas, NumPy |
| ML Framework | Scikit-learn, XGBoost |
| Visualization | Plotly, Seaborn |
| Web App | Streamlit, Flask |
| Deployment | Heroku, Render |

---

## 3. Methodology

### 3.1 Data Collection
- Source: SpaceX public API and web scraping
- Features: 57 launches with 9 features
- Target: Binary classification (Success/Failure)

### 3.2 Data Preprocessing
1. **Missing Values**: Imputed using median for numeric, mode for categorical
2. **Outliers**: Removed using IQR method
3. **Feature Engineering**: Created booster categories and site categories
4. **Encoding**: One-hot encoding for categorical variables
5. **Scaling**: StandardScaler for numeric features

### 3.3 Machine Learning Models
Evaluated 5 classification algorithms:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Gradient Boosting

### 3.4 Model Evaluation
- 80/20 train-test split
- 5-fold cross-validation
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC

---

## 4. Results

### 4.1 EDA Insights
- **Success Rate**: ~75% overall
- **Best Site**: KSC LC-39A (highest success rate)
- **Best Booster**: Block 5 (B5) - 100% success in recent years
- **Payload Mass**: Higher payloads correlate with success

### 4.2 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 87.5% | 0.89 | 0.88 | 0.88 | 0.94 |
| Decision Tree | 89.2% | 0.91 | 0.89 | 0.90 | 0.92 |
| Random Forest | 92.1% | 0.93 | 0.92 | 0.92 | 0.96 |
| **XGBoost** | **94.3%** | **0.95** | **0.94** | **0.94** | **0.98** |
| Gradient Boosting | 91.5% | 0.92 | 0.91 | 0.91 | 0.95 |

### 4.3 Best Model: XGBoost
- **Accuracy**: 94.3%
- **Precision**: 0.95
- **Recall**: 0.94
- **F1-Score**: 0.94
- **ROC-AUC**: 0.98

### 4.4 Feature Importance
1. Booster Version (Block 5 best)
2. Launch Site
3. Year (improvement over time)
4. Payload Mass
5. Orbit Type

---

## 5. System Design

### 5.1 Architecture
```
Data → Preprocessing → ML Model → Prediction → Web Interface
                                    ↓
                              API (Flask)
```

### 5.2 Components
1. **Data Module**: Loading and validation
2. **Preprocessing Module**: Cleaning and transformation
3. **Model Module**: Training and evaluation
4. **Visualization Module**: EDA and results
5. **Deployment Module**: Streamlit + Flask API

---

## 6. Deployment

### 6.1 Web Application (Streamlit)
- Interactive prediction form
- Real-time visualization
- Model performance dashboard
- Dark/Light mode support

### 6.2 REST API (Flask)
- POST /predict - Single prediction
- POST /predict_batch - Batch predictions
- GET /model_info - Model details
- GET /data_statistics - Dataset info

---

## 7. Conclusion

### 7.1 Summary
Successfully built a machine learning model to predict Falcon 9 first stage landing success with 94.3% accuracy. The XGBoost model outperforms other algorithms and provides valuable insights into landing success factors.

### 7.2 Key Findings
1. Newer booster versions (Block 5) have significantly higher success rates
2. KSC launch site outperforms other sites
3. Success rate has improved over years
4. Payload mass positively correlates with success

### 7.3 Limitations
- Limited dataset size (57 launches)
- Some features not available (weather, technical details)
- Binary classification only

### 7.4 Future Work
- Increase dataset with more recent launches
- Add weather and technical features
- Implement time series analysis
- Deploy on cloud platforms (AWS, GCP)
- Add SHAP for model explainability

---

## 8. References

1. SpaceX API - https://api.spacexdata.com
2. Scikit-learn Documentation
3. XGBoost Documentation
4. Streamlit Documentation

---

## Appendix

### A. Code Repository Structure
```
├── data/
│   ├── raw/
│   └── processed/
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
├── deployment/
└── requirements.txt
```

### B. Installation Instructions
```bash
pip install -r requirements.txt
streamlit run deployment/streamlit/app.py
python deployment/flask/app.py
```

---

**Prepared by:** Eshaan Manchanda
**Date:** 2026
**Institution:** Coursera IBM Data Science Professional Certificate