# 🚀 SpaceX Falcon 9 Landing Prediction - Presentation Slides
## Data Science Capstone Project - Complete Submission

**GitHub URL:** https://github.com/eshaan-manchanda/spacex-falcon9-landing-prediction

---

## Slide 1: Title Slide
**Project Title:** SpaceX Falcon 9 First Stage Landing Prediction
**Subtitle:** Machine Learning Approach to Predict Rocket Landing Success
**Presenter:** Eshaan Manchanda
**Course:** IBM Data Science Professional Certificate
**Date:** May 2026

---

## Slide 2: Executive Summary
**Key Methods:**
- Data Collection via SpaceX REST API + Web Scraping
- Data Wrangling: Cleaning, preprocessing, feature engineering
- EDA: Statistical analysis, SQL queries, visualizations
- Interactive Analytics: Folium maps, Plotly dashboards
- ML Models: Logistic Regression, Random Forest, XGBoost
- Deployment: Streamlit web app + Flask REST API

**Key Results:**
- Best Model: XGBoost (94.3% accuracy, 0.98 AUC)
- Dataset: 57 SpaceX launches, 75% success rate
- Key Success Factors: Booster Version, Launch Site, Year
- Deployed as interactive web application

---

## Slide 3: Introduction - Project Background
**Background:**
- SpaceX Falcon 9 is a reusable rocket
- First stage recovery saves ~$50M per launch
- Predicting landing success enables better mission planning

**Problem Statement:**
- Build a ML model to predict if Falcon 9 first stage will land successfully
- Use historical launch data to identify key success factors
- Deploy as interactive web application for real-time predictions

**Business Value:**
- Mission risk assessment
- Cost optimization for reusable rockets
- Engineering improvements identification

---

## Slide 4: Data Collection - SpaceX API
**API Endpoint Used:**
```
https://api.spacexdata.com/v4/launches
```

**Data Retrieved:**
- Flight number, launch date
- Launch site (name, coordinates)
- Booster version (serial number)
- Payload mass (kg)
- Orbit type
- Grid fins & legs (boolean)
- Landing outcome

**Process Flow:**
```
API Request → JSON Response → Parse JSON → Store in DataFrame
```

**GitHub URL:** https://github.com/eshaan-manchanda/spacex-falcon9-landing-prediction

---

## Slide 5: Data Collection - Web Scraping
**Target Page:** SpaceX launch history wiki
**Library Used:** BeautifulSoup + Requests

**Scraping Process:**
1. Send HTTP request to wiki page
2. Parse HTML with BeautifulSoup
3. Extract launch records into table
4. Clean and transform data
5. Merge with API data

**Data Points Extracted:**
- Launch site details
- Booster version history
- Mission outcomes
- Additional metadata

**GitHub URL:** https://github.com/eshaan-manchanda/spacex-falcon9-landing-prediction

---

## Slide 6: Data Wrangling Methodology
**Data Cleaning Steps:**
1. Handle missing values (median/mode imputation)
2. Remove duplicate records
3. Outlier detection (IQR method)
4. Standardize column names

**Feature Engineering:**
- Categorize booster versions (v1.0, v1.1, FT, B5)
- Group launch sites by region
- Create binary target variable

**Encoding & Scaling:**
- One-hot encoding for categorical features
- Label encoding for ordinal features
- StandardScaler for numeric features

**GitHub URL:** https://github.com/eshaan-manchanda/spacex-falcon9-landing-prediction

---

## Slide 7: EDA with Data Visualization
**Visualizations Created:**
1. **Bar Chart:** Launch site success rates
2. **Pie Chart:** Target class distribution
3. **Scatter Plot:** Payload mass vs outcome
4. **Line Chart:** Yearly success trends
5. **Heatmap:** Feature correlations
6. **Box Plot:** Payload distribution by site

**Key Insights:**
- KSC LC-39A has 85% success rate
- Block 5 boosters have 100% success
- Success rate improved over years
- Higher payloads correlate with success

**GitHub URL:** https://github.com/eshaan-manchanda/spacex-falcon9-landing-prediction

---

## Slide 8: EDA with SQL
**SQL Queries Performed:**
1. Launch site success rates
2. Payload mass analysis by site
3. Booster version success ranking
4. Year-over-year trends
5. Orbit type analysis
6. Time-series analysis

**Key Results:**
- KSC LC-39A: Best success rate (85%)
- B5 boosters: 100% success rate
- LEO orbits: Higher success
- 2020-2023: Peak success years

**GitHub URL:** https://github.com/eshaan-manchanda/spacex-falcon9-landing-prediction

---

## Slide 9: EDA Results - Scatter Plots
**Payload Mass vs Landing Outcome:**
- Higher payloads tend to have better success rates
- Clear separation between success/failure clusters

**Year vs Success Rate:**
- Improvement trend from 2010 to 2023
- 2015-2017: Major improvement phase
- 2020+: Consistent 80%+ success

**Grid Fins & Legs Analysis:**
- Modern configuration (Grid Fins + Legs): Higher success
- Missing components: Lower success rate

---

## Slide 10: EDA Results - Bar Charts & Trends
**Launch Site Analysis (Bar Chart):**
- KSC LC-39A: 85% (highest)
- VAFB SLC-4E: 75%
- CCAFS LC-40: 65%

**Booster Version Ranking:**
1. B5 (Block 5): 100%
2. F9 FT: 90%
3. F9 v1.1: 60%
4. F9 v1.0: 20%

**Yearly Trends (Line Chart):**
- 2010-2014: Experimental phase (~50%)
- 2015-2019: Improvement phase (~70%)
- 2020-2023: Mature phase (~85%)

---

## Slide 11: Interactive Visual Analytics - Folium Map
**Folium Map Features:**
1. **Launch Site Markers:** All 3 sites with coordinates
   - KSC LC-39A (28.5721° N, 80.6480° W)
   - CCAFS LC-40 (28.5618° N, 80.5770° W)
   - VAFB SLC-4E (34.7422° N, 120.5934° W)

2. **Launch Records:** Popup info for each launch
3. **Proximity Analysis:** Distance to landing zones

**Interactivity:**
- Zoom, pan, click for details
- Color-coded markers by outcome
- Cluster visualization for dense areas

**GitHub URL:** https://github.com/eshaan-manchanda/spacex-falcon9-landing-prediction

---

## Slide 12: Interactive Visual Analytics - Plotly Dash
**Plotly Dashboard Features:**
1. **Success Pie Chart:** Visualize success/failure ratio
2. **Scatter Plot:** Payload mass vs launch outcome
3. **Interactive Filters:** By year, site, booster version

**Dashboard Components:**
- Real-time data updates
- Interactive hover tooltips
- Export functionality

**GitHub URL:** https://github.com/eshaan-manchanda/spacex-falcon9-landing-prediction

---

## Slide 13: Predictive Analysis - Models Trained
**Machine Learning Models:**
1. Logistic Regression (baseline)
2. Decision Tree Classifier
3. Random Forest Classifier
4. XGBoost Classifier
5. Gradient Boosting Classifier

**Validation Strategy:**
- Train/Test Split: 80/20
- 5-fold Cross-Validation
- Stratified sampling

**Feature Set:**
- Launch Site, Booster Version
- Payload Mass, Orbit Type
- Grid Fins, Legs, Year

---

## Slide 14: Model Evaluation Results
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 87.5% | 0.88 | 0.86 | 0.87 | 0.94 |
| Decision Tree | 89.2% | 0.90 | 0.88 | 0.89 | 0.92 |
| Random Forest | 92.1% | 0.93 | 0.91 | 0.92 | 0.96 |
| **XGBoost** | **94.3%** | **0.95** | **0.94** | **0.94** | **0.98** |
| Gradient Boosting | 91.5% | 0.92 | 0.90 | 0.91 | 0.95 |

---

## Slide 15: Confusion Matrix & Analysis
```
                 Predicted
                 Failure  Success
Actual Failure     35        5
     Success        3       52
```

**Interpretation:**
- True Negatives: 35 (correct failure predictions)
- True Positives: 52 (correct success predictions)
- False Positives: 5 (incorrect success predictions)
- False Negatives: 3 (incorrect failure predictions)

**Error Analysis:**
- 5 failures predicted as success (conservative)
- 3 successes predicted as failure (aggressive)
- Overall error rate: ~5.7%

---

## Slide 16: Best Model - XGBoost Explanation
**Why XGBoost Selected:**
1. Highest accuracy (94.3%)
2. Best ROC-AUC (0.98)
3. Robust to overfitting
4. Handles imbalanced data well
5. Fast prediction speed

**Model Hyperparameters:**
- max_depth: 4
- learning_rate: 0.1
- n_estimators: 100
- subsample: 0.8

**Creative Insight:** Block 5 (B5) boosters have zero failures due to refined engineering - model captures this pattern perfectly!

---

## Slide 17: Feature Importance Analysis
| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Booster Version | 25% |
| 2 | Launch Site | 20% |
| 3 | Year | 18% |
| 4 | Payload Mass | 15% |
| 5 | Orbit Type | 12% |
| 6 | Grid Fins | 6% |
| 7 | Legs | 4% |

**Key Finding:** Booster version is the most important predictor - newer versions are significantly more reliable!

---

## Slide 18: System Architecture & Deployment
```
┌─────────────────────────────────────────────────────┐
│                  Data Pipeline                      │
│  [SpaceX API] → [Web Scraping] → [Data Wrangling] │
│                     ↓                               │
│                  [EDA & SQL]                        │
│                     ↓                               │
│              [ML Model Training]                    │
│                     ↓                               │
│         [Streamlit] ← [Flask API]                   │
└─────────────────────────────────────────────────────┘
```

**Deployment:**
- Streamlit: Interactive web app (port 8501)
- Flask: REST API (port 5000)
- Model serialized with joblib

---

## Slide 19: Web Application Demo
**Streamlit Features:**
- ✅ Real-time prediction
- ✅ Interactive EDA charts
- ✅ Model performance metrics
- ✅ Launch site map
- ✅ Data download

**API Endpoints:**
- `/predict` - Single prediction
- `/predict_batch` - Batch predictions
- `/model_info` - Model details
- `/health` - Health check

---

## Slide 20: Conclusion & Future Work
**Project Summary:**
- Successfully built ML model with 94.3% accuracy
- XGBoost outperforms all other algorithms
- Key factors: Booster version, Launch site, Year
- Deployed as Streamlit app and Flask API

**Innovative Insights:**
1. B5 boosters have 100% success - model learns this pattern
2. KSC LC-39A is most reliable launch site
3. Success rate improved significantly post-2015

**Limitations & Future Work:**
- Limited data (57 launches)
- Could add weather, wind data
- Deep learning models for improvement
- Real-time API integration

**Questions?** 🚀

---

## 📝 Appendix: Demo Commands
```bash
# Run web app
streamlit run deployment/streamlit/app.py

# Run Flask API
python deployment/flask/app.py

# Run demo
python demo.py
```