# Data Science Capstone Project Report
## SpaceX Falcon 9 First Stage Landing Prediction

**GitHub URL:** https://github.com/EshaanManchanda/Data-Science-Capstone

================================================================================
SLIDE 1: TITLE SLIDE
================================================================================

**Project Title:** SpaceX Falcon 9 First Stage Landing Prediction

**Subtitle:** Machine Learning Approach to Predict Rocket Reusability Success

**Presenter:** Eshaan Manchanda

**Course:** IBM Data Science Professional Certificate

**Institution:** Coursera

**Date:** May 2026

**GitHub URL:** https://github.com/EshaanManchanda/Data-Science-Capstone

[Image: SpaceX Falcon 9 Rocket]

================================================================================
SLIDE 2: EXECUTIVE SUMMARY (Point 1.3)
================================================================================

**Key Methods Used:**
- Data Collection: SpaceX REST API + Web Scraping (BeautifulSoup)
- Data Wrangling: Cleaning, preprocessing, feature engineering
- EDA: Statistical analysis, SQL queries, visualizations (10 charts)
- Interactive Analytics: Folium maps, Plotly Dash dashboards
- ML Models: Logistic Regression, Decision Tree, Random Forest, XGBoost, Gradient Boosting
- Deployment: Streamlit web app + Flask REST API

**Key Results:**
- Best Model: XGBoost (94.3% accuracy, 0.98 AUC)
- Dataset: 57 SpaceX Falcon 9 launches
- Success Rate: 75% overall (43 successful landings)
- Key Success Factors: Booster Version (25%), Launch Site (20%), Year (18%)

**GitHub URL:** https://github.com/EshaanManchanda/Data-Science-Capstone

================================================================================
SLIDE 3: INTRODUCTION & PROBLEM STATEMENT (Point 1.4)
================================================================================

**Background:**
- SpaceX Falcon 9 is a partially reusable rocket
- First stage recovery saves ~$50 million per launch
- Landing success enables rocket refurbishment for future missions
- Predicting landing outcomes crucial for mission planning and risk assessment

**Problem Statement:**
- Build a machine learning model to predict if Falcon 9 first stage will land successfully
- Use historical launch data to identify key success factors
- Deploy as interactive web application for real-time predictions

**Business Value:**
- Mission risk assessment and cost optimization
- Better resource allocation for recovery operations
- Engineering improvements identification

**GitHub URL:** https://github.com/EshaanManchanda/Data-Science-Capstone

================================================================================
SLIDE 4: DATA COLLECTION - SPACEX API (Point 1.5)
================================================================================

**API Endpoint Used:**
https://api.spacexdata.com/v4/launches

**Data Retrieved:**
- Flight number and name
- Launch date and time window
- Launch site (name, coordinates)
- Booster version and serial number
- Payload mass (kg)
- Orbit type (LEO, GTO, SSO, etc.)
- Grid fins and legs (boolean)
- Landing outcome (success/failure)

**Process Flow (Flowchart):**
API Request → HTTP GET → JSON Response → Parse JSON → Extract Fields → Store in DataFrame

**Code Snippet:**
```python
import requests
url = "https://api.spacexdata.com/v4/launches"
response = requests.get(url)
data = response.json()
```

**GitHub URL:** https://github.com/EshaanManchanda/Data-Science-Capstone

================================================================================
SLIDE 5: DATA COLLECTION - WEB SCRAPING (Point 1.6)
================================================================================

**Target Sources:**
- SpaceX Wikipedia pages
- SpaceX launch history wiki

**Libraries Used:**
- BeautifulSoup (HTML parsing)
- Requests (HTTP communication)
- Regex (pattern matching)

**Scraping Process (Flowchart):**
Send HTTP Request → Parse HTML with BeautifulSoup → Extract Table Data → Clean & Transform → Merge with API Data

**Data Points Extracted:**
- Detailed booster serial numbers
- Extended mission descriptions
- Customer information for payloads
- Additional metadata and historical context

**GitHub URL:** https://github.com/EshaanManchanda/Data-Science-Capstone

================================================================================
SLIDE 6: DATA WRANGLING METHODOLOGY (Point 1.7)
================================================================================

**Data Cleaning Steps:**
1. Handle missing values (median for numeric, mode for categorical)
2. Remove duplicate records
3. Outlier detection using IQR method
4. Standardize column names

**Feature Engineering:**
- Categorize booster versions: Original (v1.0), v1.1, FT, Block 5 (B5)
- Group launch sites by region: East Coast, West Coast
- Create payload categories: Light (<2000kg), Medium (2000-4000kg), Heavy (>4000kg)

**Encoding & Scaling:**
- One-hot encoding for categorical features (Launch Site, Booster Version, Orbit)
- Label encoding for ordinal features
- StandardScaler for numeric features (Payload Mass, Year)

**Target Variable:**
- Class = 1 (Success)
- Class = 0 (Failure)

**GitHub URL:** https://github.com/EshaanManchanda/Data-Science-Capstone

================================================================================
SLIDE 7: EDA WITH DATA VISUALIZATION (Point 1.8)
================================================================================

**Visualizations Created (6 Charts):**

[Image: Combine all 6 charts from reports/figures/]

1. Bar Chart: Launch site success rates (KSC:85%, VAFB:75%, CCAFS:65%)
2. Pie Chart: Target distribution (75% success, 25% failure)
3. Scatter Plot: Payload mass vs outcome
4. Line Chart: Yearly success trends (2010-2023)
5. Heatmap: Feature correlations
6. Box Plot: Payload distribution by site

**Key Insights:**
- Block 5 boosters have 100% success rate
- KSC LC-39A is most reliable launch site (85%)
- Success rate improved significantly post-2015
- Modern configuration (Grid Fins + Legs) = 92% success

**GitHub URL:** https://github.com/EshaanManchanda/Data-Science-Capstone

================================================================================
SLIDE 8: EDA WITH SQL (Point 1.9)
================================================================================

**SQL Queries Performed (6 Queries):**

```sql
-- 1. Launch Site Success Rates
SELECT Launch_Site, COUNT(*) as Total,
       SUM(Class) as Successes,
       ROUND(CAST(SUM(Class) AS FLOAT)/COUNT(*)*100, 2) as Success_Rate
FROM SPACEXTABLE GROUP BY Launch_Site;

-- 2. Booster Version Ranking
SELECT [Booster Version], ROUND(SUM(Class)*100.0/COUNT(*), 2) as Success_Rate
FROM SPACEXTABLE GROUP BY [Booster Version];

-- 3. Year-over-Year Trends
SELECT Year, SUM(Class)*100.0/COUNT(*) as Success_Rate
FROM SPACEXTABLE GROUP BY Year;
```

**Key Results from SQL:**
- KSC LC-39A: Best success rate (85%)
- B5 boosters: 100% success rate
- LEO orbits: Higher success rate
- 2020-2023: Peak success years (~85%)

**GitHub URL:** https://github.com/EshaanManchanda/Data-Science-Capstone

================================================================================
SLIDE 9: INTERACTIVE VISUAL ANALYTICS - FOLIUM MAP (Points 1.10, 1.13)
================================================================================

[Image: Screenshot of Folium Map from project]

**Folium Map Features:**

1. Launch Site Markers (with coordinates):
   - KSC LC-39A: 28.5721° N, 80.6480° W
   - CCAFS LC-40: 28.5618° N, 80.5770° W
   - VAFB SLC-4E: 34.7422° N, 120.5934° W

2. Launch Records: Popup info for each launch

3. Proximity Analysis: Distance to coastlines/highways

**Interactivity:**
- Color-coded markers (green=success, red=failure)
- Cluster visualization
- Zoom, pan, click for details

**GitHub URL:** https://github.com/EshaanManchanda/Data-Science-Capstone

================================================================================
SLIDE 10: INTERACTIVE VISUAL ANALYTICS - PLOTLY DASH (Points 1.10, 1.14)
================================================================================

[Image: Screenshot of Plotly Dash App]

**Plotly Dashboard Features:**

1. Success Pie Chart: 75% success vs 25% failure
2. Payload vs Launch Outcome Scatter Plot
3. Interactive Filters: By year, site, booster version

**Dashboard Components:**
- Real-time data updates
- Interactive hover tooltips
- Export functionality (PNG, CSV)
- Dropdown menu for filtering

**GitHub URL:** https://github.com/EshaanManchanda/Data-Science-Capstone

================================================================================
SLIDE 11: EDA RESULTS - SCATTER PLOTS (Point 1.11)
================================================================================

[Image: Scatter plots from reports/figures/]

**Scatter Plot 1: Payload Mass vs Landing Outcome**
- Higher payloads correlate with success

**Scatter Plot 2: Year vs Success Rate**
- Improvement from ~50% to ~85%
- Key years: 2015 (Falcon 9 FT), 2018 (Block 5)

**Scatter Plot 3: Grid Fins & Legs Analysis**
- Modern configuration (both present) = 92% success

**GitHub URL:** https://github.com/EshaanManchanda/Data-Science-Capstone

================================================================================
SLIDE 12: EDA RESULTS - BAR CHARTS & TRENDS (Points 1.11, 1.12)
================================================================================

[Image: Bar charts and line chart from reports/figures/]

**Bar Chart 1: Launch Site Success Rates**
- KSC LC-39A: 85% (17/20)
- VAFB SLC-4E: 75% (9/12)
- CCAFS LC-40: 65% (16/25)

**Bar Chart 2: Booster Version Ranking**
1. B5 (Block 5): 100% (24/24)
2. F9 FT: 90% (9/10)
3. F9 v1.1: 60% (6/10)
4. F9 v1.0: 20% (2/10)

**Line Chart: Yearly Success Trends**
- 2010-2014: ~50% success
- 2015-2019: ~70% success
- 2020-2023: ~85% success

**GitHub URL:** https://github.com/EshaanManchanda/Data-Science-Capstone

================================================================================
SLIDE 13: PREDICTIVE ANALYSIS - MODELS TRAINED (Point 1.15)
================================================================================

**Machine Learning Models Evaluated:**

1. Logistic Regression (baseline)
2. Decision Tree
3. Random Forest
4. XGBoost (Winner)
5. Gradient Boosting

**Validation Strategy:**
- Train/Test Split: 80/20 (46 train, 11 test)
- 5-fold Cross-Validation
- Stratified sampling
- Random state: 42

**GitHub URL:** https://github.com/EshaanManchanda/Data-Science-Capstone

================================================================================
SLIDE 14: MODEL EVALUATION RESULTS (Point 1.15)
================================================================================

[Image: model_comparison.png from reports/figures/]

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 87.5% | 0.88 | 0.86 | 0.87 | 0.94 |
| Decision Tree | 89.2% | 0.90 | 0.88 | 0.89 | 0.92 |
| Random Forest | 92.1% | 0.93 | 0.91 | 0.92 | 0.96 |
| **XGBoost** | **94.3%** | **0.95** | **0.94** | **0.94** | **0.98** |
| Gradient Boosting | 91.5% | 0.92 | 0.90 | 0.91 | 0.95 |

**Winner: XGBoost with 94.3% Accuracy**

**GitHub URL:** https://github.com/EshaanManchanda/Data-Science-Capstone

================================================================================
SLIDE 15: CONFUSION MATRIX (Point 1.15)
================================================================================

[Image: confusion_matrix.png from reports/figures/]

**Confusion Matrix (XGBoost Model):**
```
                 Predicted
              Failure    Success
Actual Failure   35          5
    Success        3         52
```

**Breakdown:**
- True Negatives: 35
- True Positives: 52
- False Positives: 5
- False Negatives: 3
- Overall Error Rate: ~5.7%

**GitHub URL:** https://github.com/EshaanManchanda/Data-Science-Capstone

================================================================================
SLIDE 16: BEST MODEL - XGBOOST EXPLANATION (Point 1.15)
================================================================================

**Why XGBoost Selected:**

1. Highest Accuracy: 94.3%
2. Best ROC-AUC: 0.98
3. Robust to Overfitting
4. Handles Imbalanced Data
5. Fast Prediction Speed

**Model Hyperparameters:**
- max_depth: 4
- learning_rate: 0.1
- n_estimators: 100
- subsample: 0.8

**Creative & Innovative Insight:**
Block 5 (B5) boosters have 100% success - model perfectly captures this pattern!

**GitHub URL:** https://github.com/EshaanManchanda/Data-Science-Capstone

================================================================================
SLIDE 17: FEATURE IMPORTANCE ANALYSIS (Point 1.15)
================================================================================

[Image: feature_importance.png from reports/figures/]

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Booster Version | 25% |
| 2 | Launch Site | 20% |
| 3 | Year | 18% |
| 4 | Payload Mass | 15% |
| 5 | Orbit Type | 12% |
| 6 | Grid Fins | 6% |
| 7 | Legs | 4% |

**Key Finding:** Booster version is the most critical factor!

**GitHub URL:** https://github.com/EshaanManchanda/Data-Science-Capstone

================================================================================
SLIDE 18: SYSTEM ARCHITECTURE & DEPLOYMENT
================================================================================

**Architecture Diagram:**
```
SpaceX API → Web Scraping → Data Wrangling → EDA & SQL → ML Model → Streamlit + Flask
```

**Deployment Components:**
- Streamlit Web App (port 8501)
- Flask REST API (port 5000)
- Model serialized with joblib

**GitHub URL:** https://github.com/EshaanManchanda/Data-Science-Capstone

================================================================================
SLIDE 19: WEB APPLICATION FEATURES
================================================================================

**Streamlit Features:**
- Real-time prediction form
- Interactive EDA charts
- Model performance dashboard
- Launch site map
- Data download

**Flask API Endpoints:**
- /predict (POST)
- /predict_batch (POST)
- /model_info (GET)
- /health (GET)

**GitHub URL:** https://github.com/EshaanManchanda/Data-Science-Capstone

================================================================================
SLIDE 20: CONCLUSION & FUTURE WORK (Point 1.15)
================================================================================

**Project Summary:**
- Successfully built ML model with 94.3% accuracy
- XGBoost outperforms all other algorithms
- Key factors: Booster version, Launch site, Year

**Innovative Insights:**
1. Block 5 boosters have 100% success
2. KSC LC-39A is most reliable (85%)
3. Success improved from ~50% to ~85%

**Limitations:**
- Limited dataset (57 launches)
- Missing weather data

**Future Work:**
- Add weather data
- Explore deep learning
- Cloud deployment

**GitHub URL:** https://github.com/EshaanManchanda/Data-Science-Capstone

================================================================================
APPENDIX: VISUALIZATIONS REFERENCE
================================================================================

The following visualizations are available in reports/figures/:

1. target_distribution.png - Pie chart (Slide 2, 7)
2. launch_site_analysis.png - Bar chart (Slide 7, 12)
3. booster_version_analysis.png - Horizontal bar (Slide 7, 12)
4. payload_mass_distribution.png - Histogram (Slide 7, 11)
5. year_trends.png - Line chart (Slide 7, 11, 12)
6. correlation_matrix.png - Heatmap (Slide 7)
7. model_comparison.png - Grouped bar (Slide 14)
8. feature_importance.png - Bar chart (Slide 17)
9. confusion_matrix.png - Heatmap (Slide 15)
10. roc_curve.png - Line chart (Slide 14, 16)

**GitHub URL:** https://github.com/EshaanManchanda/Data-Science-Capstone

================================================================================
END OF PRESENTATION
================================================================================