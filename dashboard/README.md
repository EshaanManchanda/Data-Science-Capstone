# Power BI Dashboard Structure
## SpaceX Falcon 9 Landing Prediction

---

## 📊 Dashboard Overview

This document outlines the structure for creating a Power BI dashboard to visualize the SpaceX Falcon 9 landing prediction project.

---

## 📁 Required Data Connections

### 1. Main Dataset
- **Source**: `data/raw/spacex_data.csv`
- **Fields**:
  - Flight Number
  - Launch Site
  - Booster Version
  - Payload Mass (kg)
  - Grid Fins
  - Legs
  - Year
  - Class (Target)

### 2. Model Results
- **Source**: Manual import or API
- **Fields**: Model name, Accuracy, Precision, Recall, F1-Score, ROC-AUC

### 3. Feature Importance
- **Source**: Model output
- **Fields**: Feature name, Importance score

---

## 📑 Recommended Pages

### Page 1: Executive Summary
- Key metrics (Total launches, Success rate)
- Pie chart for outcome distribution
- Year trend line chart
- Top 3 performing launch sites

### Page 2: Launch Site Analysis
- Bar chart: Launches by site
- Stacked bar: Success vs Failure by site
- Map visualization (if geographic data available)
- KPIs: Best site, worst site

### Page 3: Booster Performance
- Horizontal bar: Success rate by booster version
- Timeline: Booster usage over years
- Comparison: Old vs new boosters

### Page 4: Payload Analysis
- Scatter: Payload mass vs outcome
- Box plots: Payload distribution by outcome
- Histogram: Payload mass distribution

### Page 5: Model Performance
- Comparison table: All models
- Bar chart: Metrics comparison
- ROC curve visualization
- Confusion matrix heatmap

### Page 6: Predictions
- Interactive prediction form
- Gauge chart for probability
- Recent predictions log

---

## 📈 Recommended Visualizations

| Visualization | Type | Purpose |
|---------------|------|---------|
| Success Rate | KPI Card | Quick metric |
| Outcome Distribution | Pie Chart | Proportions |
| Launches by Year | Line Chart | Trends |
| Site Performance | Bar Chart | Comparison |
| Payload Analysis | Scatter Plot | Correlation |
| Model Metrics | Clustered Bar | Comparison |
| ROC Curve | Line Chart | Model quality |
| Confusion Matrix | Matrix/Table | Errors |

---

## 🔧 Power BI Tips

### 1. Data Transformation (Power Query)
```
- Remove unnecessary columns
- Create calculated columns:
  - Year extraction
  - Success/Failure label
  - Booster Category
- Group by for aggregations
```

### 2. DAX Measures
```DAX
Success Rate = DIVIDE(
    CALCULATE(COUNTROWS(Data), Data[Class] = 1),
    COUNTROWS(Data)
)

Total Launches = COUNTROWS(Data)

Avg Payload = AVERAGE(Data[Payload Mass (kg)])
```

### 3. Slicers
- Year (slider)
- Launch Site (dropdown)
- Booster Version (multi-select)
- Outcome (radio)

### 4. Filters
- Page-level filters
- Visual-level filters
- Cross-filtering enabled

---

## 🎨 Color Scheme

| Element | Color |
|---------|-------|
| Success | #4ECDC4 (Teal) |
| Failure | #FF6B6B (Red) |
| Neutral | #3498DB (Blue) |
| Accent | #2ECC71 (Green) |
| Background | #FFFFFF or #1E1E1E |

---

## 📤 Export Options

- Power BI Service publish
- PDF export for reports
- PNG export for slides
- PowerPoint embed

---

## 🔗 API Integration (Optional)

Connect to Flask API for real-time predictions:
- Endpoint: `POST /predict`
- Refresh: On-demand or scheduled

---

**Note**: This structure can also be adapted for Tableau or other BI tools.