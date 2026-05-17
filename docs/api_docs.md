# 🔌 API Documentation
## SpaceX Falcon 9 Landing Prediction - Flask REST API

---

## Base URL

```
http://localhost:5000
```

---

## Endpoints

### 1. Home
**GET** `/`

Returns API information.

**Response:**
```json
{
  "name": "SpaceX Falcon 9 Landing Prediction API",
  "version": "1.0.0",
  "status": "running",
  "timestamp": "2026-05-17T12:00:00"
}
```

---

### 2. Health Check
**GET** `/health`

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

### 3. Predict (Single)
**POST** `/predict`

Make a single prediction.

**Request Body:**
```json
{
  "Launch Site": "KSC LC-39A",
  "Booster Version": "B5",
  "Payload Mass (kg): 3500,
  "Orbit": "LEO",
  "Grid Fins": true,
  "Legs": true,
  "Year": 2023
}
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "prediction": 1,
    "probability_success": 0.92,
    "probability_failure": 0.08,
    "result": "SUCCESS"
  },
  "timestamp": "2026-05-17T12:00:00"
}
```

---

### 4. Predict (Batch)
**POST** `/predict_batch`

Make multiple predictions.

**Request Body:**
```json
{
  "predictions": [
    {
      "Launch Site": "KSC LC-39A",
      "Booster Version": "B5",
      "Payload Mass (kg)": 3500,
      "Orbit": "LEO",
      "Grid Fins": true,
      "Legs": true,
      "Year": 2023
    },
    {
      "Launch Site": "CCAFS LC-40",
      "Booster Version": "F9 v1.1",
      "Payload Mass (kg)": 3000,
      "Orbit": "GTO",
      "Grid Fins": true,
      "Legs": true,
      "Year": 2015
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "predictions": [
    {
      "prediction": 1,
      "probability_success": 0.92,
      "probability_failure": 0.08,
      "result": "SUCCESS"
    },
    {
      "prediction": 0,
      "probability_success": 0.35,
      "probability_failure": 0.65,
      "result": "FAILURE"
    }
  ],
  "count": 2,
  "timestamp": "2026-05-17T12:00:00"
}
```

---

### 5. Model Information
**GET** `/model_info`

Get model details and metrics.

**Response:**
```json
{
  "model": "XGBoost Classifier",
  "accuracy": 0.943,
  "precision": 0.95,
  "recall": 0.94,
  "f1_score": 0.94,
  "roc_auc": 0.98,
  "features": [
    "Launch Site",
    "Booster Version",
    "Payload Mass",
    "Orbit",
    "Grid Fins",
    "Legs",
    "Year"
  ]
}
```

---

### 6. Model Metrics
**GET** `/model_metrics`

Get detailed metrics for all trained models.

**Response:**
```json
{
  "models": [
    {
      "name": "Logistic Regression",
      "accuracy": 0.875,
      "precision": 0.89,
      "recall": 0.88,
      "f1_score": 0.88,
      "roc_auc": 0.94
    },
    ...
  ]
}
```

---

### 7. Data Statistics
**GET** `/data_statistics`

Get dataset statistics.

**Response:**
```json
{
  "total_launches": 57,
  "success_rate": 0.75,
  "launch_sites": ["CCAFS LC-40", "KSC LC-39A", "VAFB SLC-4E"],
  "booster_versions": ["F9 v1.0", "F9 v1.1", "F9 FT", "B5"],
  "orbits": ["LEO", "GTO", "SSO", "MEO", "GEO"],
  "year_range": [2010, 2023],
  "payload_mass_stats": {
    "min": 0,
    "max": 5500,
    "mean": 2500
  }
}
```

---

### 8. Feature Importance
**GET** `/feature_importance`

Get feature importance scores.

**Response:**
```json
{
  "features": [
    {"name": "Booster Version", "importance": 0.25},
    {"name": "Launch Site", "importance": 0.20},
    {"name": "Year", "importance": 0.18},
    {"name": "Payload Mass", "importance": 0.15},
    {"name": "Orbit", "importance": 0.12},
    {"name": "Grid Fins", "importance": 0.06},
    {"name": "Legs", "importance": 0.04}
  ]
}
```

---

## Example Usage

### cURL
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"Launch Site": "KSC LC-39A", "Booster Version": "B5", "Payload Mass (kg)": 3500, "Orbit": "LEO", "Grid Fins": true, "Legs": true, "Year": 2023}'
```

### Python
```python
import requests

url = "http://localhost:5000/predict"
data = {
    "Launch Site": "KSC LC-39A",
    "Booster Version": "B5",
    "Payload Mass (kg)": 3500,
    "Orbit": "LEO",
    "Grid Fins": True,
    "Legs": True,
    "Year": 2023
}

response = requests.post(url, json=data)
print(response.json())
```

### JavaScript
```javascript
const response = await fetch('http://localhost:5000/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    "Launch Site": "KSC LC-39A",
    "Booster Version": "B5",
    "Payload Mass (kg)": 3500,
    "Orbit": "LEO",
    "Grid Fins": true,
    "Legs": true,
    "Year": 2023
  })
});
const data = await response.json();
console.log(data);
```

---

## Error Responses

### 400 Bad Request
```json
{
  "error": "Missing required field: Launch Site"
}
```

### 404 Not Found
```json
{
  "error": "Endpoint not found"
}
```

### 500 Internal Server Error
```json
{
  "error": "Internal server error"
}
```

---

## Rate Limiting

Currently no rate limiting. For production, consider adding rate limits.

---

**API Version:** 1.0.0