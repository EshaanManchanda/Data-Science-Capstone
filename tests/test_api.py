"""
API Test Script
Test all Flask API endpoints
"""

import requests
import json
import time

BASE_URL = "http://localhost:5000"


def test_health():
    """Test health endpoint"""
    print("Testing /health...")
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'healthy'
    print(f"  ✓ Health check: {data}")


def test_home():
    """Test home endpoint"""
    print("Testing /...")
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    data = response.json()
    assert 'name' in data
    print(f"  ✓ Home: {data['name']}")


def test_predict():
    """Test prediction endpoint"""
    print("Testing /predict...")

    payload = {
        "Launch Site": "KSC LC-39A",
        "Booster Version": "B5",
        "Payload Mass (kg)": 3500,
        "Orbit": "LEO",
        "Grid Fins": True,
        "Legs": True,
        "Year": 2023
    }

    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 'prediction' in data
    print(f"  ✓ Prediction: {data['prediction']['result']}")


def test_predict_batch():
    """Test batch prediction endpoint"""
    print("Testing /predict_batch...")

    payload = {
        "predictions": [
            {
                "Launch Site": "KSC LC-39A",
                "Booster Version": "B5",
                "Payload Mass (kg)": 3500,
                "Orbit": "LEO",
                "Grid Fins": True,
                "Legs": True,
                "Year": 2023
            },
            {
                "Launch Site": "CCAFS LC-40",
                "Booster Version": "F9 v1.0",
                "Payload Mass (kg)": 500,
                "Orbit": "GTO",
                "Grid Fins": True,
                "Legs": True,
                "Year": 2012
            }
        ]
    }

    response = requests.post(f"{BASE_URL}/predict_batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 'predictions' in data
    assert len(data['predictions']) == 2
    print(f"  ✓ Batch: {data['count']} predictions")


def test_model_info():
    """Test model info endpoint"""
    print("Testing /model_info...")
    response = requests.get(f"{BASE_URL}/model_info")
    assert response.status_code == 200
    data = response.json()
    assert 'accuracy' in data
    print(f"  ✓ Model accuracy: {data['accuracy']}")


def test_model_metrics():
    """Test model metrics endpoint"""
    print("Testing /model_metrics...")
    response = requests.get(f"{BASE_URL}/model_metrics")
    assert response.status_code == 200
    data = response.json()
    assert 'models' in data
    print(f"  ✓ Models: {len(data['models'])} models")


def test_data_statistics():
    """Test data statistics endpoint"""
    print("Testing /data_statistics...")
    response = requests.get(f"{BASE_URL}/data_statistics")
    assert response.status_code == 200
    data = response.json()
    assert 'total_launches' in data
    print(f"  ✓ Total launches: {data['total_launches']}")


def test_feature_importance():
    """Test feature importance endpoint"""
    print("Testing /feature_importance...")
    response = requests.get(f"{BASE_URL}/feature_importance")
    assert response.status_code == 200
    data = response.json()
    assert 'features' in data
    print(f"  ✓ Features: {len(data['features'])} features")


def main():
    """Run all tests"""
    print("\n" + "="*50)
    print("🧪 Testing API Endpoints")
    print("="*50 + "\n")

    try:
        test_home()
        test_health()
        test_predict()
        test_predict_batch()
        test_model_info()
        test_model_metrics()
        test_data_statistics()
        test_feature_importance()

        print("\n" + "="*50)
        print("✅ All API tests passed!")
        print("="*50 + "\n")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\nMake sure the Flask API is running:")
        print("  python deployment/flask/app.py")


if __name__ == "__main__":
    main()