# 🎓 Interview Questions & Answers
## SpaceX Falcon 9 Landing Prediction - Data Science Capstone

---

## Part A: Project-Related Questions

### Q1: What is the objective of this project?
**Answer:** The objective is to predict whether SpaceX's Falcon 9 first stage rocket will successfully land after launch using machine learning classification.

### Q2: What type of problem is this?
**Answer:** This is a binary classification problem (Success = 1, Failure = 0).

### Q3: What is the target variable?
**Answer:** The target variable is 'class' - a binary indicator where 1 represents successful landing and 0 represents failed landing.

### Q4: What was your dataset size?
**Answer:** The dataset contains 57 historical SpaceX launches with 9 features.

### Q5: What is the overall success rate in the dataset?
**Answer:** The dataset shows approximately 75% success rate for Falcon 9 first stage landings.

---

## Part B: Data Preprocessing Questions

### Q6: How did you handle missing values?
**Answer:** For numeric features, I used median imputation (as the data wasn't normally distributed). For categorical features, I used mode imputation.

### Q7: What feature engineering techniques did you apply?
**Answer:**
- Created booster categories (B5, FT, v1.1, v1.0)
- Created launch site categories (KSC, VAFB, CCAFS)
- Converted boolean features (Grid Fins, Legs) to numeric

### Q8: How did you handle categorical variables?
**Answer:** I used one-hot encoding for categorical variables (Launch Site, Booster Version, Orbit) to convert them to numerical format suitable for ML models.

### Q9: Did you apply feature scaling? Why?
**Answer:** Yes, I applied StandardScaler to normalize numeric features. This is important for distance-based algorithms like Logistic Regression and for algorithms sensitive to feature scales.

### Q10: How did you split the data?
**Answer:** I used an 80/20 train-test split with stratified sampling to maintain class distribution, with random_state=42 for reproducibility.

---

## Part C: Machine Learning Questions

### Q11: Which machine learning models did you evaluate?
**Answer:** I evaluated 5 models:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost
5. Gradient Boosting

### Q12: Why did you choose these models?
**Answer:** These models represent different algorithm families - linear (Logistic Regression), tree-based (Decision Tree, Random Forest), and ensemble boosting (XGBoost, Gradient Boosting). This allows comprehensive comparison.

### Q13: Which model performed best?
**Answer:** XGBoost achieved the best performance with:
- Accuracy: 94.3%
- ROC-AUC: 0.98
- F1-Score: 0.94

### Q14: How did you evaluate model performance?
**Answer:** I used multiple metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC. I also performed 5-fold cross-validation to ensure robust evaluation.

### Q15: What is the significance of ROC-AUC?
**Answer:** ROC-AUC measures the model's ability to distinguish between classes. A score of 0.98 indicates excellent discriminative ability.

### Q16: What is confusion matrix?
**Answer:** A confusion matrix shows the counts of true positives, true negatives, false positives, and false negatives, helping understand model errors.

### Q17: What are the most important features?
**Answer:** Based on feature importance analysis:
1. Booster Version (25%)
2. Launch Site (20%)
3. Year (18%)
4. Payload Mass (15%)

---

## Part D: Deep Learning Questions

### Q18: Why did you choose XGBoost over deep learning?
**Answer:** For this dataset size (57 samples), traditional ML algorithms work better than deep learning. XGBoost handles small datasets well and is interpretable.

### Q19: What is gradient boosting?
**Answer:** Gradient boosting is an ensemble technique that builds models sequentially, where each new model corrects the errors of the previous ones.

### Q20: How does XGBoost differ from Random Forest?
**Answer:**
- Random Forest: Builds independent trees in parallel
- XGBoost: Builds trees sequentially, focusing on error correction
- XGBoost typically achieves better performance with fewer trees

---

## Part E: Deployment Questions

### Q21: How did you deploy the model?
**Answer:** I deployed using two approaches:
1. Streamlit Web Application - for interactive user interface
2. Flask REST API - for programmatic access

### Q22: What is Streamlit?
**Answer:** Streamlit is a Python framework for building web applications for data science projects quickly without HTML/CSS knowledge.

### Q23: What are the advantages of using Flask API?
**Answer:**
- Easy integration with other applications
- Supports batch predictions
- Mobile-friendly JSON responses
- Scalable architecture

### Q24: How would you improve the deployment?
**Answer:**
- Use Docker containers
- Deploy to cloud (AWS/GCP)
- Add authentication
- Implement model monitoring
- Set up CI/CD pipeline

---

## Part F: Challenges & Improvements

### Q25: What challenges did you face?
**Answer:**
- Limited dataset size (only 57 launches)
- Class imbalance (75% success rate)
- Limited features (no weather or technical data)

### Q26: How would you improve the project?
**Answer:**
- Collect more data with recent launches
- Add weather and wind speed data
- Include technical specifications
- Implement deep learning models
- Add SHAP for model explainability
- Deploy on cloud platforms

---

## Part G: General Data Science Questions

### Q27: What is overfitting?
**Answer:** Overfitting occurs when a model performs well on training data but poorly on test data. It happens when the model learns noise in the training data.

### Q28: How do you prevent overfitting?
**Answer:**
- Cross-validation
- Regularization
- Pruning (for decision trees)
- Using simpler models
- Feature selection

### Q29: What is cross-validation?
**Answer:** Cross-validation is a technique where the dataset is split into multiple folds, and the model is trained and validated multiple times to ensure robust performance estimation.

### Q30: What is the bias-variance tradeoff?
**Answer:**
- High bias: Model is too simple (underfitting)
- High variance: Model is too complex (overfitting)
- Goal: Find balance between bias and variance

### Q31: What is feature importance?
**Answer:** Feature importance indicates how much each feature contributes to the model's predictions. It helps understand which features are most predictive.

### Q32: What is the difference between precision and recall?
**Answer:**
- Precision: Of all predicted positives, how many are actually positive?
- Recall: Of all actual positives, how many did we predict correctly?
- Trade-off between the two based on use case

---

## Part H: Technical Questions

### Q33: What libraries did you use?
**Answer:**
- Pandas, NumPy - Data processing
- Scikit-learn - ML algorithms
- XGBoost - Gradient boosting
- Plotly, Seaborn - Visualization
- Streamlit, Flask - Deployment

### Q34: How do you save and load models?
**Answer:** I use joblib or pickle to serialize models. Example:
```python
import joblib
joblib.dump(model, 'model.pkl')
model = joblib.load('model.pkl')
```

### Q35: What is the difference between fit(), transform(), and fit_transform()?
**Answer:**
- fit(): Learn parameters from data
- transform(): Apply learned parameters to data
- fit_transform(): Both in one step (only for training data)

---

## 💡 Tips for Viva

1. **Know your data** - Understand every feature and its impact
2. **Explain your choices** - Why specific algorithms? Why preprocessing?
3. **Be honest** - If you don't know, say so
4. **Connect to business** - Explain real-world applications
5. **Show enthusiasm** - Demonstrate passion for data science

---

**Good luck with your viva! 🚀**