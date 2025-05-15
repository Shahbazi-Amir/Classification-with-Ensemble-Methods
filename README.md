# 🎯 Final Report: Breast Cancer Classification with Ensemble Methods

This project demonstrates a complete machine learning pipeline for classifying breast cancer tumors as benign or malignant using the Breast Cancer Wisconsin dataset.

## 🧠 Models Used

- **Random Forest**
- **XGBoost**
- **LightGBM**

All models were trained on the Breast Cancer dataset from `sklearn.datasets`.

## 📦 Key Steps Performed

1. **Data Loading & Preprocessing**
   - Dataset loaded using `load_breast_cancer()`
   - Converted to DataFrame for better visualization
   - Checked for missing values
   - Visualized target distribution

2. **Train/Test Split**
   - Data split into 80% train and 20% test
   - Class weights calculated to handle imbalance:
     ```python
     class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
     ```

3. **Model Training**
   - Trained all three models with class weights
   - Example:
     ```python
     rf_model = RandomForestClassifier(random_state=42, class_weight=class_weights_dict)
     rf_model.fit(X_train, y_train)
     ```

4. **Evaluation Metrics**
   - Accuracy
   - Confusion Matrix
   - Classification Report
   - ROC-AUC Score
   - ROC Curve visualization

5. **Model Saving**
   - All models saved using `joblib.dump()`:
     ```python
     joblib.dump(xgb_model, "models/xgboost_model.pkl")
     ```

6. **API Creation**
   - A Flask-based API was created in `api/api_app.py`
   - Accepts input via:
     - ✅ GET request (`?features=...`)
     - ✅ POST request (JSON body)

7. **Testing the API**
   - Successfully tested using:
     - ✅ Python script (`testapi.py`)
     - ✅ Browser / Postman
     - ✅ Direct cURL command

---

## ✅ Conclusion

The XGBoost model achieved the highest accuracy and was successfully deployed via a RESTful API for real-time prediction.