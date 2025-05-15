# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

# Convert to DataFrame for better visualization
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
print(df.head())

# Check for missing values
print("Missing Values in Each Column:")
print(df.isnull().sum())

# Visualize the distribution of the target variable
sns.countplot(x='target', data=df)
plt.title('Distribution of Target Variable (0: Benign, 1: Malignant)')
plt.xlabel('Target')
plt.ylabel('Count')
plt.show()


# Split the data into features and target
X = df.drop('target', axis=1)  # Features (all columns except 'target')
y = df['target']              # Target (the 'target' column)

# Split into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Check the distribution of classes in the training set
print("Class distribution in the training set:")
print(y_train.value_counts())


# Set class weights to handle imbalanced data
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
print("\nClass Weights:")
print(class_weights_dict)


# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Build the Random Forest model with class weights
rf_model = RandomForestClassifier(random_state=42, class_weight=class_weights_dict)

# Train the model on the training data
rf_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))


# Import necessary libraries for visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# Import necessary libraries
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Build the XGBoost model with class weights
xgb_model = xgb.XGBClassifier(random_state=42, scale_pos_weight=class_weights_dict[1] / class_weights_dict[0])

# Train the model on the training data
xgb_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the model
print("Confusion Matrix (XGBoost):")
print(confusion_matrix(y_test, y_pred_xgb))

print("\nClassification Report (XGBoost):")
print(classification_report(y_test, y_pred_xgb))

print("Accuracy Score (XGBoost):")
print(accuracy_score(y_test, y_pred_xgb))

# Import necessary libraries for visualization
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Compute the confusion matrix
cm_xgb = confusion_matrix(y_test, y_pred_xgb)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted 0', 'Predicted 1'], 
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix (XGBoost)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()



# Import necessary libraries
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Build the LightGBM model with class weights
lgb_model = lgb.LGBMClassifier(random_state=42, scale_pos_weight=class_weights_dict[1] / class_weights_dict[0])

# Train the model on the training data
lgb_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_lgb = lgb_model.predict(X_test)

# Evaluate the model
print("Confusion Matrix (LightGBM):")
print(confusion_matrix(y_test, y_pred_lgb))

print("\nClassification Report (LightGBM):")
print(classification_report(y_test, y_pred_lgb))

print("Accuracy Score (LightGBM):")
print(accuracy_score(y_test, y_pred_lgb))


# Import necessary libraries for visualization
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Compute the confusion matrix
cm_lgb = confusion_matrix(y_test, y_pred_lgb)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm_lgb, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted 0', 'Predicted 1'], 
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix (LightGBM)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Define a function to evaluate models
def evaluate_model(model, model_name, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of class 1
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification Report
    report = classification_report(y_test, y_pred)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # ROC-AUC Score
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Print results
    print(f"--- Evaluation for {model_name} ---")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    print("Accuracy Score:", accuracy)
    print("ROC-AUC Score:", roc_auc)
    
    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({model_name})')
    plt.legend()
    plt.show()

# Evaluate Random Forest
evaluate_model(rf_model, "Random Forest", X_test, y_test)

# Evaluate XGBoost
evaluate_model(xgb_model, "XGBoost", X_test, y_test)

# Evaluate LightGBM
evaluate_model(lgb_model, "LightGBM", X_test, y_test)




