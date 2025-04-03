# -*- coding: utf-8 -*-
"""Hardi Makwana_Predicting Disease Outbreaks and Personalized Risk Scoring Using Machine Learning.ipynb

# **Predicting Disease Outbreaks and Personalized Risk Scoring Using Machine Learning**

# **Parkinson's Disease**

# Import Necessary Libraries
"""

# Step 1: Import Necessary Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin

"""# Define the Problem and Load Dataset
Define the objective: Predict Parkinson's disease presence based on features.
"""

# Step 2: Define the Problem and Load Dataset
# Define the objective: Predict Parkinson's disease presence based on features.
parkinsons_data = pd.read_csv('/content/drive/MyDrive/Prediction_of_Disease_Outbreaks/parkinsons.csv')

# Display dataset information
print("--- Dataset Information ---")
print(parkinsons_data.info())
print(parkinsons_data.describe())

# Check for missing values
print("\nMissing Values:\n", parkinsons_data.isnull().sum())

# Drop irrelevant columns
parkinsons_data = parkinsons_data.drop(columns=['name'])

"""# Perform EDA"""

# Step 3: Perform EDA
# Correlation matrix
# Exploratory Data Analysis
def correlation_analysis(data, dataset_name):
    plt.figure(figsize=(16, 14))  # Increase the size of the figure
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f", annot_kws={"size": 12},
                cbar_kws={"shrink": 0.8}, linewidths=1.5)  # Adjust linewidths for better visibility of boxes
    plt.title(f"Correlation Matrix for {dataset_name}", fontsize=18)  # Increase title font size
    plt.xticks(fontsize=12)  # Increase x-axis tick font size
    plt.yticks(fontsize=12)  # Increase y-axis tick font size
    plt.show()
    print("\n")

# Call the function for each dataset
correlation_analysis(parkinsons_data, "Parkinson’s Disease")

# Distribution of the target variable
sns.countplot(x='status', data=parkinsons_data)
plt.title("Distribution of Parkinson's Disease (1: Has Disease, 0: Healthy)")
plt.show()

"""# Split Data into Training and Test Sets (80-20 Split)"""

# Step 4: Split Data into Training and Test Sets (80-20 Split)
# Define features and target variable
X = parkinsons_data.drop(columns=['status'])
y = parkinsons_data['status']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

"""# Train and Evaluate Models"""

# Step 5: Train and Evaluate Models
# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_preds)
print(f"\nRandom Forest Accuracy: {rf_accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, rf_preds))

# XGBoost Wrapper for Compatibility
class SklearnXGBWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.model = XGBClassifier(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

# Initialize XGBoost Wrapper
xgb_model = SklearnXGBWrapper(random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_preds)
print(f"\nXGBoost Accuracy: {xgb_accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, xgb_preds))

"""# Perform Feature Engineering"""

# Step 6: Perform Feature Engineering
# Feature importance analysis using XGBoost
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title("Feature Importance (XGBoost)")
plt.show()

"""# Use Cross-Validation to Check Model Reliability"""

# Step 7: Use Cross-Validation to Check Model Reliability
cv_scores_rf = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"Random Forest Cross-Validation Accuracy: {cv_scores_rf.mean() * 100:.2f}%")

cv_scores_xgb = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"XGBoost Cross-Validation Accuracy: {cv_scores_xgb.mean() * 100:.2f}%")

"""# Final Evaluation on Test Set"""

# Step 8: Final Evaluation on Test Set
if rf_accuracy > xgb_accuracy:
    best_model = "Random Forest"
    final_preds = rf_model.predict(X_test)
    final_accuracy = rf_accuracy
else:
    best_model = "XGBoost"
    final_preds = xgb_model.predict(X_test)
    final_accuracy = xgb_accuracy

print(f"\nBest Model: {best_model}")
print(f"Final Test Set Accuracy: {final_accuracy * 100:.2f}%")
print("Confusion Matrix:\n", confusion_matrix(y_test, final_preds))
print("Final Classification Report:\n", classification_report(y_test, final_preds))

"""# Predict on new input data"""

# Step 9: Predict on new input data
input_data = (197.07600, 206.89600, 192.05500, 0.00289, 0.00001, 0.00166, 0.00168, 0.00498, 0.01098, 0.09700, 0.00563, 0.00680, 0.00802, 0.01689, 0.00339, 26.77500, 0.422229, 0.741367, -7.348300, 0.177551, 1.743867, 0.085569)

# Convert the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the numpy array for prediction
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardize the input data using the same scaler
input_data_scaled = scaler.transform(input_data_reshaped)

# Use the best model for prediction (choose based on earlier evaluation)
prediction = xgb_model.predict(input_data_scaled) if xgb_accuracy > rf_accuracy else rf_model.predict(input_data_scaled)

# Output the result
if prediction[0] == 0:
    print("The Person does not have Parkinson's Disease")
else:
    print("The Person has Parkinson's Disease")

"""# Save Model"""

# Step 10: Save the Model
model_filename = '/content/drive/MyDrive/parkinsons_model.sav'
pickle.dump(rf_model, open(model_filename, 'wb'))  # Save your chosen model
print(f"Model saved to {model_filename}")

"""# **Diabetes Disease**

# Import Necessary Libraries
"""

# Step 1: Import Necessary Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

"""# Define the Problem and Load Dataset"""

# Step 2: Define the Problem and Load Dataset
# Define the objective: Predict diabetes presence based on features.
diabetes_data = pd.read_csv('/content/drive/MyDrive/Prediction_of_Disease_Outbreaks/diabetes.csv')

# Display dataset information
print("--- Dataset Information ---")
print(diabetes_data.info())
print(diabetes_data.describe())

# Check for missing values
print("\nMissing Values:\n", diabetes_data.isnull().sum())

"""# Perform EDA"""

# Step 3: Perform EDA
# Correlation matrix
def correlation_analysis(data, dataset_name):
    plt.figure(figsize=(16, 14))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f", annot_kws={"size": 10},
                cbar_kws={"shrink": 0.8}, linewidths=0.5, square=True)
    plt.title(f"Correlation Matrix for {dataset_name}", fontsize=16)
    plt.xticks(fontsize=10, rotation=45, ha='right')
    plt.yticks(fontsize=10, rotation=0)
    plt.tight_layout()
    plt.show()
    print("\n")

correlation_analysis(diabetes_data, "Diabetes")

# Distribution of the target variable
sns.countplot(x='Outcome', data=diabetes_data)
plt.title("Distribution of Diabetes (1: Diabetic, 0: Non-Diabetic)")
plt.show()

"""# Split Data into Training and Test Sets (80-20 Split)"""

# Step 4: Split Data into Training and Test Sets (80-20 Split)
# Define features and target variable
X = diabetes_data.drop(columns=['Outcome'])
y = diabetes_data['Outcome']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

"""# Train and Evaluate Random Forest Model"""

# Step 5: Train and Evaluate Random Forest Model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_preds)
print(f"\nRandom Forest Accuracy: {rf_accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, rf_preds))

# Evaluate Random Forest with Cross-Validation
cv_scores_rf = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
rf_cv_accuracy = cv_scores_rf.mean() * 100
print(f"Random Forest Cross-Validation Accuracy: {rf_cv_accuracy:.2f}%")

"""# Train and Evaluate SVM Model"""

# Step 6: Train and Evaluate SVM Model
svm_model = SVC(kernel='rbf', random_state=42, probability=True)
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_preds)
print(f"\nSVM Accuracy: {svm_accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, svm_preds))

# Evaluate SVM with Cross-Validation
cv_scores_svm = cross_val_score(svm_model, X_train, y_train, cv=5, scoring='accuracy')
svm_cv_accuracy = cv_scores_svm.mean() * 100
print(f"SVM Cross-Validation Accuracy: {svm_cv_accuracy:.2f}%")

"""# Choose the Best Model"""

# Step 7: Choose the Best Model
if rf_cv_accuracy > svm_cv_accuracy:
    best_model = rf_model
    best_model_name = "Random Forest"
    best_accuracy = rf_cv_accuracy
else:
    best_model = svm_model
    best_model_name = "SVM"
    best_accuracy = svm_cv_accuracy

print(f"\nBest Model: {best_model_name}")
print(f"Best Cross-Validation Accuracy: {best_accuracy:.2f}%")

"""# Final Evaluation on Test Set with Best Model"""

# Step 8: Final Evaluation on Test Set with Best Model
final_preds = best_model.predict(X_test)
final_accuracy = accuracy_score(y_test, final_preds)
print(f"\nFinal Test Set Accuracy: {final_accuracy * 100:.2f}%")
print("Confusion Matrix:\n", confusion_matrix(y_test, final_preds))
print("Final Classification Report:\n", classification_report(y_test, final_preds))

"""# Predict on new input data"""

# Step 9: Predict on new input data
input_data = (6, 148, 72, 35, 0, 33.6, 0.627, 50)  # Example input

# Convert the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the numpy array for prediction
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardize the input data using the same scaler
input_data_scaled = scaler.transform(input_data_reshaped)

# Use the best model for prediction
prediction = best_model.predict(input_data_scaled)

# Output the result
if prediction[0] == 0:
    print("The Person does not have Diabetes")
else:
    print("The Person has Diabetes")

"""# Save Model"""

import pickle

# Step 10: Save the Model
model_filename = '/content/drive/MyDrive/diabetes_model.sav'
pickle.dump(rf_model, open(model_filename, 'wb'))  # Save your chosen model
print(f"Model saved to {model_filename}")

"""# **Heart Disease**

# Import Necessary Libraries
"""

# Step 1: Import Necessary Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

"""# Define the Problem and Load Dataset"""

# Step 2: Define the Problem and Load Dataset
# Define the objective: Predict heart disease presence based on features.
heart_data = pd.read_csv('/content/drive/MyDrive/Prediction_of_Disease_Outbreaks/heart.csv')

# Display dataset information
print("--- Dataset Information ---")
print(heart_data.info())
print(heart_data.describe())

# Check for missing values
print("\nMissing Values:\n", heart_data.isnull().sum())

"""# Perform EDA"""

# Step 3: Perform EDA
# Correlation matrix
def correlation_analysis(data, dataset_name):
    plt.figure(figsize=(16, 14))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f", annot_kws={"size": 10},
                cbar_kws={"shrink": 0.8}, linewidths=0.5, square=True)
    plt.title(f"Correlation Matrix for {dataset_name}", fontsize=16)
    plt.xticks(fontsize=10, rotation=45, ha='right')
    plt.yticks(fontsize=10, rotation=0)
    plt.tight_layout()
    plt.show()
    print("\n")

correlation_analysis(heart_data, "Heart Disease")

# Distribution of the target variable
sns.countplot(x='target', data=heart_data)
plt.title("Distribution of Heart Disease (1: Has Disease, 0: Healthy)")
plt.show()

"""# Split Data into Training and Test Sets (80-20 Split)"""

# Step 4: Split Data into Training and Test Sets (80-20 Split)
# Define features and target variable
X = heart_data.drop(columns=['target'])
y = heart_data['target']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

"""# Train and Evaluate Random Forest Model"""

# Step 5: Train and Evaluate Random Forest Model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_preds)
print(f"\nRandom Forest Accuracy: {rf_accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, rf_preds))

# Evaluate Random Forest with Cross-Validation
cv_scores_rf = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
rf_cv_accuracy = cv_scores_rf.mean() * 100
print(f"Random Forest Cross-Validation Accuracy: {rf_cv_accuracy:.2f}%")

"""# Train and Evaluate SVM Model"""

# Step 6: Train and Evaluate SVM Model
svm_model = SVC(kernel='rbf', random_state=42, probability=True)
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_preds)
print(f"\nSVM Accuracy: {svm_accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, svm_preds))

# Evaluate SVM with Cross-Validation
cv_scores_svm = cross_val_score(svm_model, X_train, y_train, cv=5, scoring='accuracy')
svm_cv_accuracy = cv_scores_svm.mean() * 100
print(f"SVM Cross-Validation Accuracy: {svm_cv_accuracy:.2f}%")

"""# Choose the Best Model"""

# Step 7: Choose the Best Model
if rf_cv_accuracy > svm_cv_accuracy:
    best_model = rf_model
    best_model_name = "Random Forest"
    best_accuracy = rf_cv_accuracy
else:
    best_model = svm_model
    best_model_name = "SVM"
    best_accuracy = svm_cv_accuracy

print(f"\nBest Model: {best_model_name}")
print(f"Best Cross-Validation Accuracy: {best_accuracy:.2f}%")

"""# Final Evaluation on Test Set with Best Model"""

# Step 8: Final Evaluation on Test Set with Best Model
final_preds = best_model.predict(X_test)
final_accuracy = accuracy_score(y_test, final_preds)
print(f"\nFinal Test Set Accuracy: {final_accuracy * 100:.2f}%")
print("Confusion Matrix:\n", confusion_matrix(y_test, final_preds))
print("Final Classification Report:\n", classification_report(y_test, final_preds))

"""# Predict on new input data"""

# Step 9: Predict on new input data
input_data = (57, 1, 2, 130, 236, 0, 0, 174, 0, 0.0, 1, 1, 3)  # Example input

# Convert the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the numpy array for prediction
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardize the input data using the same scaler
input_data_scaled = scaler.transform(input_data_reshaped)

# Use the best model for prediction
prediction = best_model.predict(input_data_scaled)

# Output the result
if prediction[0] == 0:
    print("The Person does not have Heart Disease")
else:
    print("The Person has Heart Disease")

"""# Save Model"""

import pickle

# Step 10: Save the Model
model_filename = '/content/drive/MyDrive/heart_model.sav'
pickle.dump(rf_model, open(model_filename, 'wb'))  # Save your chosen model
print(f"Model saved to {model_filename}")
