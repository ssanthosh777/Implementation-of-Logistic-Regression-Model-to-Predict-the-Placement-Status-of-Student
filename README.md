# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
Step 1: Import Required Libraries
Step 2: Load the Dataset
Step 3: Copy Data & Drop Unwanted Columns
Step 4: Check Data Quality
Step 5: Encode Categorical Variables
Step 6: Define Features (X) and Target (y)
Step 7: Split into Training and Testing Sets
Step 8: Build and Train Logistic Regression Model
Step 9: Make Predictions
Step 10: Evaluate the Model
Step 11: Predict for a New Student
```
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SANTHOSH S
RegisterNumber: 212224100052
*/
```
```python
# 1. Import Required Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 2. Load the Dataset
# NOTE: Change the file path if your CSV is in a different location
data = pd.read_csv("Placement_Data.csv")
# View first 5 rows
print("First 5 rows of the dataset:")
print(data.head())

# 3. Create a Copy and Drop Unwanted Columns
data1 = data.copy()

# Dropping 'sl_no' (serial number) and 'salary' (not needed for predicting placement)
data1 = data1.drop(["sl_no", "salary"], axis=1)

print("\nData after dropping 'sl_no' and 'salary':")
print(data1.head())

# 4. Check for Missing and Duplicate Values
print("\nChecking for missing values (True = missing):")
print(data1.isnull().any())

print("\nNumber of duplicate rows:")
print(data1.duplicated().sum())

# 5. Encode Categorical Variables using LabelEncoder
# Columns that are categorical (object type)
cat_cols = ["gender", "ssc_b", "hsc_b", "hsc_s", 
            "degree_t", "workex", "specialisation", "status"]

le = LabelEncoder()

for col in cat_cols:
    data1[col] = le.fit_transform(data1[col])

print("\nData after Label Encoding:")
print(data1.head())

# 6. Define Features (X) and Target (y)
# X = all columns except 'status'
X = data1.iloc[:, :-1]
# y = 'status' column
y = data1["status"]

print("\nFeatures (X) sample:")
print(X.head())

print("\nTarget (y) sample:")
print(y.head())

# 7. Split the Dataset into Training and Testing Sets
# test_size=0.2 → 20% test data, 80% training data
# random_state=0 → same split every time (for reproducibility)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

print("\nTraining and testing shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# 8. Create and Train the Logistic Regression Model
# solver='liblinear' works well for small datasets
lr = LogisticRegression(solver="liblinear")

# Train the model
lr.fit(X_train, y_train)

# 9. Make Predictions on the Test Set
y_pred = lr.predict(X_test)

print("\nPredicted values (y_pred):")
print(y_pred)

# 10. Evaluate Model Performance
# Accuracy: percentage of correctly predicted labels
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
# Classification Report: precision, recall, F1-score
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 11. Predict Placement for a New Student
# Order of features must match X columns:
# ['gender', 'ssc_p', 'ssc_b', 'hsc_p', 'hsc_b', 'hsc_s',
#  'degree_p', 'degree_t', 'workex', 'etest_p', 'specialisation', 'mba_p']

# Example student data (after encoding categorical values manually)
# NOTE: These categorical numeric codes depend on how LabelEncoder encoded them.
# Here we assume:
# gender: 1 (e.g., Male)
# ssc_p: 80
# ssc_b: 1
# hsc_p: 90
# hsc_b: 1
# hsc_s: 1
# degree_p: 90
# degree_t: 1
# workex: 0 (No work experience)
# etest_p: 85
# specialisation: 1
# mba_p: 85

new_student = [[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]]

new_prediction = lr.predict(new_student)

print("\nPrediction for new student (0 = Not Placed, 1 = Placed):")
print(new_prediction[0])
```

## Output:
<img width="725" height="320" alt="Screenshot 2026-11-18 101806" src="https://github.com/user-attachments/assets/1dd27bfc-3dbf-4e75-af62-a37d3e6d6a71" />
<img width="740" height="325" alt="Screenshot 2026-11-18 101810" src="https://github.com/user-attachments/assets/90e208f0-23ab-4de9-89c8-ca8cca2619ed" />
<img width="507" height="401" alt="Screenshot 2026-11-18 101815" src="https://github.com/user-attachments/assets/47bc62ee-dcec-49b8-9fe5-ef14b4a570b2" />
<img width="768" height="335" alt="Screenshot 2026-11-18 101821" src="https://github.com/user-attachments/assets/d2169c75-cd52-4f9b-9e90-2399d713b804" />
<img width="755" height="480" alt="Screenshot 2026-11-18 101830" src="https://github.com/user-attachments/assets/8396a3b7-b528-43d5-ae6f-a280943ab901" />
<img width="315" height="130" alt="Screenshot 2026-11-18 101835" src="https://github.com/user-attachments/assets/6f4bbf18-5aee-44d3-9f56-bc4b9b1776e4" />
<img width="410" height="98" alt="Screenshot 2026-11-18 101840" src="https://github.com/user-attachments/assets/28c0b845-a38d-4004-94a4-2c390ebf922a" />
<img width="742" height="80" alt="Screenshot 2026-11-18 101845" src="https://github.com/user-attachments/assets/01f8f89c-a4eb-4c78-b2c0-151892b22440" />
<img width="587" height="263" alt="Screenshot 2026-11-18 101849" src="https://github.com/user-attachments/assets/11130bb2-4046-45bc-aec2-c4b13ef3b62e" />
<img width="715" height="105" alt="Screenshot 2026-11-18 103209" src="https://github.com/user-attachments/assets/542a61d2-bdb6-4a9c-91e0-d654475c6ffc" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
