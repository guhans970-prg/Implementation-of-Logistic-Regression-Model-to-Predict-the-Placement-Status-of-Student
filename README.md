# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the placement dataset and remove unnecessary columns such as serial number and salary. 2.Convert all categorical attributes into numerical values using label encoding. 3.Split the dataset into training and testing sets using train–test split. 4.Train a Logistic Regression model using the training data. 5.Predict placement status and evaluate the model using accuracy and confusion matrix.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: GUHAN S
RegisterNumber: 212225240044
*/
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
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)
<img width="1426" height="330" alt="695975665-a0e606ec-5fa1-4bcd-bf38-5fda86611733" src="https://github.com/user-attachments/assets/f92ffadf-e813-4a35-b600-e0e2e42727dd" />
<img width="1425" height="322" alt="695975966-bd37cd65-489e-4375-a774-5bbe74efafb3" src="https://github.com/user-attachments/assets/2b4eb739-d662-4d81-99ba-d82910f66e24" />
<img width="1427" height="432" alt="695976233-af69fdd8-aed7-4b7a-be2d-bdfdf68f77b9" src="https://github.com/user-attachments/assets/f0ffcd3f-6972-45fa-8625-e3cc934ab996" />
<img width="1423" height="332" alt="695976506-08e3059e-f455-403b-8fb2-481e66a2a7ca" src="https://github.com/user-attachments/assets/18478e26-3938-4b99-8413-ec1b0b22d9c0" />
<img width="1427" height="507" alt="695976802-78d1a079-ccd4-4448-9f62-0dee6ec1148e" src="https://github.com/user-attachments/assets/7859f9d3-8381-48ab-b879-2c1fa85a42b2" />
<img width="1417" height="157" alt="695977084-f1a314fc-3194-48e1-9143-49630b3a26e1" src="https://github.com/user-attachments/assets/41e2d67c-58c8-4d61-aa01-453c360b45a7" />

<img width="1423" height="110" alt="695977314-90cf9dda-c3ea-4782-8655-dc233b10f079" src="https://github.com/user-attachments/assets/bf538d14-f776-43b8-8fbc-bc84f5ca5c57" />
<img width="1422" height="161" alt="695978234-8446b4de-dde2-46aa-a916-63c965c06f50" src="https://github.com/user-attachments/assets/10e7b60b-407e-4448-8bb9-7ad07471e9f3" />
#RESULT:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
