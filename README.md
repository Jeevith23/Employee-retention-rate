# **Employee retention rate prediction**

## Abstract

We live in a time period where the past saw a sudden increase in employees in the IT sector due to the covid pandemic and the future where recession is unavoidable due to over employment. Still, the attrition rates have not affected as much as predicted due to recession. Employee attrition is a major problem faced by companies in the present years. In the perspective of the employer, despite employees having a history of multiple companies, employers are ready to recruit them for the experience of the specific employee, even if their loyalty is doubted. This research study aims to do a study on the employee attrition rates in the recent past using machine learning algorithms. The reason for employees leaving is analyzed considering factors like Designation , age, work environment, work-life balance, etc. The proposed work yields a significant performance in predicting the employee attrition. By using feature selection and exploring the different ensemble classifiers accuracy of 80% was achieved using 6 features.

---
![Employee Retention](https://github.com/Jeevith23/Employee-retention-rate/assets/139576422/6c19e103-0a16-4b34-bc1f-e72ccacaaf23)

---
## Introduction

- Voluntary or involuntary departure of a working employee from a company.
- The majority of people eventually move on, or the corporation forcibly terminates them, forcing them to do so. 
- When your company is struggling, it may be simpler to keep employees who leave on their own rather than to eliminate employment. 
- The attrition rate has changed through time as a result of growing competition and stricter requirements for employee proficiency. 
- Employee recruitment and training are very expensive processes. Companies must look for, hire, and train new staff. 
- Losing experienced employees, especially high performers, is challenging to manage and has a detrimental impact on an organization's performance and success. 
- The study focuses on the elements that could influence the employee attrition rate. 
 ---

## Attribute Description

![Attribute Description](https://github.com/Jeevith23/Employee-retention-rate/assets/139576422/58374db8-23b6-4042-a7dd-ad8570583ea7)


---
## Steps to upload Excel file in Google collab

Step 1:
![step1](https://github.com/Jeevith23/Employee-retention-rate/assets/139576422/1ab1a4d3-7223-42b6-ae06-ef0d5817d266)

Step 2:
![step2](https://github.com/Jeevith23/Employee-retention-rate/assets/139576422/6a6338a3-85f5-49af-81c0-cfe0c771489b)

Step 3:
![step3](https://github.com/Jeevith23/Employee-retention-rate/assets/139576422/9d475a92-6214-4ed6-a461-7ede57de1ce9)


---
## *Work Pipeline*

![WORKPIPELINE](https://github.com/Jeevith23/Employee-retention-rate/assets/139576422/94f18651-e3bc-4582-bf5a-719e3c67a5e9)


---
## Data Preprocessing

- Removing the unwanted columns/fields in the dataset.
- Checking for Null values.
- Fill the Null values.
- Data Type – `Numerical` – Use Mean / Median
- Data Type – `Text` – Mode
- Change the `Categorical` values to numerical.
- `Labelencoder`
- Word to vector

``` python3
X = data[['Age', 'Workperiod in Months', 'Gender', 'Department', 'Designation']]
y = data['Retention']

# Encoding the X features
X_encoded = pd.get_dummies(X)

# Encoding the y feature
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
print(y)

```
---

## Splitting the data for training and testing

``` python3
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2,random_state=42)#,stratify = y

```

Here the features in X are the `Independednt` features which help us to determine the `Target` feature y -(Retention).

---

## Choose the Machine Learning Model

For the above features and the target variable we can use the following model,
- GradientBoostingClassifier – 0.794
- LogisticRegression – 0.686
- XGBClassifier – 0.8


Hence here we have high accuracy in `XGB Classifier`. 

## *Using the different Models*

1. GradientBoostingClassifier

``` python3
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
```
2. LogisticRegression

``` python3
from sklearn.linear_model import LogisticRegression
# Create an instance of LogisticRegression
model = LogisticRegression()

# Train the logistic regression model
model.fit(X_train, y_train)

# Make predictions on the test set using logistic regression
y_pred = model.predict(X_test)

# Calculate accuracy for logistic regression
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy (Logistic Regression):', accuracy)
```

3.  XGBClassifier

``` pyhton3
# Create an instance of XGBClassifier
model5 = xgb.XGBClassifier()

# Train the model
model5.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model5.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
```
---
## About XGB Classifier
- XGBClassifier uses an ensemble of weak prediction models called decision trees. It builds trees sequentially, where each subsequent tree tries to correct the mistakes made by the previous trees. This process is known as gradient boosting.
- XGBClassifier optimizes an objective function that quantifies the model's performance. The objective function measures the difference between the predicted and actual values and guides the algorithm to minimize this difference.

![XGBCLASSIFIER](https://github.com/Jeevith23/Employee-retention-rate/assets/139576422/282b3634-0553-47b0-bd50-7aeae29a094c)


---
## Deploy the model

To deploy the model first we need to load the model as a file(.jkl) for which we are using the joblib library function.

``` python3
import joblib
# Save the trained model to a file
joblib.dump(model5, 'xgb_model.pkl')
# Load the saved Random Forest Classifier from 'xgb_model.pkl'
model5 = joblib.load('xgb_model.pkl')
```
---
## Get the User Input
The model is completely trained with a specific dataset, Hence the user should select the `Input` from the `Choice given`.

---

## The Model predicts and Displays the Output

The output would be,
- Retention - `1`: The employee is predicted to stay
- Retention - `0`: The employee is predicted to leave 

---
## Analyse the Data

Here we have some graphs and visuals of the data which we have analyzed.

![Visualization](https://github.com/Jeevith23/Employee-retention-rate/assets/139576422/c5e45b51-7690-4c56-89f5-410db86df871)

