Predicting heart disease using machine learning(Logistic Regression)

## License

[MIT](https://choosealicense.com/licenses/mit/)

  This notebook looks into various machine learning library using Python and data science libraries in an attempt to build a machine learning model. Predicting whether a person has heart disease or not based on their medical attributes.

1. Problem Statement

Predicting whether a person has heart disease or not based on their medical attributes?.


2. Data

- The original data came from the Cleavland data from the UCI Machine Learning Repository. https://archive.ics.uci.edu/ml/datasets/heart+Disease
- it is available on Kaggle. https://www.kaggle.com/ronitf/heart-disease-uci

3. Attribute Information:-

- Age: The data consists a sample of people between age 28 and 77.
- Sex: Both gender have been included in this data set. Later for analysis, Females have been assigned value ‘0’ and males have been assigned value ‘1’.
- ChestPainType: There are four types of chest pain: Typical Angina (TA), Atypical Angina (ATA), Non-Anginal Pain (NAP), Asymptomatic(ASY)
- RestingBP: Resting blood pressure (in mm Hg).
- Cholesterol: The person’s cholesterol measurement in mg/dl.
- FastingBS: A person’s fasting blood sugar level (if ‘< 120 mg/dl’ then ‘0’ or ‘> 120 mg/dl’).
- RestingECG: Resting Electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality [ST], 2 = showing probable or definite left ventricular hypertrophy [LVH]).
- MaxHR: A person’s maximum heart rate achieved.
- ExerciseAngina: Exercise induced angina (No = 0, Yes = 1).
- Oldpeak: ST depression induced by exercise relative to rest (‘ST’ relates to positions on the ECG plot).
- ST_Slope: The slope of the peak exercise ST segment (1 = Up-sloping, 2 = Flat, 3 = Down-sloping).
- HeartDisease: Does a person has heart disease (No = 0, Yes = 1).


## Methods used
- Logistic Regression(Classification)

- RandomForestClassifier

- SVM

- KNN

- RandomizedSearchCV

- GridSearchCV

