#!/usr/bin/env python
# coding: utf-8

# # Predicting heart disease using machine learning(Logistic Regression)

# - This notebook looks into various machine learning library using Python and data science libraries in an attempt to build a machine learning model. Predicting whether a person has heart disease or not based on their medical attributes.

# # 1. Problem Statement
# - Predicting whether a person has heart disease or not based on their medical attributes?.

# # 2. Data
# - The original data came from the Cleavland data from the UCI Machine Learning Repository. https://archive.ics.uci.edu/ml/datasets/heart+Disease
# - it is available on Kaggle. https://www.kaggle.com/ronitf/heart-disease-uci

# # 3. Attribute Information:-
# - Age: The data consists a sample of people between age 28 and 77.
# - Sex: Both gender have been included in this data set. Later for analysis, Females have been assigned value ‘0’ and males have been assigned value ‘1’.
# - ChestPainType: There are four types of chest pain: Typical Angina (TA), Atypical Angina (ATA), Non-Anginal Pain (NAP), Asymptomatic(ASY)
# - RestingBP: Resting blood pressure (in mm Hg).
# - Cholestoral: The person’s cholesterol measurement in mg/dl.
# - FastingBS: A person’s fasting blood sugar level (if ‘< 120 mg/dl’ then ‘0’ or ‘> 120 mg/dl’).
# - RestingECG: Resting Electrocardiographic measurement (0  = normal, 1 = having ST-T wave abnormality [ST], 2 = showing probable or definite left ventricular hypertrophy [LVH]).
# - MaxHR: A person’s maximum heart rate achieved.
# - ExerciseAngina: Exercise induced angina (No = 0, Yes = 1).
# - Oldpeak: ST depression induced by exercise relative to rest (‘ST’ relates to positions on the ECG plot).
# - ST_Slope: The slope of the peak exercise ST segment (1 = Up-sloping, 2 = Flat, 3 = Down-sloping).
# - HeartDisease: Does a person has heart disease (No = 0, Yes = 1).

# # Set Working Directory

# In[1]:


import os 
os.chdir("C:\\Users\\Ashishkumar Rana\\Desktop\\MSC_Cariculum\\SEM1\\Case Study") # Change working directory
os.getcwd() # get the current working directory


# # Import Required Library

# In[2]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import train_test_split # To partion the data
from sklearn.linear_model import LogisticRegression # Library for Logistic Regression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix # Importing performance matrix, accuracy score and confusion matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_roc_curve


# # Ignore warning message

# In[3]:


import warnings
warnings.filterwarnings('ignore')


# # Import data set in to Python

# In[4]:


data = pd.read_csv("heart.csv")


# In[5]:


data


# # Creating a Copy of Orginal data 

# In[6]:


df = data.copy() # Creating a copy of original data set so that our original data will not have any impact when we made any changes in updates file


# In[7]:


df


# # Exploratory data analysis.
#  -  Know about data.
#  -  Data Preprocessing(Deal with missing values)
#  -  Cross table and data visualization

# # Data Pre-Processing

# In[8]:


df.head() # To see first Five row of the data


# In[9]:


df.tail()


# In[10]:


df.shape


# In[11]:


df.size


# # Let's Know more about data

# In[12]:


df.info()


# - There is no missing values present in our data sets.
# - Age, RestingBp, Cholesterol, FastingBS, MaxHR and HeartDisease are of Integer data type.
# - Sex, ChestPainType, RestingECG , ExerciseAngina,  ST_Slope and Oldpeak are of Object and float data  type respectively.

# # To check missing value in data if any

# In[13]:


df.isnull()


# In[14]:


df.isnull().sum()


# - Since we dont have any missing value in data set 

# # To check duplicate value if  any

# In[15]:


df.duplicated().sum()


# - So no duplicate value is there in data

# # Descriptive statistics of the data

# In[16]:


df.describe() # Descriptive statistics gives us five number theory i.e. Min, Max,Count, mean, Standard deviation and Quartiles.


# In[17]:


df.describe(include= "O")


# In[18]:


df.columns


# In[19]:


df["ExerciseAngina"].value_counts()


# In[20]:


df["ST_Slope"].value_counts()


# # Correlation.
# ## Relationship between independent variable
# 

# In[21]:


df.corr()


# In[22]:


sns.heatmap(df.corr(), annot= True)


# # Visualization

# # Frequency Distribution of HeartDisease

# In[23]:


df["HeartDisease"].value_counts().plot(kind="bar", color=["Green", "Red"]);


# ## Compare HeartDisease column with sex column 

# In[24]:


df["Sex"].value_counts()


# # Gender Vs HeartDisease

# In[25]:


Gender_Vs_HeartDisease = pd.crosstab(index = df["Sex"],
                               columns = df["HeartDisease"],
                               margins= True,
                               normalize = "index") #Include Row and Column total


# In[26]:


Gender_Vs_HeartDisease


# - 45% people are not affacted by HeartDisease
# - 55% people are affacted by HeartDisease

# #### Create a plot of crosstab
# 

# In[27]:


pd.crosstab(df.HeartDisease, df.Sex).plot(kind="bar",figsize=(10, 6),
                                    color=["Green", "Red"])

plt.title("Heart Disease Frequency for Sex")
plt.xlabel("0 = No Diesease, 1 = Disease")
plt.ylabel("Count")
plt.legend(["Female", "Male"]);
plt.xticks(rotation=0);


# - The Proportion of Male for both the cases having HeartDisease and not having HeartDisease is more as comapre to Female.

# In[28]:


plt.figure(figsize=(10, 6))

# Scatter plot of a person having disease

plt.scatter(df.Age[df.HeartDisease==1],
            df.MaxHR[df.HeartDisease==1],
            c="Salmon")

# Scatter plot of a person not having disease

plt.scatter(df.Age[df.HeartDisease==0],
            df.MaxHR[df.HeartDisease==0],
            c="Blue")

# Add some helpful info
plt.title("Heart Disease in function of Age and Max Heart Rate")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate")
plt.legend(["Disease", "No Disease"]);


# - People between age group 52 and  65 having more Heartrate and most of them having Heartdisease.

# # Histogram.
# ## Distribution of Age 

# In[29]:


sns.displot(df["Age"], bins= 10, kde= True)


# - People with Age 55-65 are high in frequancy

# In[30]:


sns.boxplot("HeartDisease", "Age", data = df)


# - People within age between 43 to 58 are not having any HeartDisease.
# - People within age between 52 to 63 are having HeartDisease.

# # Reindexing the Sex Column as 0 and 1 for Female and Male Respectively.

# In[31]:


df['Sex'][df['Sex'] == 'F' ] = 0
df['Sex'][df['Sex'] == 'M' ] = 1


# # Heart Disease Frequency per Chest Pain Type
# - cp - chest pain type
# - 0: Typical angina: chest pain related decrease blood supply to the heart
# - 1: Atypical angina: chest pain not related to heart
# - 2: Non-anginal pain: typically esophageal spasms (non heart related)
# - 3: Asymptomatic: chest pain not showing signs of disease

# In[32]:


df["ChestPainType"].value_counts()


# In[33]:



df['ChestPainType'][df['ChestPainType'] == 'TA'] =  0
df['ChestPainType'][df['ChestPainType'] == 'ATA'] = 1
df['ChestPainType'][df['ChestPainType'] == 'NAP'] = 2
df['ChestPainType'][df['ChestPainType'] == 'ASY'] = 3


# # Change FastingBS Integer to object data type

# - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# - '>126' mg/dL signals diabetes

# In[34]:


df["FastingBS"].value_counts()


# In[35]:


'''

df['FastingBS'][df['FastingBS'] == 0] = 'lower than 120mg/ml'
df['FastingBS'][df['FastingBS'] == 1] = 'greater than 120mg/ml'

'''


# In[36]:


Gender_Vs_FastingBS = pd.crosstab(index = df["Sex"],
                               columns = df["FastingBS"],
                               margins= True,
                               normalize = "index") #Include Row and Column total


# In[37]:


Gender_Vs_FastingBS


# In[38]:


pd.crosstab(df.FastingBS, df.Sex).plot(kind="bar",figsize=(10, 6),
                                    color=["Green", "Red"])

plt.title("FastingBS Frequency for Sex")
plt.xlabel("0 = No diabetes, 1 = diabetes")
plt.ylabel("No. oF People")
plt.legend(["Female", "Male"]);
plt.xticks(rotation=0);


# - 76% People having no diabetes where as 24% peope having diabetes.
# - 76% Females not having diabetes where 14% Females having diabetes
# - 75% Male not having diabetes where 25% Male having diabetes

# # RestingECG

# In[39]:


df['RestingECG'][df['RestingECG'] == "Normal"] = 0
df['RestingECG'][df['RestingECG'] == "ST"] = 1
df['RestingECG'][df['RestingECG'] == "LVH"] = 2


# In[40]:


df["RestingECG"].value_counts()


# In[41]:


HeartDisease_Vs_RestingECG = pd.crosstab(index = df["HeartDisease"],
                               columns = df["RestingECG"],
                               margins= True,
                               normalize = "index") #Include Row and Column total


# In[42]:


HeartDisease_Vs_RestingECG


# - Percentage of people having Normal Ecg is 60%
# - Percentage of people having Standard Ecg is 19% 
# - Percentage of people having LVH Ecg is 21%

# # ExerciseAngina

# In[43]:


df['ExerciseAngina'][df['ExerciseAngina'] == "N"] = 0
df['ExerciseAngina'][df['ExerciseAngina'] == "Y"] = 1


# # ST_Slope

# In[44]:


df['ST_Slope'][df['ST_Slope'] == "Up"] = 1
df['ST_Slope'][df['ST_Slope'] == "Flat"] = 2
df['ST_Slope'][df['ST_Slope'] == "Down"] = 3


# In[45]:


df.columns


# In[46]:


df1 = df


# In[47]:


df1


# In[48]:


df.info()


# In[49]:


df1.head()


# In[50]:


df1.tail()


# In[51]:


df1.corr()


# In[52]:


Correlation = sns.heatmap(df1.corr(), annot= True)


# # Visualization

# In[53]:


# Make the crosstab more visual
pd.crosstab(df1.ChestPainType, df1.HeartDisease).plot(kind="bar",
                                   figsize=(10, 6),
                                   color=["salmon", "lightblue"])

# Add some communication
plt.title("Heart Disease Frequency Per Chest Pain Type")
plt.xlabel("Chest Pain Type")
plt.ylabel("Number of People")
plt.legend(["No Disease", "Disease"])
plt.xticks(rotation=0);


# - People with 1st type of chest pain are least vulnerable to heartdisese.
# - People with 3rd type of chest pain are more vulnerable to heartdisese.

# In[54]:


sns.distplot(df1.RestingBP[df1.HeartDisease==0])
sns.distplot(df1.RestingBP[df1.HeartDisease==1])
plt.legend(['0','1'])
plt.show()


# In[55]:


sns.countplot(x = "FastingBS", hue = "HeartDisease", data = df1)


# In[56]:


sns.countplot(x = "RestingECG", hue = "HeartDisease", data = df1)


# In[57]:


sns.countplot(x = "ST_Slope", hue = "HeartDisease", data = df1)


# # Model Building

# # Logistic Regression.
# - It is a machine learning classification algoritm that is use to predict the probability of categorical dependent variable.
# - Using Logistic regression we will build a classifier model based on avaialable data.

# In[58]:


df1


# In[59]:


df1.to_csv("HeartDisease_Cleaned_data.csv")


# # Storing the column name

# In[60]:


col_list = list(df1.columns)


# In[61]:


col_list


# # Seprating the input variable from the data

# In[62]:


input_variable = list(set(col_list)- set(["HeartDisease"]))


# In[63]:


input_variable


# In[64]:


len(input_variable)


# # Storing the value of input variable

# In[65]:


x = df[input_variable].values


# In[66]:


x


# # Storing the output variable as y

# In[67]:


y = df1["HeartDisease"].values


# In[68]:


y


# In[69]:


x.shape


# In[70]:


y.shape


# In[71]:


x.size


# In[72]:


y.size


# # Spliting the data into train and test¶

# In[73]:


train_x, test_x, train_y, test_y = train_test_split(x,y,test_size = 0.3, random_state=0)


# In[74]:


train_x


# In[75]:


test_x 


# In[76]:


train_y


# In[77]:


test_y


# In[78]:


from sklearn.linear_model import LogisticRegression # Library for Logistic Regression

logistic = LogisticRegression()


# In[79]:


logistic.fit(train_x,train_y)


# In[80]:


logistic.coef_


# In[81]:


logistic.intercept_


# In[82]:


prediction = logistic.predict(test_x)


# In[83]:


prediction


# In[84]:


accuracy_score = accuracy_score(test_y, prediction)


# In[85]:


accuracy_score


# # Making prediction

# In[86]:


print(test_y) 
y_predic = logistic.predict(test_x)
y_predic


# # Model Evaluation

# In[87]:


logistic.predict_proba(x)


# ## Confusion matrix

# In[88]:


confusion_matrix(test_y,y_predic)


# In[89]:


## 227 / 276 = 0.822463768115942


# In[129]:


# Put models in a dictionary
models = {"SVM": svm.SVC(),
          "KNN": KNeighborsClassifier(),
          "Random Forest": RandomForestClassifier()}

# Create a function to fit and score models
def fit_and_score(models, train_x, test_x, train_y, test_y):
    """
    Fits and evaluates given machine learning models.
    models : a dict of differetn Scikit-Learn machine learning models
    X_train : training data (no labels)
    X_test : testing data (no labels)
    y_train : training labels
    y_test : test labels
    """
    # Set random seed
    np.random.seed(42)
    # Make a dictionary to keep model scores
    model_scores = {}
    # Loop through models
    for name, model in models.items():
        # Fit the model to the data
        model.fit(train_x, train_y)
        # Evaluate the model and append its score to model_scores
        model_scores[name] = model.score(test_x, test_y)
    return model_scores


# In[130]:


model_scores = fit_and_score(models=models,
                             train_x = train_x,
                             test_x = test_x,
                             train_y = train_y,
                             test_y = test_y)

model_scores


# # Model Comparison

# In[131]:


model_compare = pd.DataFrame(model_scores, index=["accuracy"])
model_compare.T.plot.bar();


# # Hyperparameter tuning

# In[93]:


# Let's tune KNN

train_scores = []
test_scores = []

# Create a list of differnt values for n_neighbors
neighbors = range(1, 21)

# Setup KNN instance
knn = KNeighborsClassifier()

# Loop through different n_neighbors
for i in neighbors:
    knn.set_params(n_neighbors=i)
    
    # Fit the algorithm
    knn.fit(train_x, train_y)
    
    # Update the training scores list
    train_scores.append(knn.score(train_x, train_y))
    
    # Update the test scores list
    test_scores.append(knn.score(test_x, test_y))


# In[94]:


train_scores


# In[95]:


test_scores


# In[96]:


plt.plot(neighbors, train_scores, label="Train score")
plt.plot(neighbors, test_scores, label="Test score")
plt.xticks(np.arange(1, 21, 1))
plt.xlabel("Number of neighbors")
plt.ylabel("Model score")
plt.legend()

print(f"Maximum KNN score on the test data: {max(test_scores)*100:.2f}%")


# # Hyperparameter tuning with RandomizedSearchCV
# - We're going to tune:
# 
# - LogisticRegression()
# - RandomForestClassifier()
# 
#   - using RandomizedSearchCV

# In[97]:


# Create a hyperparameter grid for LogisticRegression
log_reg_grid = {"C": np.logspace(-4, 4, 20),
                "solver": ["liblinear"]}

# Create a hyperparameter grid for RandomForestClassifier
rf_grid = {"n_estimators": np.arange(10, 1000, 50),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2)}


# Now we've got hyperparameter grids setup for each of our models, let's tune them using RandomizedSearchCV...

# In[98]:


# Tune LogisticRegression

np.random.seed(42)

# Setup random hyperparameter search for LogisticRegression
rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                param_distributions=log_reg_grid,
                                cv=5,
                                n_iter=20,
                                verbose=True)

# Fit random hyperparameter search model for LogisticRegression
rs_log_reg.fit(train_x, train_y)


# In[99]:


rs_log_reg.best_params_


# In[100]:


rs_log_reg.score(test_x, test_y)


# Now we've tuned LogisticRegression(), let's do the same for RandomForestClassifier()

# In[101]:


# Setup random seed
np.random.seed(42)

# Setup random hyperparameter search for RandomForestClassifier
rs_rf = RandomizedSearchCV(RandomForestClassifier(), 
                           param_distributions=rf_grid,
                           cv=5,
                           n_iter=20,
                           verbose=True)

# Fit random hyperparameter search model for RandomForestClassifier()
rs_rf.fit(train_x, train_y)


# In[102]:


# Find the best hyperparameters
rs_rf.best_params_


# In[103]:


# Evaluate the randomized search RandomForestClassifier model
rs_rf.score(test_x, test_y)


# # Hyperparamter Tuning with GridSearchCV
# - Since our LogisticRegression model provides the best scores so far, we'll try and improve them again using GridSearchCV

# In[104]:


# Different hyperparameters for our LogisticRegression model
log_reg_grid = {"C": np.logspace(-4, 4, 30),
                "solver": ["liblinear"]}

# Setup grid hyperparameter search for LogisticRegression
gs_log_reg = GridSearchCV(LogisticRegression(),
                          param_grid=log_reg_grid,
                          cv=5,
                          verbose=True)

# Fit grid hyperparameter search model
gs_log_reg.fit(train_x, train_y);


# In[105]:


# Check the best hyperparmaters
gs_log_reg.best_params_


# In[106]:


# Evaluate the grid search LogisticRegression model
gs_log_reg.score(test_x, test_y)


# # Evaluting our tuned machine learning classifier, beyond accuracy
#    - ROC curve and AUC score
#    - Confusion matrix
#    - Classification report
#    - Precision
#    - Recall
#    - F1-score
#      - and it would be great if cross-validation was used where possible.

# # To make comparisons and evaluate our trained model, first we need to make predictions.

# In[107]:


# Make predictions with tuned model
y_preds = gs_log_reg.predict(test_x)


# In[108]:


y_preds


# In[109]:


test_y


# In[110]:


# Plot ROC curve and calculate and calculate AUC metric
plot_roc_curve(gs_log_reg, test_x, test_y)


# In[111]:


# Confusion matrix
print(confusion_matrix(test_y, y_predic))


# In[112]:


sns.set(font_scale=1.5)

def plot_conf_mat(test_y, y_preds):
    """
    Plots a nice looking confusion matrix using Seaborn's heatmap()
    """
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(confusion_matrix(test_y, y_predic),
                     annot=True,
                     cbar=False)
    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    
    bottom, top = ax.get_ylim()
#     ax.set_ylim(bottom + 0.5, top - 0.5)
    
plot_conf_mat(test_y, y_preds)


# Now we've got a ROC curve, an AUC metric and a confusion matrix, let's get a classification report as well as cross-validated precision, recall and f1-score.

# In[113]:


print(classification_report(test_y, y_preds))


# # Calculate evaluation metrics using cross-validation

# We're going to calculate accuracy, precision, recall and f1-score of our model using cross-validation and to do so we'll be using cross_val_score()

# In[114]:


# Check best hyperparameters
gs_log_reg.best_params_


# In[115]:


# Create a new classifier with best parameters
clf = LogisticRegression(C= 9.236708571873866,
                         solver="liblinear")


# In[116]:


# Cross-validated accuracy
cv_acc = cross_val_score(clf,
                         x,
                         y,
                         cv=5,
                         scoring="accuracy")
cv_acc


# In[117]:


cv_acc = np.mean(cv_acc)
cv_acc


# In[118]:


# Cross-validated precision
cv_precision = cross_val_score(clf,
                         x,
                         y,
                         cv=5,
                         scoring="precision")
cv_precision=np.mean(cv_precision)
cv_precision


# In[119]:


# Cross-validated recall
cv_recall = cross_val_score(clf,
                         x,
                         y,
                         cv=5,
                         scoring="recall")
cv_recall = np.mean(cv_recall)
cv_recall


# In[120]:


# Cross-validated f1-score
cv_f1 = cross_val_score(clf,
                         x,
                         y,
                         cv=5,
                         scoring="f1")
cv_f1 = np.mean(cv_f1)
cv_f1


# In[121]:


# Visualize cross-validated metrics
cv_metrics = pd.DataFrame({"Accuracy": cv_acc,
                           "Precision": cv_precision,
                           "Recall": cv_recall,
                           "F1": cv_f1},
                          index=[0])

cv_metrics.T.plot.bar(title="Cross-validated classification metrics",
                      legend=False);


# # Feature Importance

# - Feature importance is another as asking, "which features contributed most to the outcomes of the model and how did they contribute?"
# 
# - Finding feature importance is different for each machine learning model. One way to find feature importance is to search for "(MODEL NAME) feature importance".
# 
# - Let's find the feature importance for our LogisticRegression model...

# In[122]:


# Fit an instance of LogisticRegression
clf = LogisticRegression(C= 9.236708571873866,
                         solver="liblinear")

clf.fit(train_x, train_y);


# In[123]:


# Check coef_
clf.coef_


# In[124]:


df.head()


# In[125]:


# Match coef's of features to columns
feature_dict = dict(zip(df.columns, list(clf.coef_[0])))
feature_dict


# In[126]:


# Visualize feature importance
feature_df = pd.DataFrame(feature_dict, index=[0])
feature_df.T.plot.bar(title="Feature Importance", legend=False);


# In[127]:


pd.crosstab(df["Sex"], df["HeartDisease"])


# In[128]:


pd.crosstab(df["ST_Slope"], df["HeartDisease"])


# In[ ]:




