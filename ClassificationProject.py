#!/usr/bin/env python
# coding: utf-8

# ### Step 1: Import Libraries & Data

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

# Load dataset
data = pd.read_csv("data/online_shoppers_intention.csv")
data


# ### Step 2: Exploratory Data Analysis (EDA)

# #### Basic Analysis:

# In[ ]:


print(data.info())
print(data.describe(include='all'))


# In[ ]:


#show any nulls in the columns (our dataset has no missing values)
print(data.isnull().sum())


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

# Numerical features
num_features = data.select_dtypes(include=['int64', 'float64']).columns

# Histogram distributions for numerical features
data[num_features].hist(bins=20, figsize=(16, 12), color='skyblue', edgecolor='black')
plt.tight_layout()
plt.suptitle('Numerical Feature Distributions', y=1.02, fontsize=16)
plt.show()


# In[ ]:


# Explicitly select categorical (object, bool) columns
cat_features = data.select_dtypes(include=['object', 'bool']).columns

# Count plots for categorical features without warnings
for col in cat_features:
    plt.figure(figsize=(8, 4))
    sns.countplot(y=col, 
                  data=data, 
                  order=data[col].value_counts().index, 
                  hue=col, 
                  palette='pastel', 
                  legend=False)
    plt.title(f'Distribution of {col}', fontsize=14)
    plt.xlabel('Count')
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()


# #### Correlation Heatmap:

# In[ ]:


# Encode Revenue as numeric
data['Revenue'] = data['Revenue'].astype(int)

numeric_features = data.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(12,10))
sns.heatmap(numeric_features.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# #### Inference & Dimensionality Reduction:

# In[ ]:


# Drop highly correlated features
# BounceRates and ExitRates highly correlated (~0.91)
data_reduced = data.drop('BounceRates', axis=1)
print("Dropped BounceRates due to high correlation with ExitRates")


# #### Checking Class Imbalance:

# In[ ]:


sns.countplot(x='Revenue', data=data_reduced)
plt.title("Distribution of Revenue")
plt.show()

print(data_reduced['Revenue'].value_counts())


# ### Step 3: Data Preprocessing

# #### Encoding Categorical Features:

# In[ ]:


data_encoded = pd.get_dummies(data_reduced, drop_first=True)


# #### Splitting Data:

# In[ ]:


X = data_encoded.drop('Revenue', axis=1)
y = data_encoded['Revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# #### Handling Class Imbalance with SMOTE:

# In[ ]:


smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("After SMOTE oversampling:")
print(y_train_res.value_counts())


# ### Step 4: Feature Scaling

# In[ ]:


scaler = StandardScaler()
X_train_res_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)


# ### Step 5: Model Training with Optimization Techniques
# 

# #### Model 1: Logistic Regression with Optimization
#  - Optimization: Adjusting solver and regularization (hyperparameters: C and solver).

# In[ ]:


model_logreg = LogisticRegression(solver='liblinear', C=1.0, max_iter=500, random_state=42)
model_logreg.fit(X_train_res_scaled, y_train_res)
print("Logistic Regression Model Trained.")


# #### Model 2: Random Forest with Optimization
# - Optimization: Hyperparameter tuning with number of estimators and max depth.

# In[ ]:


model_rf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
model_rf.fit(X_train_res, y_train_res)
print("Random Forest Model Trained.")


# #### Model 3 (Extra): K-Nearest Neighbors (KNN)
# - Optimization: tuning K parameter.

# In[ ]:


model_knn = KNeighborsClassifier(n_neighbors=7)
model_knn.fit(X_train_res_scaled, y_train_res)
print("K-nearest Neighbors Model Trained.")


# ### Step 6: Evaluation of Models

# #### Logistic Regression:

# In[ ]:


y_pred_logreg = model_logreg.predict(X_test_scaled)
print("Logistic Regression Report:")
print(classification_report(y_test, y_pred_logreg))


# #### Random Forest:

# In[ ]:


y_pred_rf = model_rf.predict(X_test)
print("Random Forest Report:")
print(classification_report(y_test, y_pred_rf))


# #### KNN:

# In[ ]:


y_pred_knn = model_knn.predict(X_test_scaled)
print("KNN Report:")
print(classification_report(y_test, y_pred_knn))


# #### Confusion Matricies

# In[ ]:


fig, ax = plt.subplots(1,3,figsize=(18,5))

sns.heatmap(confusion_matrix(y_test, y_pred_logreg), annot=True, fmt='d', ax=ax[0], cmap='Blues')
ax[0].set_title('Logistic Regression')

sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', ax=ax[1], cmap='Greens')
ax[1].set_title('Random Forest')

sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt='d', ax=ax[2], cmap='Purples')
ax[2].set_title('K-Nearest Neighbors')

plt.show()


# ### Step 7: Model Comparison & Discussion

# In[ ]:


accuracy_scores = {
    'Logistic Regression': accuracy_score(y_test, y_pred_logreg),
    'Random Forest': accuracy_score(y_test, y_pred_rf),
    'KNN': accuracy_score(y_test, y_pred_knn)
}

for model, acc in accuracy_scores.items():
    print(f"{model} accuracy: {acc:.4f}")

sns.barplot(x=list(accuracy_scores.keys()), y=list(accuracy_scores.values()))
plt.title('Accuracy Comparison of Models')
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.show()


# ### Outcome:
# - Logistic Regression: interpretable, faster training, performed well after scaling and oversampling.
# 
# - Random Forest: robust, handles nonlinear relationships and categorical variables well, optimization significantly improves performance.
# 
# - KNN: sensitive to scaling, suitable after preprocessing, performance varies significantly based on K selection.

# In[ ]:




