import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

df = pd.read_csv("data/diabetes.csv")


print(df.head())
print(df.info())
print(df.describe())


print((df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] == 0).sum())

# I replaced zeros with median values
cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for col in cols:
    df[col] = df[col].replace(0, df[col].median())

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Simple visualisations

# Correlation heatmap using Blue and Red
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='cool warm')
plt.show()

# Outcome Distribution In Bar Chart
sns.countplot(x='Outcome', data=df)
plt.show()

# Features and Outcome, Scatterplot
sns.pairplot(df, hue='Outcome')
plt.show()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Modeling

# Split data into training and testing parts
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
lr = LogisticRegression(max_iter=700)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Classification reports
print("Logistic Regression Report:")
print(classification_report(y_test, y_pred_lr))
print("\nRandom Forest Report:")
print(classification_report(y_test, y_pred_rf))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# ROC curves for logistic regression and random forest

# Logistic Regression ROC
y_prob_lr = lr.predict_proba(X_test)[:,1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

# Random Forest ROC
y_prob_rf = rf.predict_proba(X_test)[:,1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)


# visualisation
plt.figure(figsize=(8,6))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# Confusion matrixes for used models


# Logistic Regression Confusion matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(5,4))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes','Diabetes'], yticklabels=['No Diabetes','Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Logistic Regression Confusion Matrix')
plt.show()

# Random Forest Confusion matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(5,4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', xticklabels=['No Diabetes','Diabetes'], yticklabels=['No Diabetes','Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Confusion Matrix')
plt.show()



