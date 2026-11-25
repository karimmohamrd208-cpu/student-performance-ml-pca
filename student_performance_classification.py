# Import necessary libraries for data handling, visualization, and machine learning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2



# Load the dataset and display basic information
file_path = "/xAPI-Edu-Data.csv"
df = pd.read_csv(file_path)

print(df.head())
print(df.columns)

# Handle missing values and encode categorical variables into numerical values
df = df.dropna()

le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Separate features (X) and target variable (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Split the data into training and testing sets and apply feature scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the machine learning models to be used for classification
models = {
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

# Train and evaluate models before applying dimensionality reduction
print("Before Dimensionality Reduction")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(name, accuracy_score(y_test, y_pred))

# Apply Principal Component Analysis (PCA) for dimensionality reduction and evaluate models
pca = PCA(n_components=5)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("After PCA")

for name, model in models.items():
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)
    print(name, accuracy_score(y_test, y_pred))

# Apply Feature Selection using Chi-Square test and evaluate models
selector = SelectKBest(score_func=chi2, k=5)

X_new = selector.fit_transform(abs(X), y)

X_train_fs, X_test_fs, y_train_fs, y_test_fs = train_test_split(X_new, y, test_size=0.2, random_state=42)

print("After Feature Selection")

for name, model in models.items():
    model.fit(X_train_fs, y_train_fs)
    y_pred = model.predict(X_test_fs)
    print(name, accuracy_score(y_test_fs, y_pred))

# Perform hyperparameter tuning using GridSearchCV and visualize the results
param_grid = {'n_estimators': [50, 100, 150]}

grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
grid.fit(X_train, y_train)

results = pd.DataFrame(grid.cv_results_)

plt.plot(results['param_n_estimators'], results['mean_test_score'])
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.title('Hyperparameter Tuning')
plt.show()

# Evaluate the model using confusion matrix and classification report
model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()

print(classification_report(y_test, y_pred))

# Visualize and compare model performance before and after dimensionality reduction

import matplotlib.pyplot as plt

labels = ['Before', 'PCA', 'Feature Selection']

rf_scores = [0.854, 0.708, 0.781]
lr_scores = [0.823, 0.771, 0.792]

x = range(len(labels))

plt.plot(x, rf_scores, marker='o', label='Random Forest')
plt.plot(x, lr_scores, marker='o', label='Logistic Regression')

plt.xticks(x, labels)
plt.xlabel('Method')
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison')

plt.legend()
plt.show()