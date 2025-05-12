# Iris Flower Classification - Template
# Author: [KELOMPOK 1]
# Date: [12-5-2025]

# 1. Import Libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 2. Load Dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# 3. Exploratory Data (opsional visualisasi)
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
df['target_name'] = df['target'].apply(lambda i: target_names[i])

# Visualisasi distribusi fitur
sns.pairplot(df, hue='target_name')
plt.suptitle("Visualisasi Dataset Iris", y=1.02)
plt.show()

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model Training
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 6. Model Prediction
y_pred = model.predict(X_test)

# 7. Evaluation
print("Akurasi Model:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualisasi Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", 
            xticklabels=target_names, yticklabels=target_names, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
