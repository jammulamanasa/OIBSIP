import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('iris.csv')

# Preview the dataset
print(data.head())

# Define feature variables (X) and the target variable (y)
X = data.drop('species', axis=1)
y = data['species']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the models
svm_model = SVC()
lr_model = LogisticRegression(max_iter=200)
dt_model = DecisionTreeClassifier()

# Train the models
svm_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)

# Make predictions
svm_pred = svm_model.predict(X_test)
lr_pred = lr_model.predict(X_test)
dt_pred = dt_model.predict(X_test)

# Evaluate the models
print("SVM Model Accuracy:", accuracy_score(y_test, svm_pred))
print("Logistic Regression Model Accuracy:", accuracy_score(y_test, lr_pred))
print("Decision Tree Model Accuracy:", accuracy_score(y_test, dt_pred))

print("\nSVM Model Classification Report:\n", classification_report(y_test, svm_pred))
print("\nLogistic Regression Classification Report:\n", classification_report(y_test, lr_pred))
print("\nDecision Tree Classification Report:\n", classification_report(y_test, dt_pred))

print("\nSVM Confusion Matrix:\n", confusion_matrix(y_test, svm_pred))
print("\nLogistic Regression Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))
print("\nDecision Tree Confusion Matrix:\n", confusion_matrix(y_test, dt_pred))
