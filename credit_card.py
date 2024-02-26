import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset with explicitly specified data types
data = pd.read_csv('fraudTrain.csv')

# Drop irrelevant columns
data = data.drop(columns=['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 
                           'first', 'last', 'street', 'city', 'state', 'zip', 'job', 
                           'dob', 'trans_num'])

# Convert categorical variables to numerical representation if needed (e.g., one-hot encoding)
data = pd.get_dummies(data, columns=['category', 'gender'])

# Separate features and target variable
X = data.drop('is_fraud', axis=1)
y = data['is_fraud']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)

# Predict on the testing set
logreg_y_pred = logreg_model.predict(X_test)

# Calculate accuracy for Logistic Regression
logreg_accuracy = accuracy_score(y_test, logreg_y_pred)

# Print evaluation metrics for Logistic Regression
print("Logistic Regression Model Evaluation:")
print("Accuracy:", logreg_accuracy)

# Initialize and train the Decision Tree Classifier model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Predict on the testing set
dt_y_pred = dt_model.predict(X_test)

# Calculate accuracy for Decision Tree Classifier
dt_accuracy = accuracy_score(y_test, dt_y_pred)

# Print evaluation metrics for Decision Tree Classifier
print("\nDecision Tree Classifier Model Evaluation:")
print("Accuracy:", dt_accuracy)

# Calculate mean amount for fraudulent and normal transactions
mean_amount_fraud = data[data['is_fraud'] == 1]['amt'].mean()
mean_amount_normal = data[data['is_fraud'] == 0]['amt'].mean()

print("\nMean amount for fraudulent transactions:", mean_amount_fraud)
print("Mean amount for normal transactions:", mean_amount_normal)

# Plot heatmap of correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()

# Plot confusion matrix for Logistic Regression
logreg_cm = confusion_matrix(y_test, logreg_y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(logreg_cm, annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.show()

# Plot confusion matrix for Decision Tree Classifier
dt_cm = confusion_matrix(y_test, dt_y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(dt_cm, annot=True, fmt="d", cmap="Blues")
plt.title("Decision Tree Classifier Confusion Matrix")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.show()
