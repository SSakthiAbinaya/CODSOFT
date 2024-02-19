import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
data = pd.read_csv("Churn_Modelling.csv")

# Exploratory Data Analysis
# Visualize the distribution of the target variable
plt.figure(figsize=(6, 4))
sns.countplot(x='Exited', data=data)
plt.title('Distribution of Exited')
plt.xlabel('Exited')
plt.ylabel('Count')
plt.show()

# Print the number of customers exited and not exited
exited_count = data['Exited'].value_counts()
print("Number of customers exited:", exited_count[1])
print("Number of customers not exited:", exited_count[0])

# Visualize the distribution of gender with exited status
plt.figure(figsize=(6, 4))
sns.countplot(x='Gender', hue='Exited', data=data)
plt.title('Distribution of Gender with Exited Status')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Visualize the distribution of geography using pie chart
plt.figure(figsize=(6, 6))
data['Geography'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Distribution of Geography')
plt.ylabel('')
plt.show()

# Data preprocessing
# Drop unnecessary columns like RowNumber, CustomerId, and Surname
data = data.drop(columns=["RowNumber", "CustomerId", "Surname"])

# Encode categorical variables
label_encoder = LabelEncoder()
data['Geography'] = label_encoder.fit_transform(data['Geography'])
data['Gender'] = label_encoder.fit_transform(data['Gender'])

# Split the dataset into features and target variable
X = data.drop(columns=["Exited"])
y = data["Exited"]

# Data Preprocessing
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize StandardScaler
scaler = StandardScaler()

# Scale the features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models training and evaluation
# Initialize classifiers
rf_model = RandomForestClassifier()
gb_model = GradientBoostingClassifier()
lr_model = LogisticRegression(max_iter=1000)  # Increase max_iter

# Train and evaluate Random Forest
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

# Train and evaluate Gradient Boosting
gb_model.fit(X_train_scaled, y_train)
y_pred_gb = gb_model.predict(X_test_scaled)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
precision_gb = precision_score(y_test, y_pred_gb)
recall_gb = recall_score(y_test, y_pred_gb)
f1_gb = f1_score(y_test, y_pred_gb)

# Train and evaluate Logistic Regression
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

# Printing evaluation metrics
print("Random Forest Classifier:")
print("Accuracy:", accuracy_rf)
print("Precision:", precision_rf)
print("Recall:", recall_rf)
print("F1 Score:", f1_rf)
print("\n")

print("Gradient Boosting Classifier:")
print("Accuracy:", accuracy_gb)
print("Precision:", precision_gb)
print("Recall:", recall_gb)
print("F1 Score:", f1_gb)
print("\n")

print("Logistic Regression Classifier:")
print("Accuracy:", accuracy_lr)
print("Precision:", precision_lr)
print("Recall:", recall_lr)
print("F1 Score:", f1_lr)

# Plotting accuracies
models = ['Random Forest', 'Gradient Boosting', 'Logistic Regression']
accuracies = [accuracy_rf, accuracy_gb, accuracy_lr]

plt.figure(figsize=(8, 6))
sns.barplot(x=models, y=accuracies)
plt.title('Accuracy of Different Models')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()
