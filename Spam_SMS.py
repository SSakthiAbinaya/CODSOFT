import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('spam.csv', encoding='latin1')
df = df[['v1', 'v2']]
df = df.rename(columns={'v1': 'label', 'v2': 'message'})

# Split the dataset into training and testing sets
train_data, test_data = df[:4500], df[4500:]

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_data['message'])
y_train = train_data['label']
X_test = vectorizer.transform(test_data['message'])
y_test = test_data['label']

# Define algorithms
algorithms = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(),
    'Support Vector Machine': SVC()
}

scores = {}
for name, clf in algorithms.items():
    # Train and fit the model
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Calculate scores
    score = accuracy_score(y_test, y_pred)
    print("{} Accuracy Score: {:.2f}".format(name, score))
    scores[name] = score

print("\nConfusion Matrix")
for name, score in scores.items():
    cm = confusion_matrix(y_test, y_pred)
    print("-" * 80)
    print("Model: ", name)
    print("Confusion matrix:\n", cm)

# Create a pie chart to display the number of emails classified as spam or not spam
spam_count = test_data[test_data['label'] == 'spam'].shape[0]
not_spam_count = test_data[test_data['label'] == 'ham'].shape[0]

labels = ['Spam', 'Not Spam']
sizes = [spam_count, not_spam_count]
colors = ['#DF3714', '#1B73B4']
# Plot the bar chart for algorithm comparison
plt.figure(figsize=(10, 6))
plt.bar(scores.keys(), scores.values(), color=['#536ED5', '#B872CD', '#DC5D8D'])
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
plt.title('Comparison of Algorithms in terms of Accuracy')
plt.ylim(0, 1)  # Set the y-axis limit to 0-1 for accuracy score
plt.show()

plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Distribution of Emails - Spam vs. Not Spam')
plt.show()