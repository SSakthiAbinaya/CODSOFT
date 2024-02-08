# Step 1: Importing Libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import nltk
import re
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Load datasets
train_path = "Genre Classification Dataset/train_data.txt"
test_path = "Genre Classification Dataset/test_data.txt"

train_data = pd.read_csv(train_path, sep=':::', names=['Title', 'Genre', 'Description'], engine='python')
test_data = pd.read_csv(test_path, sep=':::', names=['Title', 'Description'], engine='python')

# Step 3: EDA & Visualization - Pie chart for the top 6 genres
plt.figure(figsize=(10, 10))
genre_counts = train_data['Genre'].value_counts(normalize=True)
top_genres = genre_counts.nlargest(6)
top_genres.plot.pie(autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Top 6 Genres in Training Data')
plt.show()

# Step 4: Data preprocessing and text cleaning
nltk.download('stopwords')
from nltk.corpus import stopwords

def text_cleaning(text):
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text)
    words = [word for word in words if word.lower() not in stop_words]

    # Placeholder: Handle missing values (replace NaN with an empty string)
    text = text if pd.notnull(text) else ""

    # Placeholder: Replace the following line with additional text cleaning steps
    cleaned_text = ' '.join(words)

    return cleaned_text

# Apply text cleaning to the 'Description' column
train_data['Cleaned_Description'] = train_data['Description'].apply(text_cleaning)

# Step 5: Continue Data Preprocessing
# Convert text to lowercase and remove special characters
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters using regex
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Placeholder: Add any additional preprocessing steps

    return text

# Apply text preprocessing to the 'Cleaned_Description' column
train_data['Cleaned_Description'] = train_data['Cleaned_Description'].apply(preprocess_text)

# Step 6: Text Vectorization Using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['Cleaned_Description'])
y_train = train_data['Genre']

# Step 7: Split Data and Train a Model (Naive Bayes)
X_train, X_val, y_train, y_val = train_test_split(X_train_tfidf, y_train, test_size=0.2, random_state=42)

naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train, y_train)

# Step 8 (Continued): Mark Predictions on the Test Data
# Placeholder: Add any necessary text cleaning and preprocessing for the test data
test_data['Cleaned_Description'] = test_data['Description'].apply(text_cleaning)
test_data['Cleaned_Description'] = test_data['Cleaned_Description'].apply(preprocess_text)

# Vectorize the test data
X_test_tfidf = tfidf_vectorizer.transform(test_data['Cleaned_Description'])

# Make predictions on the test set
test_predictions = naive_bayes_classifier.predict(X_test_tfidf)

# Add predicted genres to the test data
test_data['Predicted Genre'] = test_predictions

# Step 9: Visualize the distribution of predicted genres in the test data - Pie chart for the top 6 predicted genres
plt.figure(figsize=(10, 10))
predicted_genre_counts = test_data['Predicted Genre'].value_counts(normalize=True)
top_predicted_genres = predicted_genre_counts.nlargest(6)
top_predicted_genres.plot.pie(autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Top 6 Predicted Genres in Test Data')
plt.show()

# Save the updated results to a new CSV file
test_data.to_csv('predicted_results_with_visualization_pie_chart_top_6.csv', index=False)

# Optionally, print the test data with predicted genres
print('\nTest Data with Predicted Genres:\n', test_data[['Title', 'Predicted Genre']])
