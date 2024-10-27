# spam_detection.py

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import re
import joblib

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lower case
    text = text.split()  # Split into words
    text = [ps.stem(word) for word in text if word not in stop_words]  # Stem and remove stop words
    return ' '.join(text)

# Load the dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

# Data cleaning
df = df.rename(columns={'v1': 'label', 'v2': 'message'})
df = df[['label', 'message']]  # Select only necessary columns

# Apply text preprocessing
df['cleaned_text'] = df['message'].apply(preprocess_text)

# Convert labels to binary (spam: 1, ham: 0)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Vectorize the cleaned text data
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['cleaned_text']).toarray()
y = df['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)

# Train an SVM model
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

# Evaluate models
print('Naive Bayes Accuracy:', accuracy_score(y_test, nb_pred))
print('SVM Accuracy:', accuracy_score(y_test, svm_pred))

print("\nNaive Bayes Classification Report:\n", classification_report(y_test, nb_pred))
print("\nSVM Classification Report:\n", classification_report(y_test, svm_pred))

# Optional: Save the trained models
joblib.dump(nb_model, 'naive_bayes_model.pkl')
joblib.dump(svm_model, 'svm_model.pkl')
