import pandas as pd

# Load data
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine
data = pd.concat([fake, true])

# Shuffle
data = data.sample(frac=1).reset_index(drop=True)

# Keep only needed columns
data = data[["text", "label"]]

print(data.head())

import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean_text(text):
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    
    # 3. Split into words
    words = text.split()
    
    # 4. Remove stopwords
    words = [word for word in words if word not in stop_words]
    
    # 5. Join back into sentence
    return " ".join(words)

# Apply cleaning
data["text"] = data["text"].apply(clean_text)

print(data.head())


from sklearn.feature_extraction.text import TfidfVectorizer

# Create vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# Convert text to numbers
X = vectorizer.fit_transform(data["text"])

# Labels
y = data["label"]

print(X.shape)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report

# Make predictions
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Detailed report
print(classification_report(y_test, y_pred))

print("Sample Prediction:")
print("Actual:", y_test.iloc[0])
print("Predicted:", y_pred[0])

import pickle

# Save model
pickle.dump(model, open("model.pkl", "wb"))

# Save vectorizer
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model and vectorizer saved successfully")

