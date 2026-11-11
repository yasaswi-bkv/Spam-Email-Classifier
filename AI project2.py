# SPAM EMAIL CLASSIFIER USING NLP AND NAIVE BAYES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Step 1: Load Dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

data = pd.read_csv(
    url,
    sep='\t',
    header=None,
    names=['label', 'message']
)

print("First Five Records:")
print(data.head())

# Step 2: Explore Dataset
print("\nDataset Shape:", data.shape)
print("\nMissing Values:")
print(data.isnull().sum())

print("\nMessage Distribution:")
print(data['label'].value_counts())

# Step 3: Visualize Dataset
data['label'].value_counts().plot(
    kind='bar',
    figsize=(6, 4)
)

plt.title("Spam and Not Spam Distribution")
plt.xlabel("Message Type")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()

# Step 4: Convert Labels into Numbers
data['label'] = data['label'].map({
    'ham': 0,
    'spam': 1
})

# Step 5: Separate Input and Output
X = data['message']
y = data['label']

# Step 6: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# Step 7: Convert Text into Numerical Features
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english'
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 8: Train Naive Bayes Model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

print("\nModel Trained Successfully!")

# Step 9: Evaluate Model
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy * 100, "%")

# Step 10: Classification Report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=['Not Spam', 'Spam']
))

# Step 11: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Blues')

plt.title("Spam Classifier Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks([0, 1], ['Not Spam', 'Spam'])
plt.yticks([0, 1], ['Not Spam', 'Spam'])

for i in range(2):
    for j in range(2):
        plt.text(
            j, i, cm[i, j],
            ha='center',
            va='center'
        )

plt.colorbar()
plt.show()

# Step 12: Test Custom Messages
def check_spam(message):

    message_vector = vectorizer.transform([message])
    prediction = model.predict(message_vector)

    if prediction[0] == 1:
        print("Result: SPAM MESSAGE")
    else:
        print("Result: NOT SPAM MESSAGE")

message = input("\nEnter a message: ")
check_spam(message)

# Step 13: Save Model and Vectorizer
joblib.dump(model, "spam_classifier.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nModel and Vectorizer Saved Successfully!")

# Step 14: Load and Test Saved Model
loaded_model = joblib.load("spam_classifier.pkl")
loaded_vectorizer = joblib.load("tfidf_vectorizer.pkl")

test_message = ["Congratulations! You won a free gift!"]

test_vector = loaded_vectorizer.transform(test_message)
result = loaded_model.predict(test_vector)

print("\nSaved Model Prediction:")

if result[0] == 1:
    print("SPAM")
else:
    print("NOT SPAM")
