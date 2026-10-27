
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("reviews.csv")

# Split reviews into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    data["review"],
    data["label"],
    test_size=0.25,
    random_state=42,
    stratify=data["label"]
)

# Create and train model
model = make_pipeline(
    CountVectorizer(),
    MultinomialNB()
)

model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))
