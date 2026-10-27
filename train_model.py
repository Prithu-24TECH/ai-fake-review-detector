
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

data = pd.read_csv("reviews.csv")

model = make_pipeline(
    CountVectorizer(),
    MultinomialNB()
)

model.fit(data["review"], data["label"])

review = ["Amazing amazing amazing, buy now, best ever!"]

prediction = model.predict(review)

print("Customer Review:", review[0])
print("Review Type:", prediction[0])

