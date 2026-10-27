
# 🤖 AI Fake Review Detector

An AI/ML student project that classifies customer reviews as genuine or synthetic and predicts their sentiment using Machine Learning.

## ✨ Features

- Review classification: genuine or synthetic
- Sentiment analysis: positive or negative
- Interactive Streamlit dashboard
- CSV upload for bulk review analysis
- Charts showing review and sentiment distributions
- Downloadable analysis results

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib

## 📁 Project Structure

```text
FakeReviewAI/
├── app.py
├── main.py
├── reviews.csv
├── test_reviews.csv
├── train_model.py
├── evaluate_model.py
├── requirements.txt
└── README.md
```

## 🚀 How to Run

1. Install Python.
2. Install the required libraries:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   python -m streamlit run app.py
   ```

4. Open the local URL displayed in the terminal.

## 🧠 Machine Learning

The project uses CountVectorizer to convert review text into numerical features and Multinomial Naive Bayes for classification.

The model predicts review type and sentiment using the labeled examples in `reviews.csv`.

## ⚠️ Limitations

The current dataset contains only 12 practice reviews. The model's predictions and evaluation scores are not reliable evidence of real-world fake-review detection. A larger, verified dataset and more extensive testing are needed.

## 👨‍💻 Project Status

Beginner AI/ML portfolio project — under development.
