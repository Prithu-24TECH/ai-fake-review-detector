
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Page setup
st.set_page_config(
    page_title="AI Review Analytics",
    page_icon="🤖",
    layout="wide"
)

# Custom design
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #101827, #172554);
}
h1, h2, h3, p, label {
    color: white !important;
}
div[data-testid="stMetric"] {
    background: #1e293b;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #334155;
}
.stButton > button {
    width: 100%;
    background: #38bdf8;
    color: #07111f;
    font-weight: bold;
    border-radius: 10px;
    padding: 10px;
}
.stTextArea textarea {
    background: #f1f5f9;
    color: #111827;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("🤖 AI Fake Review Detector")
st.subheader("Review Intelligence Dashboard")
st.caption(
    "Machine Learning | Review Classification | Sentiment Analysis"
)

# Load training dataset
@st.cache_data
def load_data():
    return pd.read_csv("reviews.csv")

try:
    data = load_data()
except FileNotFoundError:
    st.error("Please keep reviews.csv in the same folder as app.py.")
    st.stop()
except Exception as error:
    st.error(f"Could not load reviews.csv: {error}")
    st.stop()

required = {"review", "sentiment", "label"}

if not required.issubset(data.columns):
    st.error("reviews.csv needs review, sentiment and label columns.")
    st.stop()

data = data.dropna(subset=["review", "sentiment", "label"]).copy()

if data.empty:
    st.error("Your training dataset contains no valid reviews.")
    st.stop()

for column in ["review", "sentiment", "label"]:
    data[column] = data[column].astype(str)

# Train models
try:
    fake_model = make_pipeline(
        CountVectorizer(),
        MultinomialNB()
    )

    sentiment_model = make_pipeline(
        CountVectorizer(),
        MultinomialNB()
    )

    fake_model.fit(data["review"], data["label"])
    sentiment_model.fit(data["review"], data["sentiment"])

except ValueError as error:
    st.error(f"Could not train the models: {error}")
    st.stop()

# Dashboard overview
st.markdown("---")
st.header("📊 Dashboard Overview")
st.write("Statistics for your training dataset")

total = len(data)
genuine_count = (data["label"].str.lower() == "genuine").sum()
synthetic_count = (data["label"].str.lower() == "synthetic").sum()

c1, c2, c3 = st.columns(3)

c1.metric("Total Training Reviews", total)
c2.metric("Genuine Examples", int(genuine_count))
c3.metric("Synthetic Examples", int(synthetic_count))

left, right = st.columns(2)

with left:
    st.markdown("### Review Classification")
    label_counts = data["label"].value_counts()
    st.bar_chart(label_counts)

with right:
    st.markdown("### Sentiment Distribution")
    sentiment_counts = data["sentiment"].value_counts()
    st.bar_chart(sentiment_counts)

st.caption(
    "These charts describe the training dataset, not verified real-world fraud."
)

# Single review analysis
st.markdown("---")
st.header("📝 Analyze One Review")

review = st.text_area(
    "Enter a customer review",
    placeholder="Example: The product quality is excellent...",
    height=120
)

if st.button("🔍 Analyze Review"):
    if review.strip():
        review_type = fake_model.predict([review])[0]
        sentiment = sentiment_model.predict([review])[0]

        st.markdown("### Analysis Results")
        col1, col2 = st.columns(2)

        col1.metric("Predicted Review Type", review_type.upper())
        col2.metric("Predicted Sentiment", sentiment.upper())

        if review_type.lower() == "synthetic":
            st.warning("The model classified this review as synthetic.")
        else:
            st.success("The model classified this review as genuine.")

        st.caption("Predictions may be inaccurate with a small training dataset.")
    else:
        st.warning("Please enter a review first.")

# Bulk CSV analysis
st.markdown("---")
st.header("📂 Analyze Multiple Reviews")
st.write("Upload a CSV file containing a column named `review`.")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        uploaded_data = pd.read_csv(uploaded_file)

        if "review" not in uploaded_data.columns:
            st.error("Your CSV must contain a column named 'review'.")
        else:
            uploaded_data = uploaded_data.dropna(subset=["review"]).copy()
            uploaded_data["review"] = uploaded_data["review"].astype(str)
            uploaded_data = uploaded_data[
                uploaded_data["review"].str.strip() != ""
            ].copy()

            if uploaded_data.empty:
                st.warning("No valid reviews found in the uploaded file.")
            else:
                uploaded_data["Predicted Type"] = fake_model.predict(
                    uploaded_data["review"]
                )

                uploaded_data["Predicted Sentiment"] = (
                    sentiment_model.predict(uploaded_data["review"])
                )

                st.success(f"Analyzed {len(uploaded_data)} reviews!")

                st.dataframe(
                    uploaded_data,
                    use_container_width=True,
                    hide_index=True
                )

                # Charts for uploaded reviews
                chart_left, chart_right = st.columns(2)

                with chart_left:
                    st.markdown("### Uploaded Review Types")
                    st.bar_chart(
                        uploaded_data["Predicted Type"].value_counts()
                    )

                with chart_right:
                    st.markdown("### Uploaded Sentiments")
                    st.bar_chart(
                        uploaded_data["Predicted Sentiment"].value_counts()
                    )

                # Download results
                csv_output = uploaded_data.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "⬇️ Download Analysis Results",
                    data=csv_output,
                    file_name="review_analysis_results.csv",
                    mime="text/csv"
                )

    except Exception as error:
        st.error(f"Could not process the uploaded CSV: {error}")

# Footer
st.markdown("---")
st.caption(
    "Built with Python, Pandas, Scikit-learn and Streamlit | AI/ML Student Project"
)
