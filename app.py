import streamlit as st
import pickle
import re
import ssl
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK data (you might need to handle SSL issues here)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("stopwords")
nltk.download("wordnet")


# Load the model and vectorizer
@st.cache_resource
def load_model():
    with open("models/logistic_regression_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model


@st.cache_resource
def load_vectorizer():
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return vectorizer


# Preprocess text function (same as in your notebook)
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = [
        lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words
    ]
    return " ".join(words)


# Main app
def main():
    st.title("📰 NewsCheck - Fake News Detector")
    st.markdown(
        """
    This app uses machine learning to detect whether a news article is likely to be real or fake.
    Enter the text of a news article below to check its authenticity.
    """
    )

    # Load model and vectorizer
    model = load_model()
    vectorizer = load_vectorizer()

    # Text input
    text_input = st.text_area("Enter news article text:", height=200)

    if st.button("Check News"):
        if text_input:
            with st.spinner("Analyzing..."):
                # Preprocess the text
                processed_text = preprocess_text(text_input)

                # Vectorize
                text_vector = vectorizer.transform([processed_text])

                # Predict
                prediction = model.predict(text_vector)[0]
                probability = model.predict_proba(text_vector)[0][prediction]

                # Display results
                st.markdown("---")
                st.subheader("Results")

                if prediction == 1:
                    st.success(
                        f"✅ This appears to be REAL NEWS (Confidence: {probability:.2%})"
                    )
                else:
                    st.error(
                        f"❌ This appears to be FAKE NEWS (Confidence: {probability:.2%})"
                    )

                # Show confidence meter
                st.progress(probability)
                st.caption(f"Confidence: {probability:.2%}")
        else:
            st.warning("Please enter some text to analyze.")


if __name__ == "__main__":
    main()
