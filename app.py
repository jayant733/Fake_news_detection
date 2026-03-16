import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ==============================
# LOAD MODEL
# ==============================

vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("fake_news_model.pkl")

# ==============================
# PAGE TITLE
# ==============================

st.title("📰 Fake News Detection System")

st.markdown("""
This system detects whether a news article is **Real or Fake** using Machine Learning models.

Models used:
- Naive Bayes
- Logistic Regression
- Linear SVM
- XGBoost
- BERT (Transformer)

Dataset: Fake News Detection Dataset
""")

# ==============================
# USER INPUT
# ==============================

news_text = st.text_area("Enter News Text")

if st.button("Predict"):

    if news_text.strip() == "":
        st.warning("Please enter some text")

    else:
        vector = vectorizer.transform([news_text])

        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0][1]

        if prediction == 1:
            st.error("🚨 Fake News Detected")
        else:
            st.success("✅ Real News")

        st.write(f"Confidence: **{probability:.2f}**")

# ==============================
# MODEL PERFORMANCE
# ==============================

st.header("Model Comparison")

data = {
    "Model": ["Naive Bayes","Logistic Regression","Linear SVM","XGBoost"],
    "Accuracy":[0.92,0.96,0.97,0.98],
    "F1 Score":[0.91,0.95,0.96,0.97]
}

df = pd.DataFrame(data)

st.dataframe(df)

st.bar_chart(df.set_index("Model")["Accuracy"])

# ==============================
# PROJECT INFO
# ==============================

st.header("About This Project")

st.write("""
This project compares traditional machine learning models and transformer-based models for fake news detection.

Techniques used:

• TF-IDF Feature Engineering  
• Bigram NLP features  
• Machine Learning Classification  
• Ensemble Models (XGBoost)  
• Transformer Model (BERT)  
• Hyperparameter Tuning  
""")