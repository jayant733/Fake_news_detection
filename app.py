import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

from src.data_utils import resolve_data_path, resolve_repo_path

# ==============================
# LOAD MODEL + VECTORIZER
# ==============================

vectorizer = joblib.load(resolve_repo_path("vectorizer.pkl"))
model = joblib.load(resolve_repo_path("fake_news_model.pkl"))
TRAINING_DATA = resolve_data_path()

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(page_title="Fake News Detector", layout="wide")

st.title("📰 Fake News Detection System")

st.markdown("Detect whether a news article is **Real or Fake** using ML.")

# ==============================
# SIDEBAR NAVIGATION
# ==============================

menu = st.sidebar.selectbox(
    "Navigation",
    ["Prediction", "EDA Insights", "Model Analysis"]
)

# ==============================
# 1️⃣ PREDICTION SECTION
# ==============================

if menu == "Prediction":

    st.header("🔍 News Prediction")

    news_text = st.text_area("Enter News Text")

    if st.button("Predict"):

        if news_text.strip() == "":
            st.warning("Please enter some text")

        else:
            vector = vectorizer.transform([news_text])
            fake_prob = float(model.predict_proba(vector)[0][1])
            real_prob = 1.0 - fake_prob

            THRESHOLD = 0.75

            if fake_prob >= THRESHOLD:
                st.error("Fake News Detected")
            else:
                st.success("✅ Real News")

            col1, col2 = st.columns(2)

            col1.metric("Fake Probability", f"{fake_prob:.2f}")
            col2.metric("Real Probability", f"{real_prob:.2f}")

            st.progress(int(real_prob * 100))
            st.caption(f"Confidence: {max(real_prob, fake_prob)*100:.1f}%")

# ==============================
# 2️⃣ EDA INSIGHTS
# ==============================

elif menu == "EDA Insights":

    st.header("📊 Data Insights")

    df = pd.read_csv(TRAINING_DATA)

    TEXT_COL = "text" if "text" in df.columns else "content"
    df = df.dropna(subset=[TEXT_COL, "label"])

    df["full_text"] = df[TEXT_COL]

    fake_text = df[df["label"] == 1]["full_text"]
    real_text = df[df["label"] == 0]["full_text"]

    if st.button("Show Most Common Words (Fake News)"):

        fake_words = Counter(" ".join(fake_text).split())
        common_fake = fake_words.most_common(20)

        words, counts = zip(*common_fake)

        plt.figure()
        plt.bar(words, counts)
        plt.xticks(rotation=45)
        plt.title("Top Words in Fake News")
        st.pyplot(plt)

    if st.button("Show Most Common Words (Real News)"):

        real_words = Counter(" ".join(real_text).split())
        common_real = real_words.most_common(20)

        words, counts = zip(*common_real)

        plt.figure()
        plt.bar(words, counts)
        plt.xticks(rotation=45)
        plt.title("Top Words in Real News")
        st.pyplot(plt)

# ==============================
# 3️⃣ MODEL ANALYSIS
# ==============================

elif menu == "Model Analysis":

    st.header("📈 Model Analysis")

    # Dummy dataset reload (same pipeline)
    df = pd.read_csv(TRAINING_DATA)

    TEXT_COL = "text" if "text" in df.columns else "content"
    df = df.dropna(subset=[TEXT_COL, "label"])

    df["full_text"] = df[TEXT_COL]

    X = vectorizer.transform(df["full_text"])
    y = df["label"]

    # --------------------------
    # CONFUSION MATRIX
    # --------------------------
    if st.button("Show Confusion Matrix"):

        y_pred = model.predict(X)

        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(cm)

        fig, ax = plt.subplots()
        disp.plot(ax=ax)
        st.pyplot(fig)

    # --------------------------
    # ROC CURVE
    # --------------------------
    if st.button("Show ROC Curve"):

        y_prob = model.predict_proba(X)[:, 1]

        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0,1], [0,1], linestyle="--")
        plt.legend()
        plt.title("ROC Curve")
        st.pyplot(plt)

    # --------------------------
    # FEATURE IMPORTANCE (LOGISTIC)
    # --------------------------
    if st.button("Top 20 Important Words"):

        if not hasattr(model, "coef_"):
            st.info("Feature importance is only available for linear models with coefficients.")
            st.stop()

        feature_names = vectorizer.get_feature_names_out()
        coefs = model.coef_[0]

        top_indices = np.argsort(coefs)[-20:]

        words = [feature_names[i] for i in top_indices]

        plt.figure()
        plt.barh(words, coefs[top_indices])
        plt.title("Top 20 Important Words (Fake)")
        st.pyplot(plt)
