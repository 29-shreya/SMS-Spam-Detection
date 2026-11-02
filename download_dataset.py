
"""Professional SMS Spam Detection Application - SpamGuard."""

import streamlit as st
import joblib
import pandas as pd

from datetime import datetime

from config import MODEL_FILE, REPORT_DIR

from database import (
    initialize_database,
    save_message,
    get_all_messages
)


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="SpamGuard - SMS Spam Detection",
    page_icon="📩",
    layout="wide",
    initial_sidebar_state="expanded"
)

initialize_database()

THRESHOLD_FILE = REPORT_DIR / "spam_threshold.txt"


# =====================================================
# 2. MODEL PERFORMANCE METRICS
# =====================================================
# These are the previously reported evaluation results.
# They are not recalculated every time the application runs.

MODEL_METRICS = {
    "Accuracy": 96.90,
    "Spam Precision": 84.78,
    "Spam Recall": 91.41,
    "Spam F1-Score": 87.97
}


# =====================================================
# 3. CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main-title {
    font-size: 34px;
    font-weight: 750;
    margin-bottom: 5px;
}

.subtitle {
    color: #9299A5;
    font-size: 15px;
    margin-bottom: 25px;
}

div[data-testid="stMetric"] {
    background: rgba(128, 128, 128, 0.08);
    padding: 18px;
    border-radius: 12px;
    border: 1px solid rgba(128, 128, 128, 0.18);
}

div[data-testid="stMetricValue"] {
    font-weight: 700;
}

.stButton button {
    border-radius: 8px;
    font-weight: 600;
}

.section-heading {
    font-size: 23px;
    font-weight: 700;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 4. LOAD MODEL AND THRESHOLD
# =====================================================

@st.cache_resource
def get_model():

    if not MODEL_FILE.exists():
        return None

    return joblib.load(MODEL_FILE)


def get_threshold():

    if not THRESHOLD_FILE.exists():
        return 0.5

    try:
        return float(
            THRESHOLD_FILE.read_text(
                encoding="utf-8"
            ).strip()
        )
    except (ValueError, OSError):
        return 0.5


model = get_model()
threshold = get_threshold()


# =====================================================
# 5. SUSPICIOUS KEYWORDS
# =====================================================

SUSPICIOUS_KEYWORDS = [
    "urgent", "verify", "password", "account",
    "blocked", "click here", "winner", "won",
    "free", "prize", "claim", "bank",
    "limited time", "congratulations", "otp",
    "suspended", "immediately", "payment"
]


def find_suspicious_keywords(message):

    message_lower = message.lower()

    return [
        word for word in SUSPICIOUS_KEYWORDS
        if word in message_lower
    ]


# =====================================================
# 6. MESSAGE PREDICTION
# =====================================================

def classify_message(message):

    if model is None:
        raise RuntimeError(
            "Trained model not found. Please train the model first."
        )

    probabilities = model.predict_proba([message])[0]

    classes = list(model.classes_)

    spam_index = None

    for index, label in enumerate(classes):

        if str(label).strip().lower() in [
            "spam", "1", "true"
        ]:
            spam_index = index
            break

    if spam_index is None:
        raise ValueError(
            "Could not identify the SPAM class in the trained model."
        )

    spam_probability = float(probabilities[spam_index])

    prediction = (
        "spam"
        if spam_probability >= threshold
        else "ham"
    )

    confidence = max(
        spam_probability,
        1 - spam_probability
    )

    return prediction, confidence, spam_probability


# =====================================================
# 7. DATABASE HELPERS
# =====================================================

def store_message(message, prediction, probability):

    save_message(
        message,
        prediction,
        probability,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


def get_history_dataframe():

    records = get_all_messages()

    columns = [
        "ID",
        "Message",
        "Prediction",
        "Spam Probability",
        "Date"
    ]

    if records is None or len(records) == 0:
        return pd.DataFrame(columns=columns)

    if isinstance(records, pd.DataFrame):

        df = records.copy()

        if len(df.columns) == 5:
            df.columns = columns

        return df

    return pd.DataFrame(records, columns=columns)


# =====================================================
# 8. SIDEBAR NAVIGATION
# =====================================================

with st.sidebar:

    st.title("📩 SpamGuard")

    st.caption("Machine Learning Detection System")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Message Detection",
            "Message History",
            "Model Performance",
            "About Project"
        ],
        key="navigation"
    )

    st.divider()

    st.markdown("### Model Information")

    st.write("**Algorithm:** Complement Naive Bayes")
    st.write("**Feature Extraction:** TF-IDF")
    st.write("**Database:** SQLite")

    st.divider()

    st.caption("Developed using Python and Streamlit")


# =====================================================
# 9. COMMON HEADER
# =====================================================

st.markdown(
    '<div class="main-title">📩 SMS Spam Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent message classification using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


# =====================================================
# 10. DASHBOARD
# =====================================================

if page == "Dashboard":

    st.header("📊 Dashboard Overview")

    history = get_history_dataframe()

    total_messages = len(history)

    if total_messages > 0:

        spam_count = (
            history["Prediction"]
            .astype(str)
            .str.lower()
            .eq("spam")
            .sum()
        )

        ham_count = total_messages - spam_count

    else:
        spam_count = 0
        ham_count = 0

    spam_percentage = (
        (spam_count / total_messages) * 100
        if total_messages > 0 else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Messages", total_messages)
    col2.metric("Spam Messages", int(spam_count))
    col3.metric("Ham Messages", int(ham_count))
    col4.metric("Spam Percentage", f"{spam_percentage:.2f}%")

    st.divider()

    st.subheader("📈 Message Classification Analysis")

    if total_messages > 0:

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:

            st.markdown("**Spam vs Ham Distribution**")

            distribution = pd.DataFrame({
                "Message Type": ["Spam", "Ham"],
                "Count": [spam_count, ham_count]
            })

            st.bar_chart(
                distribution.set_index("Message Type")
            )

        with chart_col2:

            st.markdown("**Message Count Comparison**")

            comparison = pd.DataFrame({
                "Message Type": ["Spam", "Ham"],
                "Messages": [spam_count, ham_count]
            })

            st.bar_chart(
                comparison.set_index("Message Type")
            )

        st.subheader("🕒 Recent Message Activity")

        st.dataframe(
            history.head(5),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No messages have been analyzed yet. "
            "Go to Message Detection to test your first message."
        )


# =====================================================
# 11. MESSAGE DETECTION
# =====================================================

elif page == "Message Detection":

    st.header("🔍 Message Detection")

    st.write(
        "Enter a message or upload a CSV file "
        "to identify spam and legitimate messages."
    )

    single_tab, bulk_tab = st.tabs([
        "Single Message Detection",
        "Bulk CSV Detection"
    ])

    # -----------------------------------------------
    # SINGLE MESSAGE
    # -----------------------------------------------

    with single_tab:

        st.subheader("Analyze a Single Message")

        message = st.text_area(
            "Enter your message:",
            height=150,
            placeholder=(
                "Example: Congratulations! "
                "You have won a free prize. Claim now!"
            )
        )

        if st.button(
            "Analyze Message",
            type="primary",
            use_container_width=True
        ):

            if not message.strip():

                st.warning("Please enter a message first.")

            elif model is None:

                st.error(
                    "Model file not found. "
                    "Please train your model first."
                )

            else:

                try:

                    prediction, confidence, spam_probability = (
                        classify_message(message)
                    )

                    store_message(
                        message,
                        prediction,
                        spam_probability
                    )

                    st.divider()

                    if prediction == "spam":

                        st.error("🚨 SPAM MESSAGE DETECTED")

                    else:

                        st.success("✅ LEGITIMATE MESSAGE (HAM)")

                    col1, col2 = st.columns(2)

                    col1.metric(
                        "Prediction",
                        prediction.upper()
                    )

                    col2.metric(
                        "Model Score",
                        f"{confidence * 100:.2f}%"
                    )

                    st.write(
                        f"Spam probability: "
                        f"**{spam_probability * 100:.2f}%**"
                    )

                    st.progress(spam_probability)

                    suspicious = find_suspicious_keywords(message)

                    if suspicious:

                        st.warning(
                            "Suspicious keywords found: "
                            + ", ".join(suspicious)
                        )

                    else:

                        st.info(
                            "No predefined suspicious keywords found."
                        )

                    st.caption(
                        "The model score is a probability output "
                        "from the classifier, not a guarantee "
                        "of real-world message safety."
                    )

                except Exception as error:

                    st.error(f"Prediction failed: {error}")

    # -----------------------------------------------
    # BULK CSV DETECTION
    # -----------------------------------------------

    with bulk_tab:

        st.subheader("Upload CSV File")

        st.write(
            "Upload a CSV file containing a column named "
            "**message**."
        )

        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=["csv"]
        )

        if uploaded_file is not None:

            try:

                input_df = pd.read_csv(uploaded_file)

                st.write("Uploaded Data Preview")

                st.dataframe(
                    input_df.head(),
                    use_container_width=True
                )

                if "message" not in input_df.columns:

                    st.error(
                        "CSV file must contain a column named 'message'."
                    )

                elif st.button(
                    "Analyze All Messages",
                    type="primary"
                ):

                    if model is None:

                        st.error("Trained model not found.")

                    else:

                        results = []

                        progress = st.progress(0)

                        messages = input_df["message"].fillna("").astype(str)

                        for index, text in enumerate(messages):

                            try:

                                prediction, confidence, probability = (
                                    classify_message(text)
                                )

                                store_message(
                                    text,
                                    prediction,
                                    probability
                                )

                                results.append({
                                    "Message": text,
                                    "Prediction": prediction.upper(),
                                    "Spam Probability": round(
                                        probability * 100, 2
                                    ),
                                    "Model Score": round(
                                        confidence * 100, 2
                                    )
                                })

                            except Exception as error:

                                results.append({
                                    "Message": text,
                                    "Prediction": "ERROR",
                                    "Spam Probability": None,
                                    "Model Score": None
                                })

                            progress.progress(
                                (index + 1) / len(messages)
                            )

                        result_df = pd.DataFrame(results)

                        st.success(
                            f"Analyzed {len(result_df)} messages."
                        )

                        st.dataframe(
                            result_df,
                            use_container_width=True,
                            hide_index=True
                        )

                        spam_total = (
                            result_df["Prediction"] == "SPAM"
                        ).sum()

                        ham_total = (
                            result_df["Prediction"] == "HAM"
                        ).sum()

                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Total Messages",
                            len(result_df)
                        )

                        col2.metric(
                            "Spam Messages",
                            int(spam_total)
                        )

                        col3.metric(
                            "Ham Messages",
                            int(ham_total)
                        )

                        csv_data = result_df.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            "⬇️ Download Results CSV",
                            data=csv_data,
                            file_name="spam_detection_results.csv",
                            mime="text/csv"
                        )

            except Exception as error:

                st.error(f"Could not process CSV file: {error}")


# =====================================================
# 12. MESSAGE HISTORY
# =====================================================

elif page == "Message History":

    st.header("📋 Message History")

    history = get_history_dataframe()

    if history.empty:

        st.info("No message history available yet.")

    else:

        col1, col2 = st.columns(2)

        with col1:

            search_text = st.text_input(
                "Search messages",
                placeholder="Enter a keyword..."
            )

        with col2:

            prediction_filter = st.selectbox(
                "Filter by prediction",
                ["All", "Spam", "Ham"]
            )

        filtered = history.copy()

        if search_text:

            filtered = filtered[
                filtered["Message"]
                .astype(str)
                .str.contains(
                    search_text,
                    case=False,
                    na=False
                )
            ]

        if prediction_filter != "All":

            filtered = filtered[
                filtered["Prediction"]
                .astype(str)
                .str.lower()
                == prediction_filter.lower()
            ]

        st.write(
            f"Showing {len(filtered)} of {len(history)} messages."
        )

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True
        )

        history_csv = filtered.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Message History",
            data=history_csv,
            file_name="spam_message_history.csv",
            mime="text/csv"
        )


# =====================================================
# 13. MODEL PERFORMANCE
# =====================================================

elif page == "Model Performance":

    st.header("📈 Model Performance")

    st.write(
        "Evaluation results from the trained SMS Spam Detection model."
    )

    st.info(
        "The following values are the previously reported "
        "evaluation metrics. They are not recalculated from "
        "the current message history."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{MODEL_METRICS['Accuracy']:.2f}%"
    )

    col2.metric(
        "Spam Precision",
        f"{MODEL_METRICS['Spam Precision']:.2f}%"
    )

    col3.metric(
        "Spam Recall",
        f"{MODEL_METRICS['Spam Recall']:.2f}%"
    )

    col4.metric(
        "Spam F1-Score",
        f"{MODEL_METRICS['Spam F1-Score']:.2f}%"
    )

    st.divider()

    st.subheader("Metric Comparison")

    performance_df = pd.DataFrame({
        "Metric": list(MODEL_METRICS.keys()),
        "Score (%)": list(MODEL_METRICS.values())
    })

    st.bar_chart(
        performance_df.set_index("Metric")
    )

    st.subheader("Understanding the Metrics")

    st.markdown("""
    - **Accuracy:** Percentage of all messages classified correctly.
    - **Precision:** Of the messages predicted as spam, how many were actually spam.
    - **Recall:** Of all actual spam messages, how many were detected.
    - **F1-Score:** Harmonic mean of precision and recall.
    """)

    st.divider()

    st.subheader("Model Configuration")

    config_col1, config_col2 = st.columns(2)

    with config_col1:

        st.write("**Algorithm:** Complement Naive Bayes")
        st.write("**Feature Extraction:** TF-IDF")
        st.write("**Classification:** Binary classification")

    with config_col2:

        st.write("**Decision Threshold:**", threshold)
        st.write("**Model File:**", str(MODEL_FILE))
        st.write("**Model Loaded:**", "Yes" if model is not None else "No")


# =====================================================
# 14. ABOUT PROJECT
# =====================================================

elif page == "About Project":

    st.header("ℹ️ About SpamGuard")

    st.write("""
    **SpamGuard** is a machine learning-based SMS spam
    detection system. It classifies text messages into
    two categories: Spam and Ham (legitimate messages).

    The project uses Natural Language Processing (NLP)
    techniques and a Complement Naive Bayes classifier.
    """)

    st.divider()

    st.subheader("🛠️ Technology Stack")

    technology_df = pd.DataFrame({
        "Technology": [
            "Python",
            "Pandas",
            "Scikit-learn",
            "TF-IDF",
            "Complement Naive Bayes",
            "Streamlit",
            "SQLite",
            "Joblib"
        ],
        "Purpose": [
            "Programming language",
            "Data processing",
            "Machine learning",
            "Text feature extraction",
            "Message classification",
            "Web application",
            "Message history storage",
            "Model saving and loading"
        ]
    })

    st.dataframe(
        technology_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("⚙️ How It Works")

    st.markdown("""
    1. User enters an SMS message or uploads a CSV file.
    2. TF-IDF converts text into numerical features.
    3. Complement Naive Bayes analyzes the message.
    4. The classification threshold determines Spam or Ham.
    5. The prediction is saved in the SQLite database.
    6. Users can view message history and download results.
    """)

    st.divider()

    st.subheader("✨ Main Features")

    st.markdown("""
    - Single SMS message detection.
    - Bulk CSV message classification.
    - Spam probability display.
    - Suspicious keyword identification.
    - SQLite message history.
    - Search and filter message records.
    - Downloadable CSV reports.
    - Model performance dashboard.
    """)

    st.divider()

    st.subheader("👩‍💻 Developer")

    st.write("**Project:** SpamGuard - SMS Spam Detection")
    st.write("**GitHub:** https://github.com/29-shreya")
    st.write("**Role:** Python Developer / AI & ML")


# =====================================================
# 15. FOOTER
# =====================================================

st.divider()

st.caption(
    "SpamGuard | SMS Spam Detection using Machine Learning"
)
