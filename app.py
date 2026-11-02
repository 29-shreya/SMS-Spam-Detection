
"""Professional SMS Spam Detection Application."""

import streamlit as st
import joblib
import pandas as pd

from datetime import datetime

from config import MODEL_FILE, REPORT_DIR

from database import (
    initialize_database,
    save_message,
    get_all_messages,
    get_dashboard_stats
)


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="📩",
    layout="wide",
    initial_sidebar_state="expanded"
)

initialize_database()

THRESHOLD_FILE = REPORT_DIR / "spam_threshold.txt"


# =====================================================
# 2. CUSTOM CSS
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
# 3. LOAD MODEL AND THRESHOLD
# =====================================================

@st.cache_resource
def get_model():

    if not MODEL_FILE.exists():
        return None

    return joblib.load(MODEL_FILE)


def get_threshold():

    if not THRESHOLD_FILE.exists():
        return 0.5

    return float(
        THRESHOLD_FILE.read_text(
            encoding="utf-8"
        ).strip()
    )


model = get_model()
threshold = get_threshold()


# =====================================================
# 4. PREDICTION FUNCTION
# =====================================================

def predict_message(message):

    probabilities = model.predict_proba([message])[0]

    classes = list(
        model.named_steps["classifier"].classes_
    )

    spam_index = classes.index("spam")

    spam_probability = float(
        probabilities[spam_index]
    )

    if spam_probability >= threshold:
        prediction = "spam"
        confidence = spam_probability
    else:
        prediction = "ham"
        confidence = 1 - spam_probability

    return prediction, confidence, spam_probability


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
# 6. DATABASE HELPERS
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

    return pd.DataFrame(records, columns=columns)


# =====================================================
# 7. SIDEBAR NAVIGATION
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
            "About Project"
        ]
    )

    st.divider()

    st.markdown("### Model Information")

    st.write("**Algorithm:** Complement Naive Bayes")
    st.write("**Feature Extraction:** TF-IDF")
    st.write("**Database:** SQLite")

    st.divider()

    st.caption("Developed using Python and Streamlit")


# =====================================================
# 8. APPLICATION HEADER
# =====================================================

st.markdown(
    '<div class="main-title">📩 SMS Spam Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Intelligent message classification using Machine Learning</div>',
    unsafe_allow_html=True
)

if model is None:

    st.error(
        "Trained model not found. Please check your models folder."
    )

    st.stop()


# =====================================================
# 9. DASHBOARD PAGE
# =====================================================

if page == "Dashboard":

    st.header("📊 Dashboard Overview")

    total, spam_count, ham_count, spam_percentage = (
        get_dashboard_stats()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Messages",
        total
    )

    col2.metric(
        "Spam Messages",
        spam_count
    )

    col3.metric(
        "Ham Messages",
        ham_count
    )

    col4.metric(
        "Spam Percentage",
        f"{spam_percentage:.2f}%"
    )

    st.divider()

    st.subheader("📈 Message Classification Analysis")

    if total > 0:

        chart_data = pd.DataFrame({
            "Message Type": ["Spam", "Ham"],
            "Count": [spam_count, ham_count]
        })

        left, right = st.columns(2)

        with left:

            st.markdown("**Spam vs Ham Distribution**")

            st.vega_lite_chart(
                chart_data,
                {
                    "mark": {
                        "type": "arc",
                        "innerRadius": 55
                    },
                    "encoding": {
                        "theta": {
                            "field": "Count",
                            "type": "quantitative"
                        },
                        "color": {
                            "field": "Message Type",
                            "type": "nominal",
                            "scale": {
                                "domain": ["Spam", "Ham"],
                                "range": ["#E05252", "#36A77A"]
                            }
                        },
                        "tooltip": [
                            {
                                "field": "Message Type",
                                "type": "nominal"
                            },
                            {
                                "field": "Count",
                                "type": "quantitative"
                            }
                        ]
                    }
                },
                use_container_width=True
            )

        with right:

            st.markdown("**Message Count Comparison**")

            st.bar_chart(
                chart_data.set_index("Message Type")
            )

    else:

        st.info(
            "No records available. Start checking messages "
            "to populate your dashboard."
        )

    st.divider()

    st.subheader("🕒 Recent Message Activity")

    history = get_history_dataframe()

    if not history.empty:

        recent = history.head(5).copy()

        recent["Prediction"] = (
            recent["Prediction"].str.upper()
        )

        recent["Spam Probability"] = (
            recent["Spam Probability"] * 100
        ).round(2).astype(str) + "%"

        st.dataframe(
            recent,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info("No message history available.")


# =====================================================
# 10. MESSAGE DETECTION PAGE
# =====================================================

elif page == "Message Detection":

    st.header("🔍 Message Detection")

    tab1, tab2 = st.tabs([
        "Single Message",
        "Bulk CSV Detection"
    ])

    # SINGLE MESSAGE DETECTION

    with tab1:

        st.subheader("Check Individual Message")

        message = st.text_area(
            "Enter your SMS message:",
            placeholder="Example: Congratulations! You won a free iPhone...",
            height=130
        )

        if st.button(
            "Analyze Message",
            type="primary",
            key="single_prediction"
        ):

            if not message.strip():

                st.warning("Please enter a message.")

            else:

                clean_message = message.strip()

                prediction, confidence, spam_probability = (
                    predict_message(clean_message)
                )

                store_message(
                    clean_message,
                    prediction,
                    spam_probability
                )

                st.session_state["last_result"] = {
                    "message": clean_message,
                    "prediction": prediction,
                    "confidence": confidence,
                    "spam_probability": spam_probability,
                    "keywords": find_suspicious_keywords(
                        clean_message
                    )
                }

        if "last_result" in st.session_state:

            result = st.session_state["last_result"]

            st.divider()

            st.subheader("Prediction Result")

            if result["prediction"] == "spam":

                st.error("🚨 SPAM MESSAGE DETECTED")

            else:

                st.success("✅ LEGITIMATE MESSAGE (HAM)")

            col1, col2 = st.columns(2)

            col1.metric(
                "Spam Probability",
                f"{result['spam_probability'] * 100:.2f}%"
            )

            col2.metric(
                "Prediction Confidence",
                f"{result['confidence'] * 100:.2f}%"
            )

            st.write("**Message:**", result["message"])

            if result["keywords"]:

                st.warning(
                    "Suspicious keywords: "
                    + ", ".join(result["keywords"])
                )

            else:

                st.info("No suspicious keywords detected.")

    # BULK CSV DETECTION

    with tab2:

        st.subheader("Bulk Message Classification")

        st.write(
            "Upload a CSV file containing a column named "
            "'message'."
        )

        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=["csv"],
            key="bulk_upload"
        )

        if uploaded_file is not None:

            try:

                uploaded_df = pd.read_csv(uploaded_file)

                if "message" not in uploaded_df.columns:

                    st.error(
                        "Your CSV must contain a 'message' column."
                    )

                else:

                    st.write("Dataset Preview")

                    st.dataframe(
                        uploaded_df.head(),
                        use_container_width=True
                    )

                    if st.button(
                        "Analyze All Messages",
                        type="primary",
                        key="bulk_analyze"
                    ):

                        results = []

                        progress = st.progress(0)

                        total_rows = len(uploaded_df)

                        for index, msg in enumerate(
                            uploaded_df["message"].fillna("").astype(str)
                        ):

                            clean_message = msg.strip()

                            if clean_message:

                                prediction, confidence, probability = (
                                    predict_message(clean_message)
                                )

                                store_message(
                                    clean_message,
                                    prediction,
                                    probability
                                )

                            else:

                                prediction = "ham"
                                confidence = 1.0
                                probability = 0.0

                            results.append({
                                "Message": msg,
                                "Prediction": prediction.upper(),
                                "Spam Probability (%)": round(
                                    probability * 100, 2
                                ),
                                "Confidence (%)": round(
                                    confidence * 100, 2
                                )
                            })

                            progress.progress(
                                (index + 1) / max(total_rows, 1)
                            )

                        st.session_state["bulk_results"] = (
                            pd.DataFrame(results)
                        )

                        st.success(
                            "Bulk analysis completed successfully!"
                        )

                    if "bulk_results" in st.session_state:

                        result_df = st.session_state["bulk_results"]

                        st.subheader("📊 Prediction Results")

                        st.dataframe(
                            result_df,
                            use_container_width=True,
                            hide_index=True
                        )

                        spam_total = int(
                            (result_df["Prediction"] == "SPAM").sum()
                        )

                        ham_total = int(
                            (result_df["Prediction"] == "HAM").sum()
                        )

                        c1, c2, c3 = st.columns(3)

                        c1.metric(
                            "Total",
                            len(result_df)
                        )

                        c2.metric(
                            "Spam",
                            spam_total
                        )

                        c3.metric(
                            "Ham",
                            ham_total
                        )

                        csv_data = result_df.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            "⬇️ Download Prediction Results",
                            data=csv_data,
                            file_name="spam_detection_results.csv",
                            mime="text/csv"
                        )

            except pd.errors.EmptyDataError:

                st.error("The CSV file is empty.")

            except Exception as error:

                st.error(f"Error: {error}")


# =====================================================
# 11. MESSAGE HISTORY PAGE
# =====================================================

elif page == "Message History":

    st.header("📋 Message History")

    history = get_history_dataframe()

    if history.empty:

        st.info("No message records found.")

    else:

        search_text = st.text_input(
            "🔎 Search message",
            placeholder="Enter words to search..."
        )

        selected_type = st.selectbox(
            "Filter by prediction",
            ["All", "SPAM", "HAM"]
        )

        filtered = history.copy()

        if search_text:

            filtered = filtered[
                filtered["Message"].str.contains(
                    search_text,
                    case=False,
                    na=False
                )
            ]

        if selected_type != "All":

            filtered = filtered[
                filtered["Prediction"].str.upper()
                == selected_type
            ]

        display_data = filtered.copy()

        display_data["Prediction"] = (
            display_data["Prediction"].str.upper()
        )

        display_data["Spam Probability"] = (
            display_data["Spam Probability"] * 100
        ).round(2).astype(str) + "%"

        st.write(
            f"**Showing {len(display_data)} of {len(history)} records**"
        )

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )

        history_csv = filtered.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Filtered History",
            data=history_csv,
            file_name="message_history.csv",
            mime="text/csv"
        )


# =====================================================
# 12. ABOUT PROJECT PAGE
# =====================================================

elif page == "About Project":

    st.header("ℹ️ About This Project")

    st.write(
        """
        SMS Spam Detection is a Machine Learning application
        that classifies text messages into two categories:
        Spam and Ham.
        """
    )

    st.subheader("Technology Stack")

    tech_data = pd.DataFrame({
        "Technology": [
            "Python",
            "Scikit-learn",
            "TF-IDF",
            "Complement Naive Bayes",
            "Streamlit",
            "SQLite",
            "Pandas"
        ],
        "Purpose": [
            "Programming language",
            "Machine Learning",
            "Text feature extraction",
            "Message classification",
            "Web application",
            "Data persistence",
            "Data processing"
        ]
    })

    st.dataframe(
        tech_data,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Main Features")

    st.markdown("""
    - Individual SMS spam detection
    - Bulk CSV message classification
    - Spam probability calculation
    - Suspicious keyword identification
    - SQLite message history
    - Interactive dashboard
    - Search and filter message records
    - Download prediction reports
    """)

    st.subheader("How It Works")

    st.markdown("""
    1. User enters a message or uploads a CSV file.
    2. TF-IDF converts text into numerical features.
    3. The trained Complement Naive Bayes model predicts the message category.
    4. The application displays SPAM or HAM.
    5. The prediction is saved in the SQLite database.
    """)


# =====================================================
# 13. FOOTER
# =====================================================

st.divider()

st.caption(
    "SMS Spam Detection System | Python | Machine Learning | "
    "Streamlit | SQLite"
)
