
"""Train and evaluate an improved spam and phishing message classifier."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

from config import DATA_FILE, MODEL_FILE, REPORT_DIR, RANDOM_STATE


# Additional examples for learning phishing-related patterns.
# These are training examples, not independent test data.
EXTRA_SPAM_MESSAGES = [
    "Your account will be blocked today. Click this link immediately to verify your password.",
    "URGENT! Your bank account is suspended. Verify your details immediately.",
    "Congratulations! You won a cash prize. Click here to claim now.",
    "Your account has been locked. Update your password using this link.",
    "You have won a free gift. Send your bank details to receive it.",
    "Your ATM card will expire today. Click here to activate it.",
    "Unusual activity detected. Verify your account immediately.",
    "Claim your free reward now by entering your personal details.",
    "Your payment failed. Click this link to update your card information.",
    "You are selected for a cash prize. Reply with your bank details.",
    "Final warning! Your account will be deleted unless you verify now.",
    "Your package is on hold. Pay the delivery fee using this link.",
    "You have received a suspicious login alert. Verify your password now.",
    "Limited time offer! Claim your free mobile phone today.",
    "Your bank account requires verification. Submit your details immediately.",
    "Click here to win a lottery prize worth thousands.",
    "Your KYC has expired. Update your personal information immediately.",
    "We detected unauthorized access. Confirm your login credentials.",
    "Congratulations! You are eligible for a free cash reward.",
    "Your account is at risk. Click the link to prevent suspension."
]


def load_dataset():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_FILE}. "
            "Run python download_dataset.py first."
        )

    df = pd.read_csv(
        DATA_FILE,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8"
    )

    df = df.dropna(subset=["label", "message"])
    df["label"] = df["label"].str.lower().str.strip()
    df = df[df["label"].isin(["ham", "spam"])]
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["message"] != ""]
    df = df.drop_duplicates(subset=["label", "message"])

    if df.empty:
        raise ValueError("No valid labelled messages found.")

    return df


def build_model():
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                analyzer="word",
                sublinear_tf=True,
                max_features=10000,
                min_df=1
            )
        ),
        ("classifier", ComplementNB(alpha=0.5))
    ])


def get_spam_probabilities(model, messages):
    spam_index = list(
        model.named_steps["classifier"].classes_
    ).index("spam")

    return model.predict_proba(messages)[:, spam_index]


def choose_threshold(y_validation, probabilities):
    """Choose a threshold that maximizes validation F1-score."""

    best_threshold = 0.50
    best_f1 = -1

    for threshold in np.arange(0.10, 0.91, 0.01):
        predictions = np.where(
            probabilities >= threshold, "spam", "ham"
        )

        _, _, f1, _ = precision_recall_fscore_support(
            y_validation,
            predictions,
            labels=["spam"],
            average=None,
            zero_division=0
        )

        if f1[0] > best_f1:
            best_f1 = f1[0]
            best_threshold = float(threshold)

    return best_threshold


def main():
    df = load_dataset()

    print("\nOriginal dataset preview:")
    print(df.head())

    print("\nOriginal class distribution:")
    print(df["label"].value_counts())

    # Keep the original test set separate from extra training examples.
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    # Split training data again to tune the threshold.
    X_fit, X_validation, y_fit, y_validation = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train
    )

    extra_df = pd.DataFrame({
        "message": EXTRA_SPAM_MESSAGES,
        "label": ["spam"] * len(EXTRA_SPAM_MESSAGES)
    })

    # Add supplemental phishing examples only to training data.
    X_fit_augmented = pd.concat(
        [X_fit.reset_index(drop=True), extra_df["message"]],
        ignore_index=True
    )

    y_fit_augmented = pd.concat(
        [y_fit.reset_index(drop=True), extra_df["label"]],
        ignore_index=True
    )

    print("\nTraining with supplemental phishing examples...")

    validation_model = build_model()
    validation_model.fit(X_fit_augmented, y_fit_augmented)

    validation_probabilities = get_spam_probabilities(
        validation_model, X_validation
    )

    threshold = choose_threshold(
        y_validation, validation_probabilities
    )

    print(f"\nSelected spam threshold: {threshold:.2f}")

    # Train the final model using all original training data
    # plus the supplemental phishing examples.
    X_final = pd.concat(
        [X_train.reset_index(drop=True), extra_df["message"]],
        ignore_index=True
    )

    y_final = pd.concat(
        [y_train.reset_index(drop=True), extra_df["label"]],
        ignore_index=True
    )

    model = build_model()

    print("\nTraining final model...")
    model.fit(X_final, y_final)

    # Evaluate on the untouched original test set.
    test_probabilities = get_spam_probabilities(model, X_test)

    predictions = np.where(
        test_probabilities >= threshold, "spam", "ham"
    )

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        labels=["ham", "spam"],
        digits=4,
        zero_division=0
    )

    print(f"\nTest accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(report)

    # Save model and threshold.
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_FILE)

    threshold_file = REPORT_DIR / "spam_threshold.txt"
    threshold_file.write_text(
        str(threshold), encoding="utf-8"
    )

    report_file = REPORT_DIR / "classification_report.txt"
    report_file.write_text(
        f"Test accuracy: {accuracy:.4f}\n"
        f"Spam threshold: {threshold:.2f}\n\n"
        f"{report}",
        encoding="utf-8"
    )

    # Save confusion matrix.
    labels = ["ham", "spam"]

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    plt.figure(figsize=(6, 4))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.title("Improved Spam Detection - Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "confusion_matrix.png",
        dpi=150
    )

    plt.close()

    print(f"\nSaved model: {MODEL_FILE}")
    print(f"Saved threshold: {threshold_file}")
    print(f"Saved evaluation report: {report_file}")
    print(f"Saved confusion matrix: {REPORT_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
