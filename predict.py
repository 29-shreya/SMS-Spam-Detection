
"""Predict whether a message is spam or ham."""

import sys
import joblib

from config import MODEL_FILE, REPORT_DIR


THRESHOLD_FILE = REPORT_DIR / "spam_threshold.txt"


def load_model():
    if not MODEL_FILE.exists():
        raise SystemExit(
            "Trained model not found. Run python train_model.py first."
        )

    if not THRESHOLD_FILE.exists():
        raise SystemExit(
            "Spam threshold not found. Run python train_model.py first."
        )

    model = joblib.load(MODEL_FILE)

    threshold = float(
        THRESHOLD_FILE.read_text(encoding="utf-8").strip()
    )

    return model, threshold


def predict_message(model, threshold, message):
    probabilities = model.predict_proba([message])[0]

    classes = list(model.named_steps["classifier"].classes_)
    spam_index = classes.index("spam")

    spam_probability = float(probabilities[spam_index])

    if spam_probability >= threshold:
        prediction = "spam"
        confidence = spam_probability
    else:
        prediction = "ham"
        confidence = 1 - spam_probability

    return prediction, confidence, spam_probability


def main():
    model, threshold = load_model()

    print("Spam Message Detector")
    print(f"Spam threshold: {threshold:.2f}")
    print("Type 'exit' to quit.")

    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:]).strip()

        if message:
            label, confidence, spam_probability = predict_message(
                model, threshold, message
            )

            print(
                f"Prediction: {label.upper()} | "
                f"Model confidence: {confidence:.1%}"
            )
            print(f"Spam probability: {spam_probability:.1%}")

        return

    while True:
        message = input("\nEnter a message: ").strip()

        if message.lower() == "exit":
            print("Goodbye!")
            break

        if not message:
            print("Please enter a message.")
            continue

        label, confidence, spam_probability = predict_message(
            model, threshold, message
        )

        print(
            f"Prediction: {label.upper()} | "
            f"Model confidence: {confidence:.1%}"
        )

        print(f"Spam probability: {spam_probability:.1%}")


if __name__ == "__main__":
    main()
