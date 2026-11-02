from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
DATA_FILE = DATA_DIR / "SMSSpamCollection"
MODEL_FILE = MODEL_DIR / "spam_classifier.joblib"
RANDOM_STATE = 42

for folder in (DATA_DIR, MODEL_DIR, REPORT_DIR):
    folder.mkdir(parents=True, exist_ok=True)
