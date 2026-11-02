
# Spam Email / SMS Detection using Machine Learning

A beginner-friendly text-classification project using **TF-IDF + Multinomial Naive Bayes**.
It trains on the UCI SMS Spam Collection dataset and predicts whether a message is `spam` or `ham` (legitimate).

> The supplied UCI dataset contains SMS messages, not full email bodies. The same text-classification pipeline can be adapted to email datasets.

## Features
- Automatic dataset download (no manual extraction required)
- Text cleaning and label normalization
- TF-IDF feature extraction
- Multinomial Naive Bayes model
- Train/test split and classification metrics
- Confusion matrix and evaluation report
- Command-line prediction
- Optional Streamlit web app
- Saved model and vectorizer

## 1. Requirements
- Windows 10/11
- Python 3.10 or 3.11 recommended

## 2. Setup (PowerShell)
Open PowerShell in this project folder:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## 3. Download the dataset
```powershell
python download_dataset.py
```
This downloads and extracts the official UCI SMS Spam Collection dataset into `dataset/`.

## 4. Train and evaluate
```powershell
python train_model.py
```
The trained model is saved in `models/`. Metrics and a confusion-matrix image are saved in `reports/`.

## 5. Predict from the terminal
```powershell
python predict.py
```
Paste a message when prompted. Type `exit` to quit.

You can also pass a message directly:
```powershell
python predict.py "Congratulations! You won a free prize. Call now!"
```

## 6. Run the web app (optional)
```powershell
streamlit run app.py
```
Open the local URL printed in the terminal (usually http://localhost:8501).

## Project structure
```text
SpamEmailDetection/
├── app.py
├── config.py
├── download_dataset.py
├── predict.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── dataset/
├── models/
└── reports/
```

## Model workflow
1. Load labelled messages (`spam` / `ham`).
2. Split data into training and testing sets with stratification.
3. Convert text to numeric TF-IDF vectors.
4. Train Multinomial Naive Bayes.
5. Evaluate using accuracy, precision, recall, F1-score and confusion matrix.
6. Save the fitted pipeline for future predictions.

## Important notes
- Accuracy alone is not enough; check precision and recall for the spam class.
- A model trained on SMS data may not perform equally well on long-form emails.
- Do not use this beginner model as the only filter for real business email security.
