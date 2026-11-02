
"""SQLite database operations for SMS Spam Detection."""

import sqlite3
from pathlib import Path


# Database will be created automatically in the project folder
BASE_DIR = Path(__file__).resolve().parent
DATABASE_FILE = BASE_DIR / "spam_history.db"


# -----------------------------------------
# 1. CONNECT TO DATABASE
# -----------------------------------------

def get_connection():
    return sqlite3.connect(DATABASE_FILE)


# -----------------------------------------
# 2. CREATE DATABASE TABLE
# -----------------------------------------

def initialize_database():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS message_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT NOT NULL,
            prediction TEXT NOT NULL,
            spam_probability REAL NOT NULL,
            date TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


# -----------------------------------------
# 3. SAVE MESSAGE
# -----------------------------------------

def save_message(message, prediction, spam_probability, date):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO message_history
        (message, prediction, spam_probability, date)
        VALUES (?, ?, ?, ?)
    """, (
        message,
        prediction,
        spam_probability,
        date
    ))

    connection.commit()
    connection.close()


# -----------------------------------------
# 4. GET ALL MESSAGE HISTORY
# -----------------------------------------

def get_all_messages():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT id, message, prediction, spam_probability, date
        FROM message_history
        ORDER BY id DESC
    """)

    records = cursor.fetchall()

    connection.close()

    return records


# -----------------------------------------
# 5. GET DASHBOARD STATISTICS
# -----------------------------------------

def get_dashboard_stats():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            COUNT(*),
            SUM(CASE WHEN prediction = 'spam' THEN 1 ELSE 0 END),
            SUM(CASE WHEN prediction = 'ham' THEN 1 ELSE 0 END)
        FROM message_history
    """)

    result = cursor.fetchone()

    connection.close()

    total = result[0] or 0
    spam = result[1] or 0
    ham = result[2] or 0

    spam_percentage = (
        spam / total * 100 if total > 0 else 0
    )

    return total, spam, ham, spam_percentage
