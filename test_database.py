
from database import (
    initialize_database,
    save_message,
    get_all_messages,
    get_dashboard_stats
)

initialize_database()

save_message(
    "Congratulations! You won a free iPhone!",
    "spam",
    0.999,
    "2026-11-02 20:00:00"
)

print("MESSAGE HISTORY:")
print(get_all_messages())

print("\nDASHBOARD STATISTICS:")
print(get_dashboard_stats())
