# Viewing the full application code

The complete Streamlit application lives in [`app.py`](../app.py), which currently spans roughly 4,000 lines. Because the file is large, the easiest ways to see the entire code are:

- **View directly in this repo:** open `app.py` in your editor or viewer of choice.
- **Print to the terminal:** run `python scripts/show_app_code.py` from the repository root to stream the whole file to stdout.
- **Save a copy for sharing:** run `python scripts/show_app_code.py > app_full_code.txt` to create a plain-text snapshot you can share or review offline.

The helper script does not modify the application; it simply reads and prints the file so you can inspect every line in one place.
