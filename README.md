# Data Quality Radar

A beginner-friendly Streamlit app that inspects any CSV, runs quality checks, offers reversible fixes, and generates a short human summary you can paste into a ticket.

The app is also deployed on Render. You can try it here:-
👉 [Open Data Quality Radar](https://your-render-url.onrender.com)

## Features
- Auto type inference, missingness, duplicate detection, simple outlier flags
- Reversible fixes with a changelog
- Plain-English narrative that references real counts
- Matplotlib charts for missingness and distributions
- Optional Great Expectations export if the library is installed

## Quick start
```bash
# Option 1: pip
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run radar/app.py

# Option 2: uv
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
streamlit run radar/app.py
```

## Repo layout
```
radar/
  app.py
  dq_checks.py
  repair.py
  summarize.py
core/
  io.py
data/
  messy_people.csv
requirements.txt
```

## Acceptance tests
1) Upload the sample `messy_people.csv`. You should see a validation summary and a cleaned CSV download.
2) The narrative should include numeric facts like counts of missing values and duplicates removed.
3) The changelog JSON should allow you to reconstruct the original from the cleaned file.

## Notes
- Great Expectations is optional. If not installed, the app will still run full checks using pandas.
- Charts are rendered with matplotlib only.
