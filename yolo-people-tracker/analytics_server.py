from flask import Flask, render_template
from pathlib import Path
import pandas as pd

app = Flask(__name__)

LOG_PATH = Path("people_log.csv")


def load_data():
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"{LOG_PATH} not found. Run the people counter first.")
    df = pd.read_csv(LOG_PATH)

    # Basic safety: sort by time
    df = df.sort_values("timestamp_sec").reset_index(drop=True)
    return df


@app.route("/")
def dashboard():
    df = load_data()

    # High-level summary
    total_entries = int(df["entries_total"].iloc[-1])
    max_in_zone = int(df["people_in_zone"].max())
    avg_in_zone = float(df["people_in_zone"].mean())
    duration_sec = float(df["timestamp_sec"].iloc[-1])

    # Time series for chart
    timestamps = df["timestamp_sec"].round(2).tolist()
    people_in_zone = df["people_in_zone"].tolist()
    people_total = df["people_total"].tolist()

    return render_template(
        "dashboard.html",
        total_entries=total_entries,
        max_in_zone=max_in_zone,
        avg_in_zone=avg_in_zone,
        duration_sec=duration_sec,
        timestamps=timestamps,
        people_in_zone=people_in_zone,
        people_total=people_total,
    )


if __name__ == "__main__":
    # Run local server: http://127.0.0.1:5000/
    app.run(debug=True)
