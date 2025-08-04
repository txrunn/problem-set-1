'''
PART 1: ETL the two datasets and save each in `data/` as .csv's
'''

from __future__ import annotations
import os, io, requests
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
UNIVERSE_URL = "https://www.dropbox.com/scl/fi/69syqjo6pfrt9123rubio/universe_lab6.feather?rlkey=h2gt4o6z9r5649wo6h6ud6dce&dl=1"
ARRESTS_URL  = "https://www.dropbox.com/scl/fi/wv9kthwbj4ahzli3edrd7/arrest_events_lab6.feather?rlkey=mhxozpazqjgmo6qqahc2vd0xp&dl=1"

def _read_feather_from_url(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    buf = io.BytesIO(r.content)
    return pd.read_feather(buf)

def run_etl():
    os.makedirs(DATA_DIR, exist_ok=True)

    pred_universe_raw = _read_feather_from_url(UNIVERSE_URL)
    arrest_events_raw = _read_feather_from_url(ARRESTS_URL)

    # Normalize dates per assignment
    if "filing_date" not in pred_universe_raw.columns or "filing_date" not in arrest_events_raw.columns:
        raise ValueError("Expected 'filing_date' in both sources; please confirm column names.")

    pred_universe_raw["arrest_date_univ"] = pd.to_datetime(pred_universe_raw["filing_date"])
    arrest_events_raw["arrest_date_event"] = pd.to_datetime(arrest_events_raw["filing_date"])
    pred_universe_raw = pred_universe_raw.drop(columns=["filing_date"])
    arrest_events_raw = arrest_events_raw.drop(columns=["filing_date"])

    # Save to data/
    uni_out = os.path.join(DATA_DIR, "pred_universe_raw.csv")
    arr_out = os.path.join(DATA_DIR, "arrest_events_raw.csv")
    pred_universe_raw.to_csv(uni_out, index=False)
    arrest_events_raw.to_csv(arr_out, index=False)

    print(f"[ETL] Saved: {uni_out}")
    print(f"[ETL] Saved: {arr_out}")
    return pred_universe_raw, arrest_events_raw

# Save both data frames to `data/` -> 'pred_universe_raw.csv', 'arrest_events_raw.csv'