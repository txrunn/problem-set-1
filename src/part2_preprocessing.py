'''
PART 2: Pre-processing
- Take the time to understand the data before proceeding
- Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
- Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
- Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 
- - So if a person was arrested on 2016-09-11, you would check to see if there was a felony arrest for that person between 2016-09-12 and 2017-09-11.
- - Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?
- Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise. 
- - Use a print statment to print this question and its answer: What share of current charges are felonies?
- Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number arrests in the one year prior to the current charge. 
- - So if someone was arrested on 2016-09-11, then you would check to see if there was a felony arrest for that person between 2015-09-11 and 2016-09-10.
- - Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?
- Print the mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()
- Print pred_universe.head()
- Return `df_arrests` for use in main.py for PART 3; if you can't figure this out, save as a .csv in `data/` and read into PART 3 in main.py
'''

from __future__ import annotations
import os
from datetime import timedelta
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
UNI_IN = os.path.join(DATA_DIR, "pred_universe_raw.csv")
ARR_IN = os.path.join(DATA_DIR, "arrest_events_raw.csv")
OUT_ARRESTS = os.path.join(DATA_DIR, "df_arrests.csv")

def _require_cols(df: pd.DataFrame, needed: list[str], where: str):
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {where}: {missing}")

def run_preprocessing() -> pd.DataFrame:
    pred_universe = pd.read_csv(UNI_IN, parse_dates=["arrest_date_univ"])
    arrest_events = pd.read_csv(ARR_IN, parse_dates=["arrest_date_event"])

    # Required columns
    _require_cols(pred_universe, ["person_id", "arrest_date_univ"], "pred_universe_raw.csv")
    _require_cols(arrest_events, ["person_id", "arrest_date_event"], "arrest_events_raw.csv")

    # If felony flag exists in universe (current charge), use; else default to 0
    cur_fel_col = None
    for c in ["current_charge_felony", "is_felony", "felony_flag", "felony", "charge_severity"]:
        if c in pred_universe.columns:
            cur_fel_col = c
            break

    df_arrests = pred_universe[["person_id", "arrest_date_univ"]].copy()
    if cur_fel_col is None:
        df_arrests["current_charge_felony"] = 0
    else:
        s = pred_universe[cur_fel_col]
        if s.dtype == bool:
            df_arrests["current_charge_felony"] = s.astype(int).values
        else:
            df_arrests["current_charge_felony"] = (
                s.astype(str).str.strip().str.upper().isin(["1","Y","T","TRUE","F","FELONY"])
            ).astype(int).values

    # Full outer join for any additional context (not strictly required to compute features)
    df_outer = pd.merge(
        pred_universe, arrest_events,
        on="person_id", how="outer", suffixes=("_univrow", "_eventrow")
    )
    # We keep df_arrests (one row per current arrest in universe) as the modeling frame.

    # Helper: group events by person for fast lookups (ensure felony indicator in events)
    fel_event_col = None
    for c in ["is_felony", "felony_flag", "felony", "charge_severity", "current_charge_felony"]:
        if c in arrest_events.columns:
            fel_event_col = c
            break
    if fel_event_col is None:
        raise ValueError("Could not find a felony indicator column in arrest_events_raw.csv.")

    ev = arrest_events.copy()
    s = ev[fel_event_col]
    if s.dtype == bool:
        ev["felony_event"] = s.astype(int)
    else:
        ev["felony_event"] = (
            s.astype(str).str.strip().str.upper().isin(["1","Y","T","TRUE","F","FELONY"])
        ).astype(int)

    ev = ev.sort_values(["person_id", "arrest_date_event"])
    g = ev.groupby("person_id", sort=False)

    # y: felony rearrest in (t+1, t+365]
    def _future_felony(pid, t):
        try:
            dfp = g.get_group(pid)
        except KeyError:
            return 0
        lower = t + timedelta(days=1)
        upper = t + timedelta(days=365)
        m = (dfp["arrest_date_event"] >= lower) & (dfp["arrest_date_event"] <= upper) & (dfp["felony_event"] == 1)
        return int(m.any())

    # num_fel_arrests_last_year: felony arrests in [t-365, t-1]
    def _past_felony_count(pid, t):
        try:
            dfp = g.get_group(pid)
        except KeyError:
            return 0
        lower = t - timedelta(days=365)
        upper = t - timedelta(days=1)
        m = (dfp["arrest_date_event"] >= lower) & (dfp["arrest_date_event"] <= upper) & (dfp["felony_event"] == 1)
        return int(m.sum())

    df_arrests["y"] = [
        _future_felony(pid, t) for pid, t in zip(df_arrests["person_id"], df_arrests["arrest_date_univ"])
    ]
    df_arrests["num_fel_arrests_last_year"] = [
        _past_felony_count(pid, t) for pid, t in zip(df_arrests["person_id"], df_arrests["arrest_date_univ"])
    ]

    # Required prints
    print("What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?")
    print(f"Answer: {df_arrests['y'].mean():.3f}")

    print("What share of current charges are felonies?")
    print(f"Answer: {df_arrests['current_charge_felony'].mean():.3f}")

    print("What is the average number of felony arrests in the last year?")
    print(f"Answer: {df_arrests['num_fel_arrests_last_year'].mean():.3f}")

    print("Mean of 'num_fel_arrests_last_year':", df_arrests["num_fel_arrests_last_year"].mean())
    print("df_arrests.head():")
    print(df_arrests.head())

    df_arrests.to_csv(OUT_ARRESTS, index=False)
    print(f"[Preprocessing] Saved: {OUT_ARRESTS}")
    return df_arrests
