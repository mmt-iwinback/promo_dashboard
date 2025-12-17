# --- BONUS REPORTS HELPERS ---

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def _prepare_offer_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["type"] = out["type"].fillna("No Promo")
    out["value"] = out["value"].fillna("No Value")
    out["offer_label"] = out["type"] + " - " + out["value"]
    # numeric value (best-effort parse for % / FS / amounts)
    out["numeric_value"] = (
        out["value"].astype(str).str.extract(r"([\d]+(?:\.\d+)?)")[0].astype(float)
    )
    out["local_created_at"] = pd.to_datetime(out["local_created_at"], errors="coerce")
    out["date"] = out["local_created_at"].dt.date
    out["month"] = out["local_created_at"].dt.to_period("M").astype(str)
    return out


def _agg_subjects(df: pd.DataFrame) -> pd.DataFrame:
    # Competitor x Offer → instances + aggregated subjects (fixes "?" in hover)
    g = (
        df.groupby(["competitor_name", "offer_label"], dropna=False)
        .agg(
            instances=("tracking_hit_id", "count"),
            subjects=("subject", lambda x: "; ".join(pd.Series(x).dropna().unique())[:2000])
        )
        .reset_index()
    )
    return g

def compute_bundles(df: pd.DataFrame, min_size: int = 2):
    """
    Returns:
      bundles_df: bundle_label, bundle_size, count
      per_hit_df: tracking_hit_id, bundle_label, bundle_size
    """
    d = _prepare_offer_cols(df)

    # 1) Get unique offers per hit
    per_hit = (
        d.groupby("tracking_hit_id")["offer_label"]
         .apply(lambda s: tuple(sorted(pd.unique(s.dropna()))))
         .reset_index(name="offers_tuple")
    )
    # 2) Keep bundles with size >= min_size
    per_hit["bundle_size"] = per_hit["offers_tuple"].apply(len)
    per_hit = per_hit[per_hit["bundle_size"] >= min_size].copy()

    # 3) Create a readable label (joined by " | ")
    def _label(tup, max_len=180):
        lab = " | ".join(tup)
        return (lab[:max_len] + "…") if len(lab) > max_len else lab

    per_hit["bundle_label"] = per_hit["offers_tuple"].apply(_label)

    # 4) Count bundle frequency
    bundles = (
        per_hit.groupby(["bundle_label", "bundle_size"])
               .size()
               .reset_index(name="count")
               .sort_values(["bundle_size", "count"], ascending=[False, False])
    )
    return bundles, per_hit



import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def preprocess_holiday_data(df):

    # Ensure datetime
    df = df.copy()

    df["local_created_at"] = pd.to_datetime(df["local_created_at"])

    # -------- HOLIDAY TYPE DETECTION ----------
    def detect_holiday(row):
        h = str(row.get("holiday") or "").lower()
        text = (
            str(row.get("subject") or "")
            + " "
            + str(row.get("translated_content") or "")
        ).lower()

        # direct holiday column
        if "black" in h and ("friday" in h or "week" in h):
            return "Black Friday"
        if "christmas" in h or "advent" in h or "holiday season" in h:
            return "Christmas"

        # text patterns
        if any(k in text for k in ["black friday", "cyber monday", "thanksgiving"]):
            return "Black Friday"
        if any(k in text for k in ["christmas", "xmas"]):
            return "Christmas"

        # emoji-based
        if any(e in text for e in ["🎄", "🎅", "❄", "⛄", "🧦"]):
            return "Christmas"
        if "🦃" in text:
            return "Black Friday"

        return np.nan

    df["holiday_type"] = df.apply(detect_holiday, axis=1)

    # -------- BUILD OFFER TEXT ----------
    def build_offer(row):
        t = str(row.get("type") or "")
        v = str(row.get("value") or "")
        pv = str(row.get("play_value") or "")
        base = v or pv or ""
        if t and base:
            return f"{t}: {base}"
        if base:
            return base
        if t:
            return t
        return None

    df["offer_text"] = df.apply(build_offer, axis=1)

    # -------- FIRST SEASONAL MENTION PER COMPETITOR ----------
    records = []
    for comp, subdf in df.groupby("competitor_name"):
        for holiday in ["Black Friday", "Christmas"]:
            hdf = subdf[subdf["holiday_type"] == holiday]
            if hdf.empty:
                continue

            first_idx = hdf["local_created_at"].idxmin()
            first_row = df.loc[first_idx]

            records.append({
                "competitor": comp,
                "holiday_type": holiday,
                "first_mention_date": first_row["local_created_at"],
                "first_subject": first_row.get("subject", ""),
                "first_offer": first_row.get("offer_text", ""),
                "first_wager": first_row.get("play_value", ""),
                "first_requirements": first_row.get("deposit_requirement", ""),
                "total_holiday_comms": hdf["tracking_hit_id"].nunique(),
            })

    first_mentions = pd.DataFrame(records)

    # -------- SEASONAL ACTIVITY MATRIX (for bar chart + heatmap) ----------
    activity = (
        first_mentions.pivot_table(
            index="competitor",
            columns="holiday_type",
            values="total_holiday_comms",
            fill_value=0,
            aggfunc="sum"
        )
        .sort_index()
    )


    return df, first_mentions, activity

def competitor_holiday_promos_table(df):
    """
    Returns a table with one row per holiday email (per tracking_hit_id),
    including competitor, holiday type, date, subject, offer, wager, and requirements.
    """

    # Reuse your preprocessing (adds 'holiday_type' and 'offer_text')
    df["type"] = df["type"].fillna("No Promo")
    df["value"] = df["value"].fillna("No Value")
    df["offer_label"] = df["type"] + " - " + df["value"]

    d = df.copy()

    # Keep only Black Friday / Christmas emails
    h = d['holiday'].astype(str).str.strip()

    # Normalize obvious values
    # --- SAFE normalize holiday column (handles float/NaN correctly) ---
    def normalize_holiday(row_h, subject, translated):
        # Convert only real strings, otherwise empty
        base = row_h if isinstance(row_h, str) else ""

        text = (str(subject) + " " + str(translated)).lower()

        # If holiday column already has a value → normalize/upgrade it
        if base:
            low = base.lower()
            if "black" in low and ("friday" in low or "week" in low):
                return "Black Friday"
            if "christmas" in low or "advent" in low or "holiday season" in low:
                return "Christmas"
            return base  # keep other labels (Anniversary, Weekend, etc.)

        # If holiday empty, upgrade from text patterns
        if any(k in text for k in ["black friday", "blackfriday", "cyber monday", "thanksgiving"]):
            return "Black Friday"
        if any(k in text for k in ["christmas", "xmas"]):
            return "Christmas"

        return ""  # no holiday detected

    d["holiday"] = d.apply(
        lambda r: normalize_holiday(r.get("holiday"), r.get("subject"), r.get("translated_content")),
        axis=1
    )
    d=d[d["holiday"].isin(["Black Friday", "Christmas"])]
    if d.empty:
        return d  # no holiday promos

    # Aggregate per competitor + tracking_hit_id + holiday
    #(one row per distinct email/communication)
    table = (
        d.groupby(["competitor_name", "tracking_hit_id", "holiday"], as_index=False)
        .agg(
            date=("local_created_at", "min"),
            subject=("subject", "first"),
            offer=("value", lambda x: ", ".join(sorted({v for v in x if v}))),
            wager=("play_value", lambda x: ", ".join(sorted({str(v) for v in x if pd.notna(v)}))),
            requirements=("deposit_requirement", lambda x: ", ".join(sorted({str(v) for v in x if pd.notna(v)}))),
        )
        .sort_values(["competitor_name", "date"])
    )

    return table
