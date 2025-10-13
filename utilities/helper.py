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
