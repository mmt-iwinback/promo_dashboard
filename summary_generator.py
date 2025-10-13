import pandas as pd

def generate_summary_and_recommendations(df):
    summary = []
    recs = []

    # Total hits
    total_hits = len(df)
    summary.append(f"Total messages tracked: **{total_hits}**")

    # Competitors
    if 'competitor_name' in df.columns:
        competitors = df['competitor_name'].unique()
        number_of_competitors = len(competitors)
        summary.append(f"Total number of competitors: **{number_of_competitors}**")
        summary.append(f"Competitors: **{', '.join(competitors)}**")


    if 'vertical' in df.columns:
        for ver in df['vertical'].unique():
            vertical_competitors = df[df['vertical']==ver]['competitor_name'].unique()
            number_of_vertical_competitors = len(vertical_competitors)
            total_messages_in_vertical = len(df[df['vertical']==ver])
            summary.append(f"Total number of {ver} competitors: **{number_of_vertical_competitors}**")
            summary.append(f"{ver} Competitors: **{', '.join(vertical_competitors)}**")
            summary.append(f"Total messages in {ver}: **{total_messages_in_vertical}**")

    # Most common promotion
    if 'promotion_types' in df.columns:
        top_promo = df['promotion_types'].value_counts().idxmax()
        promo_count = df['promotion_types'].value_counts().max()
        summary.append(f"Most common promotion type: **{top_promo}** ({promo_count} times)")
        recs.append(f"Consider emphasizing `{top_promo}` in your upcoming campaigns based on volume.")

    # Most active day
    if 'local_created_at' in df.columns:
        df['local_created_at'] = pd.to_datetime(df['local_created_at'])
        top_day = df['local_created_at'].dt.date.value_counts().idxmax()
        day_count = df['local_created_at'].dt.date.value_counts().max()
        summary.append(f"Peak activity day: **{top_day}** with {day_count} messages")
        recs.append(f"Evaluate the content and timing on {top_day} to understand what drove high engagement.")

    # Category distribution
    if 'category' in df.columns:
        most_common_cat = df['category'].value_counts().idxmax()
        summary.append(f"Most common content category: **{most_common_cat}**")
        recs.append(f"Ensure variety in content strategy if one category like `{most_common_cat}` dominates.")

    # Classification insight
    if 'classification' in df.columns and len(df['classification'].unique()) == 1:
        summary.append(f"All messages are classified as: **{df['classification'].iloc[0]}**")
        recs.append("Review classification criteria to ensure it's capturing message diversity.")

    return summary, recs
