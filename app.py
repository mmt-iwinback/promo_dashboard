import streamlit as st
import pandas as pd
from chart_generator import generate_all_charts  # This will be created from your notebook logic
from summary_generator import generate_summary_and_recommendations
from utilities.openai_observer import get_observation
import io


st.set_page_config(page_title="CRMTracker Promo Dashboard", layout="wide")
st.title("CRMTracker Promo Dashboard")
st.markdown("Upload your Excel file to generate and download promotional charts.")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("Data uploaded successfully.")

        # --- Sidebar filters ---
        st.sidebar.header("🔍 Filter Data")


        if 'vertical' in df.columns:
            selected_verticals = st.sidebar.multiselect("Select Vertical",
                                                       options=df['vertical'].dropna().unique())
            if selected_verticals:
                df = df[df['vertical'].isin(selected_verticals)]

        if 'competitor_name' in df.columns:
            selected_comps = st.sidebar.multiselect("Select Competitor(s)", options=df['competitor_name'].dropna().unique())
            if selected_comps:
                df = df[df['competitor_name'].isin(selected_comps)]

        if 'lifecycle' in df.columns:
            selected_life = st.sidebar.multiselect("Select Lifecycle(s)", options=df['lifecycle'].dropna().unique())
            if selected_life:
                df = df[df['lifecycle'].isin(selected_life)]

        if 'category' in df.columns:
            selected_cats = st.sidebar.multiselect("Select Category(ies)", options=df['category'].dropna().unique())
            if selected_cats:
                df = df[df['category'].isin(selected_cats)]

        if 'promotion_types' in df.columns:
            selected_promos = st.sidebar.multiselect("Select Promotion Types", options=df['promotion_types'].dropna().unique())
            if selected_promos:
                df = df[df['promotion_types'].isin(selected_promos)]

        if 'channel_type' in df.columns:
            selected_channels = st.sidebar.multiselect("Select Channel Types", options=df['channel_type'].dropna().unique())
            if selected_channels:
                df = df[df['channel_type'].isin(selected_channels)]

        if 'country' in df.columns:
            selected_countries = st.sidebar.multiselect("Select Country",
                                                       options=df['country'].dropna().unique())
            if selected_countries:
                df = df[df['country'].isin(selected_countries)]

        if 'tracking_id' in df.columns:
            selected_trackings = st.sidebar.multiselect("Select Trackings",
                                                       options=sorted(df['tracking_id'].dropna().unique()))
            if selected_trackings:
                df = df[df['tracking_id'].isin(selected_trackings)]

        if 'local_created_at' in df.columns:
            df['local_created_at'] = pd.to_datetime(df['local_created_at'])
            min_date, max_date = df['local_created_at'].min(), df['local_created_at'].max()
            date_range = st.sidebar.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            if isinstance(date_range, tuple) and len(date_range) == 2:
                df = df[(df['local_created_at'].dt.date >= date_range[0]) & (df['local_created_at'].dt.date <= date_range[1])]


        # --- Generate charts ---
        charts = generate_all_charts(df)
        tab_names = list(charts.keys())
        tabs = st.tabs(tab_names)

        for tab, name in zip(tabs, tab_names):
            with tab:
                fig = charts[name]
                st.subheader(name)
                if name in ["Calendar", "Competitor Calendar"]:
                    st.pyplot(fig)
                    # 3. Save figure to buffer
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png')
                    buf.seek(0)

                    # 4. Download button
                    st.download_button(
                        label="Download as PNG",
                        data=buf.getvalue(),
                        file_name=f"{name}.png",
                        mime="image/png"
                    )

                else:
                    st.plotly_chart(fig, use_container_width=True)
                    img_bytes = fig.to_image(format="png")
                    st.download_button(
                        label=f"Download '{name}' as PNG",
                        data=img_bytes,
                        file_name=f"{name}.png",
                        mime="image/png"
                    )

                # --- Auto AI observation for each chart ---
                # with st.spinner("Generating AI insight..."):
                #     sample = df.to_string()
                #     ai_prompt = f"Analyze this chart titled '{name}' using the following sample data:\n{sample}. Provide key insights and observations for a marketing analyst."
                #     insight = get_observation(ai_prompt)
                #     st.markdown("**🤖 AI Observation:**")
                #     st.markdown(insight)

        # --- Auto-summary and recommendations ---
        st.markdown("---")
        st.subheader("📊 Auto Summary and Recommendations")
        summary, recs = generate_summary_and_recommendations(df)

        with st.expander("Key Summary", expanded=True):
            for line in summary:
                st.markdown(f"- {line}")

        with st.expander("Actionable Recommendations", expanded=True):
            for rec in recs:
                st.markdown(f"✅ {rec}")

        st.success("All charts and insights generated.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("Please upload an Excel file to begin.")
