import plotly.express as px
import pandas as pd
from charts import competitor_volume, frequnecy_finder, cross_tabulation_by_competitor_lifecycle_channel, \
    heatmap_and_opportunity, \
    show_opportunity_table, show_calendar, domain_analysis, average_subject_length_table, content_types, \
    promotional_types, \
    tonality_types, communication_timeline, show_frequency_table, emoji_analysis, bonus_top_offers_stacked, bonus_top_offers_stacked_without_no_promo, \
    bonus_treemap_promotions, bundles_pairwise_heatmap, bundle_size_distribution, bundles_top_bar, bundles_treemap, \
    competitor_table, ip_analysis, plot_ips_vs_high_volume, number_of_email_first_48h, email_types_first_48h, \
    promoion_types_first_48h, show_calendar_by_competitor, build_esp_provider_heatmap, \
    build_esp_provider_table_from_providers, heatmap_by_day_hour_with_legend


def generate_all_charts(df):
    charts = {}

    # --- Chart 1: Competitor's Volume ---4
    print("Competitor's Table")
    charts['Competitor Table'] = competitor_table(df)

    print("Competitor's Volume")
    charts['Competitor Volume'] = competitor_volume(df)

    # --- Chart 2: Frequnecy Finder ---
    print("Frequnecy Finder")
    charts['Frequnecy Finder'], frequency_data = frequnecy_finder(df)

    # --- Chart 3: Frequnecy Table ---
    print("Frequnecy Table")
    charts['Frequnecy Table'] = show_frequency_table(frequency_data)

    # --- Chart 4: Cross-Tabulation ---
    print("Cross-Tabulation")
    charts["Cross-Tabulation"] = cross_tabulation_by_competitor_lifecycle_channel(df)

    # --- Chart 5: Heatmap and Opportunities ---
    print("Heatmap")
    charts["Heatmap"], heatmap_data = heatmap_and_opportunity(df)
    print("Heatmap 2")
    charts["New Heatmap"] = heatmap_by_day_hour_with_legend(df)

    # --- Chart 6: Opportunities ---
    print("Opportunities")
    charts["Opportunities"] = show_opportunity_table(heatmap_data)

    # --- Chart 7: Calendar ---
    print("Calendar")
    charts["Calendar"] = show_calendar(df)

    # --- Chart 7: Calendar ---
    print("Competitor Calendar")
    charts["Competitor Calendar"] = show_calendar_by_competitor(df)

    # --- Chart 7: Emails in the first 48h ---
    # print("Emails in the first 48h")
    # charts["Emails in the first 48h"] = number_of_email_first_48h(df)
#
    # # --- Chart 7: Emails in the first 48h ---
    # print("Email types in the first 48h")
    # charts["Email types in the first 48h"] = email_types_first_48h(df)
#
    # # --- Chart 7: Promotion types in the first 48h ---
    # print("Promotion types in the first 48h")
    # charts["Promotion types in the first 48h"] = promoion_types_first_48h(df)

    # --- Chart 8: Domain Analysis ---
    print("Domain Analysis")
    charts["Domain Analysis"] = domain_analysis(df)

    # === IP Analysis (new) ===
    print("IP Analysis")
    charts["IP Analysis"] = ip_analysis(df)

    # === IP Analysis (new) ===
    print("IP Analysis")
    charts["Total IPs vs High Volume IPs Analysis"], summary1 = plot_ips_vs_high_volume(df)

    print("ESP Providers Analysis")
    charts["ESP Providers Analysis"], df_providers = build_esp_provider_heatmap(df)

    print("ESP Providers Table")
    charts["ESP Providers Table"] = build_esp_provider_table_from_providers(df_providers)

    # --- Chart 9: Average Subject Length Analysis ---
    print("Average Subject Length Analysis")
    charts["Average Subject Length Analysis"] = average_subject_length_table(df)

    # --- Chart 9: Emoji Analysis ---
    print("Emoji Analysis")
    charts["Emoji Analysis"] = emoji_analysis(df)

    # --- Chart 10: Unique Content Types ---
    print("Unique Content Types")
    charts["Unique Content Types"] = content_types(df)

    # --- Chart 11: Promotional Types ---
    print("Promotional Types")
    charts["Promotional Types"] = promotional_types(df)

    # --- Chart 12: Promotional Types ---
    print("Tonality Types")
    charts["Tonality Types"] = tonality_types(df)

    # --- Chart 13: Communication Timeline ---
    print("Communication Timeline")
    charts["Communication Timeline"] = communication_timeline(df)

    # === Better Bonus Reports (new) ===
    print("Top Offers by Frequency Without No Promo")
    charts["Top Offers by Frequency Without No Promo"] = bonus_top_offers_stacked_without_no_promo(df)

    print("Top Offers by Frequency With No Promo")
    charts["Top Offers by Frequency"] = bonus_top_offers_stacked(df)

    # === Additional Bonus Reports (new) ===
    print("Offers by Frequency")
    charts["Offers by Frequency"] = bonus_treemap_promotions(df)

    # === Bundles Heatmap (new) ===
    print("Bundle Size Distribution")
    charts["Bundle Size Distribution"] = bundle_size_distribution(df)

    # === Bundles Heatmap (new) ===
    print("Bundles Heatmap")
    charts["Bundles Heatmap"] = bundles_pairwise_heatmap(df)

    # === Bundles Treemap (new) ===
    print("Bundles Treemap")
    charts["Bundles Treemap"] = bundles_treemap(df)

    # === Bundles Treemap (new) ===
    print("Bundles Treemap")
    charts["Bundles Top Bar"] = bundles_top_bar(df)


    # --- Chart 1: Promotion Types Bar Chart ---
    # if 'promotion_types' in df.columns:
    #     print('promotion_types')
    #     promo_counts = df['promotion_types'].value_counts().reset_index()
    #     promo_counts.columns = ['Promotion Type', 'Total Hits']
    #     fig1 = px.bar(promo_counts, x='Promotion Type', y='Total Hits', text='Total Hits')
    #     fig1.update_traces(textposition='outside')
    #     fig1.update_layout(title='Promotion Types by Frequency', xaxis_tickangle=-45)
    #     charts['Promotion Types'] = fig1

    # --- Chart 2: Hits Over Time ---
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['Date'] = df['created_at'].dt.date
        hits_by_day = df.groupby('Date').size().reset_index(name='Count')
        fig2 = px.line(hits_by_day, x='Date', y='Count', markers=True)
        fig2.update_layout(title='Hits Over Time')
        charts['Hits Over Time'] = fig2

    # --- Chart 3: Promotion Type vs Category Breakdown ---
    # if 'promotion_types' in df.columns and 'category' in df.columns:
    #     combo_counts = df.groupby(['promotion_types', 'category']).size().reset_index(name='Count')
    #     fig3 = px.bar(combo_counts, x='promotion_types', y='Count', color='category', barmode='stack')
    #     fig3.update_layout(title='Promotion Type by Category')
    #     charts['Promotion by Category'] = fig3

    # # --- Chart 4: Classification Pie Chart ---
    # if 'classification' in df.columns:
    #     class_counts = df['classification'].value_counts().reset_index()
    #     class_counts.columns = ['Classification', 'Count']
    #     fig4 = px.pie(class_counts, names='Classification', values='Count', hole=0.4)
    #     fig4.update_layout(title='Classification Distribution')
    #     charts['Classification'] = fig4
#
    # # --- Chart 5: Channel Type Pie Chart ---
    # if 'channel_type' in df.columns:
    #     channel_counts = df['channel_type'].value_counts().reset_index()
    #     channel_counts.columns = ['Channel Type', 'Count']
    #     fig5 = px.pie(channel_counts, names='Channel Type', values='Count', hole=0.3)
    #     fig5.update_layout(title='Channel Type Distribution')
    #     charts['Channels'] = fig5

    return charts
