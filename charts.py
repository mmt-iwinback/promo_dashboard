import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import calendar
from collections import Counter
import re
import emoji
from utilities.helper import _prepare_offer_cols, _agg_subjects, compute_bundles, preprocess_holiday_data, \
    competitor_holiday_promos_table
import plotly.colors as pc
import itertools
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.patches import Wedge
from matplotlib.patches import Rectangle
import requests
from collections import Counter
import colorsys
import random


def competitor_table(df):
    # Remove duplicate hits
    df = df.drop_duplicates(subset='tracking_hit_id')

    # Count communications per competitor
    competitor_counts = df[['competitor_name', 'country']].value_counts().reset_index()
    competitor_counts.columns = ['Competitor Name', 'Country', 'Number of Communications Received']

    # Add total row
    total_count = competitor_counts['Number of Communications Received'].sum()
    total_row = pd.DataFrame({
        'Competitor Name': ['TOTAL'],
        'Country': ['ALL'],
        'Number of Communications Received': [total_count]
    })
    competitor_counts = pd.concat([competitor_counts, total_row], ignore_index=True)

    # --- Build Figure ---
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(competitor_counts.columns),
                    fill_color="lightgray",
                    align="left",
                    font=dict(size=14, color="black"),
                    height=30
                ),
                cells=dict(
                    values=[competitor_counts[col] for col in competitor_counts.columns],
                    fill_color=[["white"] * (len(competitor_counts) - 1) + ["#f5f5f5"]],  # gray for total
                    align="left",
                    font=dict(size=12),
                    height=25
                )
            )
        ]
    )

    fig.update_layout(
        title="Competitor Hits",
        margin=dict(l=20, r=20, t=40, b=20),
        height=800
    )

    return fig


def competitor_volume(df):
    df = df.drop_duplicates(subset='tracking_hit_id')
    print('Columns')
    print(df.columns)
    competitor_counts = df['competitor_name'].value_counts().reset_index()
    competitor_counts.columns = ['competitor_name', 'count']
    fig = px.bar(competitor_counts, x='competitor_name', y='count', color='competitor_name',
                 title='Number of Hits per Competitor',
                 )
    fig.update_layout(
        xaxis_title='Competitor Name',
        yaxis_title='Count',
        plot_bgcolor='white',
        template='plotly',
        coloraxis=dict(colorscale="Viridis")
    )

    return fig


def find_velocity(df, days: int):
    """
    Chart data prep: per (competitor_name, tracking_id, lifecycle) volume/period and velocity over a rolling {days}-day window.
    """
    d = (
        df.drop_duplicates(subset="tracking_hit_id")
        .assign(
            local_created_at=lambda x: pd.to_datetime(x["local_created_at"], errors="coerce"),
            created_at=lambda x: pd.to_datetime(x["created_at"], errors="coerce"),
        )
        .dropna(subset=["local_created_at"])
    )

    grouped = (
        d.groupby(["competitor_name", "tracking_id", "lifecycle"], dropna=False)
        .agg(
            volume=("tracking_hit_id", "count"),
            min_date=("local_created_at", "min"),
            max_date=("local_created_at", "max"),
        )
        .reset_index()
    )

    grouped["min_date_str"] = grouped["min_date"].dt.strftime("%d-%b-%Y")
    grouped["max_date_str"] = grouped["max_date"].dt.strftime("%d-%b-%Y")
    grouped["active_period"] = (grouped["max_date"] - grouped["min_date"]).dt.days + 1
    grouped["weekly_velocity"] = (grouped["volume"] * 7 / max(days, 1)).round(2)
    #grouped["weekly_velocity"] = (grouped["volume"] * 7 / grouped["active_period"].replace(0, 1)).round(2)


    final_df = (
        grouped[[
            "competitor_name", "tracking_id", "volume",
            "min_date_str", "max_date_str", "active_period", "lifecycle", "weekly_velocity"
        ]]
        .sort_values(["competitor_name", "active_period"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return final_df


def process_comms_velocity(comms_velocity_data, period):
    """
    Post-process velocity table: rename dates, add 'period' label, sort by velocity.
    """
    df4 = pd.DataFrame(comms_velocity_data).copy()
    # standardize column names if present
    df4 = df4.rename(columns={"min_date_str": "min_date", "max_date_str": "max_date"})
    df4["weekly_velocity"] = df4["weekly_velocity"].fillna(0)

    # label period (fallback to blank if empty/None)
    label = "Since Creation -" if period == "all_time" else (str(period).capitalize() if period else "")
    df4["period"] = np.where(
        df4.get("active_period").notna(),
        (label + " " + df4["active_period"].astype(int).astype(str) + " days").str.strip(),
        label
    )

    df4 = df4.sort_values("weekly_velocity", ascending=False, ignore_index=True)
    return df4


def plot_communication_velocity(data, period, days, *_):
    """
    Chart: Pie of communication 'velocity' share per tracking_id (weekly_velocity as value).
    Data: expects columns ['competitor_name','tracking_id','weekly_velocity','volume','period','min_date','max_date','lifecycle'].
    Output: Plotly pie figure.
    """
    dfp = data.copy()

    # readable slice names like "ID — Competitor"
    dfp["slice_name"] = dfp["tracking_id"].astype(str) + " — " + dfp["competitor_name"].astype(str)

    # build pie (filter zero/NaN velocities to avoid empty slices)
    dfp = dfp[dfp["weekly_velocity"].fillna(0) > 0]
    fig = px.pie(
        dfp,
        values="weekly_velocity",
        names="slice_name",
        color_discrete_sequence=px.colors.qualitative.Vivid,
        opacity=0.9,
        custom_data=[
            dfp["competitor_name"],
            dfp["tracking_id"],
            dfp["volume"],
            dfp.get("period", ""),
            dfp["weekly_velocity"],
            dfp.get("min_date", ""),
            dfp.get("max_date", ""),
            dfp.get("lifecycle", ""),
        ],
        labels={"weekly_velocity": "Velocity"},
        title=None,
    )

    fig.update_traces(
        textposition="inside",
        textinfo="value",
        marker=dict(line=dict(color="#FFFFFF", width=1)),
        hovertemplate=(
            "Competitor: %{customdata[0][0]}<br>"
            "ID: %{customdata[0][1]}<br>"
            "Volume: %{customdata[0][2]}<br>"
            "Period: %{customdata[0][3]}<br>"
            "From: %{customdata[0][5]}<br>"
            "To: %{customdata[0][6]}<br>"
            "Lifecycle: %{customdata[0][7]}<br>"
            "Velocity: %{customdata[0][4]}<extra></extra>"
        ),
        direction="clockwise",
    )

    title_text = (
        "Warning: Incomparable results for Communication Velocity Reports. "
        "The number of days considered varies by account."
        if period == "all_time" else
        "Communication Velocity by Active Days"
    )

    fig.update_layout(
        title=dict(text=title_text, x=0.5, font=dict(size=13, color="palevioletred")),
        legend_traceorder="normal",
        legend_title_text="Competitors",
        template="plotly_white",
        margin=dict(t=60, l=40, r=40, b=40),
    )
    return fig


def frequnecy_finder(df, days: int = 30, period: str = "first"):
    """
    Runner: compute velocities, format table, and render pie; returns (fig, table_df).
    """
    table_df = find_velocity(df, days)
    formatted = process_comms_velocity(table_df, period)
    fig = plot_communication_velocity(formatted, period, days, None)
    return fig, table_df


def show_frequency_table(frequency_data):
    frequency_df = frequency_data.copy()
    frequency_df = frequency_df.sort_values(by=['active_period', 'weekly_velocity'], ascending=[False, False],
                                            ignore_index=True)

    # Reset index so 'day_of_week' is a column
    df_table = frequency_df.reset_index()
    df_table.columns = ['Index', 'Competitor Name', 'Tracking ID', 'Volume', 'Analysis Start',
                        'Analysis End',
                        'Active Period', 'Lifecycle', 'Weekly Velocity']

    # Create Plotly Table
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(df_table.columns),
                    fill_color='lightgray',
                    align='left',
                    font=dict(size=15, color='black'),
                ),
                cells=dict(
                    values=[df_table[col] for col in df_table.columns],
                    align='left',
                    fill_color=[["white"] * (len(df_table['Competitor Name'].unique()))],
                    font=dict(size=12),
                    height=35,
                )
            )
        ]
    )

    fig.update_layout(
        title='Frequency Analysis Table',
        margin=dict(l=20, r=20, t=40, b=20),
        height=1000
    )

    return fig


def cross_tabulation_by_competitor_lifecycle_channel(df):
    # Create the crosstab
    df = df.drop_duplicates(subset='tracking_hit_id')
    ct = pd.crosstab(
        index=df['competitor_name'],
        columns=[df['lifecycle'], df['channel'], df['vertical']],
        values=df['tracking_hit_id'],
        aggfunc='count'
    ).fillna(0).astype(int)

    # Step 2: Flatten the MultiIndex columns
    ct.columns = [f"{vertical} | {lifecycle} | {channel}" for lifecycle, channel, vertical in ct.columns]

    # Step 3: Reset index to include competitor_name as a column
    ct_reset = ct.reset_index()

    # Step 4: Build Plotly Table
    header = ['Competitor Name'] + list(ct.columns)
    print(header)
    values = [ct_reset[col] for col in ct_reset.columns]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=header,
                    fill_color='lightgray',
                    align='left',
                    font=dict(size=15, color='black')
                ),
                cells=dict(
                    values=values,
                    align='left',
                    fill_color='white',
                    font=dict(size=13),
                    height=24
                )
            )
        ]
    )

    fig.update_layout(
        title='Tracking Hits Table by Competitor, Lifecycle, and Channel',
        #height=34 * len(df['competitor_name'].unique())
    )

    return fig


def heatmap_and_opportunity(df):
    df = df.drop_duplicates(subset='tracking_hit_id')
    df['local_created_at'] = pd.to_datetime(df['local_created_at'])

    # Extract hour and weekday
    df['hour'] = df['local_created_at'].dt.hour
    df['day_of_week'] = df['local_created_at'].dt.day_name()

    # Define order
    weekday_order = ['Sunday', 'Saturday', 'Friday', 'Thursday', 'Wednesday', 'Tuesday', 'Monday']
    hour_order = list(range(24))

    # Create pivot table with 0s for missing values
    heatmap_data = (
        df.groupby(['day_of_week', 'hour'])
        .size()
        .unstack(fill_value=0)
        .reindex(index=weekday_order, columns=hour_order, fill_value=0)
    )

    # Create imshow heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,  # 0–23 hours
        y=heatmap_data.index,  # Monday–Sunday
        colorscale='RdYlGn_r',
        zmin=0,
        showscale=True,
        text=heatmap_data.values,
        texttemplate="%{text}",
        hovertemplate="Day: %{y}<br>Hour: %{x}<br>Communications: %{z}<extra></extra>"
    ))

    # Update layout
    fig.update_layout(
        title='Actual Hourly Volume by Day of Week',
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        xaxis=dict(dtick=1),
        height=600,
        margin=dict(t=60, l=60, r=40, b=60)
    )

    return fig, heatmap_data


def find_opportunities(heatmap_data):
    df_2dhist = heatmap_data.copy()
    opportunity_df = pd.DataFrame(index=df_2dhist.index, columns=['opp1', 'opp2'])
    for day in df_2dhist.index:
        day_hits = df_2dhist.loc[day]
        # transform NaN to 0
        day_hits = np.nan_to_num(day_hits)

        max_sum = 0
        position_max = 0

        for i in range(len(day_hits) - 6):
            if sum(day_hits[i:i + 3]) >= max_sum:
                max_sum = sum(day_hits[i:i + 3])
                position_max = i
                opportunity_1 = position_max - 1
                opportunity_2 = position_max + 3
            else:
                continue

        opportunity_df.loc[day, :] = [opportunity_1, opportunity_2]

    return opportunity_df


def show_opportunity_table(heatmap_data):
    opportunity_df = find_opportunities(heatmap_data)

    # Step 1: Set index to ordered categorical for proper weekday order
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    opportunity_df.index = pd.Categorical(opportunity_df.index, categories=day_order, ordered=True)
    opportunity_df = opportunity_df.sort_index()

    # Step 2: Reset index and set column names
    df_table = opportunity_df.reset_index()
    df_table.columns = ['Day of Week', 'Opportunity 1 (Before Congestion Hours)',
                        'Opportunity 2 (After Congestion Hours)']

    # Create Plotly Table
    fig = go.Figure(
        data=[
            go.Table(
                # columnwidth=[120, 400, 200],  # set width per column (px or relative units)
                header=dict(
                    values=list(df_table.columns),
                    fill_color='lightgray',
                    align='left',
                    font=dict(size=30, color='black')
                ),
                cells=dict(
                    values=[df_table[col] for col in df_table.columns],
                    align='left',
                    font=dict(size=20),
                    height=30
                )
            )
        ]
    )

    fig.update_layout(
        title='Identified Email Sending Opportunity Hours by Day',
        height=500
    )

    return fig


def plot_calendar_heatmap(data, year):
    data = data[data['date'].dt.year == year]

    # 🗓 Get only the unique months present in the data
    available_months = sorted(data['month'].unique())

    # 📐 Dynamically adjust subplot layout
    n_months = len(available_months)
    ncols = 4
    nrows = int(np.ceil(n_months / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten()

    for plot_idx, month_num in enumerate(available_months):
        ax = axes[plot_idx]
        month_data = data[data['month'] == month_num]

        # 📅 Create a matrix of days
        days_in_month = calendar.monthrange(year, month_num)[1]
        first_weekday = calendar.monthrange(year, month_num)[0]
        calendar_matrix = np.full((6, 7), np.nan)  # Max 6 weeks, 7 days

        for day in range(1, days_in_month + 1):
            week, weekday = divmod(first_weekday + day - 1, 7)
            value = month_data.loc[month_data['day'] == day, 'values'].sum()
            calendar_matrix[week, weekday] = value

        cax = ax.matshow(calendar_matrix, cmap='RdYlGn_r', aspect='auto', alpha=0.7)
        ax.set_title(f"{calendar.month_name[month_num]}")
        ax.set_xticks(range(7))
        ax.set_yticks(range(6))
        ax.set_xticklabels(list(calendar.day_abbr), rotation=45)
        ax.set_yticklabels(range(1, 7))

        # ➕ Add values in each cell
        for (i, j), val in np.ndenumerate(calendar_matrix):
            if not np.isnan(val):
                ax.text(j, i, f"{int(val)}", ha='center', va='center', fontsize=8)

    # ❌ Hide unused axes (if any)
    for k in range(n_months, len(axes)):
        fig.delaxes(axes[k])

    plt.tight_layout()
    fig.colorbar(cax, ax=axes[:n_months], orientation='vertical', fraction=0.02, pad=0.02)
    return fig


def show_calendar(df, year=2025):
    temp_df = df.copy()
    temp_df = temp_df.drop_duplicates(subset='tracking_hit_id')
    temp_df['created_at'] = pd.to_datetime(temp_df['created_at'])
    temp_df['local_created_at'] = pd.to_datetime(temp_df['local_created_at'])

    temp_df['date'] = pd.to_datetime(temp_df['local_created_at'].dt.date)
    temp_df['day'] = temp_df['local_created_at'].dt.day
    temp_df['month'] = temp_df['local_created_at'].dt.month

    temp_df = temp_df.groupby(['date', 'day', 'month']).agg(
        values=('tracking_hit_id', 'count'),
    ).reset_index()

    fig = plot_calendar_heatmap(temp_df, year)
    return fig


def draw_cell_piering(ax, j, i, counts_by_comp, comp_to_idx, cmap):
    total = sum(h for _, h in counts_by_comp)
    if total == 0: return
    # center of cell (image coords)
    cx, cy = j, i
    r_outer, r_inner = 0.48, 0.32  # thin ring
    angle = 90.0  # start at 12 o'clock
    for comp, hits in sorted(counts_by_comp, key=lambda x: x[0]):
        frac = hits / total
        sweep = 360 * frac
        color = cmap(comp_to_idx[comp] / len(comp_to_idx))
        ax.add_patch(Wedge((cx, cy), r_outer, angle, angle + sweep, width=r_outer - r_inner,
                           facecolor=color, edgecolor='none', alpha=0.95))
        angle += sweep


# ---------- tiny in-cell bar glyph ----------
def draw_cell_barglyph(ax, cell_col, cell_row, counts_by_comp, comp_to_idx, cmap):
    """
    Draw a small multi-competitor bar glyph inside one calendar cell.

    cell_col: int (0..6) weekday column
    cell_row: int (0..5) week row
    counts_by_comp: list[(competitor, hits)]
    """
    total = sum(h for _, h in counts_by_comp)
    if total <= 0:
        return

    # layout: a thin strip near bottom of the cell; height encodes share
    y_center = cell_row + 0.30
    strip_height = 0.18
    items = sorted(counts_by_comp, key=lambda x: x[0])  # stable order by competitor name

    n = len(items)
    pad = 0.02
    bar_width = (1 - (n + 1) * pad) / max(n, 1)

    for k, (comp, hits) in enumerate(items):
        frac = hits / total
        bar_h = strip_height * max(0.1, frac)  # keep a visible minimum
        x_left = cell_col - 0.5 + pad + k * (bar_width + pad)

        idx = comp_to_idx.get(comp)
        if idx is None:
            continue

        color = cmap(idx / len(comp_to_idx))
        ax.add_patch(
            Rectangle(
                (x_left, y_center - bar_h / 2),
                bar_width,
                bar_h,
                facecolor=color,
                edgecolor='none',
                alpha=0.95,
            )
        )


def plot_calendar_bars_by_competitor(data, year, annotate="total"):
    """
    Bars-only calendar:
      - No background fill
      - Each day cell shows tiny bars for all competitors that sent that day
    annotate:
      - "total": show total hits per day (center)
      - None: no text annotation in center
    """
    data = data[data['date'].dt.year == year].copy()

    # aggregate: (date, month, day, competitor) -> hits
    by_day_comp = (
        data.groupby(['date', 'month', 'day', 'competitor'])
        .size()
        .reset_index(name='hits')
    )
    totals = by_day_comp.groupby('date')['hits'].sum()

    # months & layout
    available_months = sorted(by_day_comp['month'].unique())
    if len(available_months) == 0:
        available_months = sorted(data['month'].unique())

    n_months = len(available_months)
    ncols = 3  # fewer columns → bigger cells
    nrows = int(np.ceil(n_months / ncols))

    # 🔹 enlarge figure
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.5 * ncols, 4 * nrows)  # bigger: more width & height per subplot
    )
    axes = np.atleast_1d(axes).flatten()

    # colors per competitor (index start at 1; 0 is transparent)
    competitors = sorted(by_day_comp['competitor'].unique())
    comp_to_idx = {c: i + 1 for i, c in enumerate(competitors)}

    base_colors = list(plt.cm.tab20.colors)  # 20 colors
    if len(competitors) > len(base_colors):
        repeats = int(np.ceil(len(competitors) / len(base_colors)))
        palette = (base_colors * repeats)[:len(competitors)]
    else:
        palette = base_colors[:len(competitors)]
    cmap = ListedColormap([(1, 1, 1, 0)] + list(palette))  # index 0 transparent

    for plot_idx, month_num in enumerate(available_months):
        ax = axes[plot_idx]

        days_in_month = calendar.monthrange(year, month_num)[1]
        first_weekday = calendar.monthrange(year, month_num)[0]  # Mon=0

        # transparent background image just to keep the grid coordinate system
        idx_matrix = np.zeros((6, 7))  # all zeros -> transparent with our cmap
        ax.matshow(idx_matrix, cmap=cmap, aspect='auto', alpha=0.0)  # fully invisible

        # tick labels and titles
        ax.set_title(f"{calendar.month_name[month_num]}")
        ax.set_xticks(range(7))
        ax.set_yticks(range(6))
        ax.set_xticklabels(list(calendar.day_abbr), rotation=45, fontsize=8)
        ax.set_yticklabels(range(1, 7), fontsize=8)

        # thin gridlines to keep boxes visible (optional)
        ax.set_xlim(-0.5, 6.5)
        ax.set_ylim(5.5, -0.5)
        ax.set_xticks(np.arange(-0.5, 7, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
        ax.grid(which='minor', linestyle='-', linewidth=0.3, alpha=0.4)

        # subset once per month for speed
        by_day_comp_m = by_day_comp[by_day_comp['month'] == month_num]

        for day in range(1, days_in_month + 1):
            week, weekday = divmod(first_weekday + day - 1, 7)
            i, j = week, weekday

            d = pd.Timestamp(year=year, month=month_num, day=day)

            # draw day number in corner
            ax.text(j - 0.42, i - 0.32, str(day), ha='left', va='top', fontsize=7, alpha=0.85)

            # all competitors that sent on d
            g = by_day_comp_m[by_day_comp_m['date'] == d][['competitor', 'hits']]
            if not g.empty:
                counts_by_comp = list(g.itertuples(index=False, name=None))  # [(competitor, hits), ...]
                # draw_cell_barglyph(ax, j, i, counts_by_comp, comp_to_idx, cmap)
                # overlay small ring wedges for all competitors who sent
                draw_cell_piering(ax, j=weekday, i=week,
                                  counts_by_comp=list(g.itertuples(index=False, name=None)),
                                  comp_to_idx=comp_to_idx, cmap=cmap)

                # center annotation
                if annotate == "total" and d in totals.index:
                    ax.text(j, i, str(int(totals.loc[d])), ha='center', va='center', fontsize=8, alpha=0.9)

    # hide unused axes
    for k in range(n_months, len(axes)):
        fig.delaxes(axes[k])

    # legend (competitor -> color)
    if competitors:
        legend_handles = [
            Patch(facecolor=palette[i], edgecolor='none', label=c) for i, c in enumerate(competitors)
        ]
        fig.legend(
            handles=legend_handles,
            loc='lower center',
            ncol=min(2, len(legend_handles)),
            bbox_to_anchor=(0.5, -0.001)
        )

    plt.tight_layout()
    return fig


def show_calendar_by_competitor(df, year=2025, competitor_col='competitor_name', annotate="total"):
    temp_df = df.copy()
    temp_df = temp_df.drop_duplicates(subset='tracking_hit_id')
    temp_df['created_at'] = pd.to_datetime(temp_df['created_at'])
    temp_df['local_created_at'] = pd.to_datetime(temp_df['local_created_at'])

    temp_df['date'] = pd.to_datetime(temp_df['local_created_at'].dt.date)
    temp_df['day'] = temp_df['local_created_at'].dt.day
    temp_df['month'] = temp_df['local_created_at'].dt.month
    temp_df = temp_df.rename(columns={competitor_col: 'competitor'})

    return plot_calendar_bars_by_competitor(
        temp_df[['date', 'day', 'month', 'competitor', ]],
        year=year,
        annotate=annotate
    )


def domain_analysis(df):
    df = df.drop_duplicates(subset='tracking_hit_id')
    df['tracking_start_at'] = pd.to_datetime(df['tracking_start_at'])
    df['created_at'] = pd.to_datetime(df['created_at'])
    # Clean the 'from' field for consistent domain extraction
    df['from'] = df['from'].str.strip().str.lower().str.extract(r'<?([\w\.-]+@[\w\.-]+)>?')[0]

    # Extract domain from 'from' field
    df['from_domain'] = df['from'].str.extract(r'@(.+)$')

    # Count number of hits per tracking_id and from_domain to analyze warming behavior
    df_domain_summary = (
        df.groupby(['competitor_name', 'tracking_id', 'from_domain'])
        .agg(
            number_of_hits=('tracking_hit_id', 'count'),
            first_seen=('local_created_at', 'min'),
            last_seen=('local_created_at', 'max'),
        )
        .reset_index()
    )

    df_domain_summary['first_seen'] = df_domain_summary['first_seen'].dt.strftime('%d-%b-%Y')
    df_domain_summary['last_seen'] = df_domain_summary['last_seen'].dt.strftime('%d-%b-%Y')

    # Sum number_of_hits per from_domain and competitor_name
    heatmap_data = df_domain_summary.groupby(['from_domain', 'competitor_name'])['number_of_hits'].sum().reset_index()

    # Now create the pivot table
    pivot_table = heatmap_data.pivot(index='from_domain', columns='competitor_name', values='number_of_hits').fillna(0)

    # Make sure pivot_table has correct structure
    z = pivot_table.values
    x = pivot_table.columns.tolist()  # Competitor names (columns)
    y = pivot_table.index.tolist()  # From domains (rows)

    max_val = max(map(max, z))

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        # colorscale='YlGnBu',
        hovertemplate='Competitor: %{x}<br>Domain: %{y}<br>Hits: %{z}<extra></extra>',
        text=z,
        texttemplate="%{text}",
        # ---- CUSTOM COLORSCALE ----
        zmin=0,
        zmax=max_val,

        # ---- SINGLE HEATMAP COLORSCALE ----
        colorscale=[
            [0.0, "#f0f0f0"],  # exactly 0
            [0.00001, "#f0f0f0"],  # still 0 (tiny offset)
            [0.000011, "#d9f0a3"],  # start of your palette
            [1.0, "#084081"],  # end of palette
        ],
        showscale=True  # hide colorbar (optional)
    ))

    # Customize layout
    fig.update_layout(
        title='Number of Hits per Domain and Competitor',
        xaxis_title='Competitor Name',
        yaxis_title='From Domain',
        height=600,
        margin=dict(t=60, l=100, r=40, b=60)
    )

    return fig


def ip_analysis(df):
    df = df.drop_duplicates(subset='tracking_hit_id')
    # Check for IP column and proceed if available
    if 'from_ip' in df.columns:
        # Clean datetime fields if needed
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

        # Group by competitor and IP address
        ip_summary = (
            df.groupby(['competitor_name', 'from_ip'])
            .agg(
                number_of_hits=('tracking_hit_id', 'count'),
                first_seen=('created_at', 'min'),
                last_seen=('created_at', 'max'),
                unique_to_addresses=('to', pd.Series.nunique)
            )
            .reset_index()
        )

        # Separate analyses by competitor
        ip_grouped = ip_summary.groupby('competitor_name')
        ip_dfs = {name: group.reset_index(drop=True) for name, group in ip_grouped}

        # Display each competitor's IP strategy

        for competitor, table in ip_dfs.items():
            print(f"{competitor} - IP Strategy")
            print(table)
    else:
        ip_summary = None  # No IP analysis possible without the column

    # Create a pivot table suitable for heatmap visualization
    pivot_table = ip_summary.pivot(index='from_ip', columns='competitor_name', values='number_of_hits').fillna(0)

    # Make sure pivot_table has correct structure
    z = pivot_table.values
    x = pivot_table.columns.tolist()  # Competitor names (columns)
    y = pivot_table.index.tolist()  # From domains (rows)

    max_val = max(map(max, z))

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        # colorscale='YlGnBu',
        hovertemplate='Competitor: %{x}<br>IP Address: %{y}<br>Hits: %{z}<extra></extra>',
        text=z,
        texttemplate="%{text}",
        # ---- CUSTOM COLORSCALE ----
        zmin=0,
        zmax=max_val,

        # ---- SINGLE HEATMAP COLORSCALE ----
        colorscale=[
            [0.0, "#f0f0f0"],  # exactly 0
            [0.00001, "#f0f0f0"],  # still 0 (tiny offset)
            [0.000011, "#d9f0a3"],  # start of your palette
            [1.0, "#084081"],  # end of palette
        ],
        showscale=True  # hide colorbar (optional)
    ))

    # Customize layout
    fig.update_layout(
        title='Number of Hits per IP Address and Competitor',
        xaxis_title='Competitor Name',
        yaxis_title='From IP Address',
        height=600,
        margin=dict(t=60, l=100, r=40, b=60)
    )

    return fig


def plot_ips_vs_high_volume(
        df: pd.DataFrame,
        *,
        volume_threshold: int = 20,  # hits per IP to count as "high volume"
        min_total_ips: int = 1,  # filter competitors with very few IPs
        title: str = "Total IPs vs High-Volume IPs by Competitor"
):
    df = df.drop_duplicates(subset='tracking_hit_id')

    # 1) Aggregate hits per (competitor, IP)
    per_ip = (
        df.groupby(["competitor_name", "from_ip"], dropna=False)["tracking_hit_id"]
        .count()
        .rename("hits")
        .reset_index()
    )

    # 2) For each competitor: total distinct IPs & high-volume IPs
    agg = (
        per_ip.assign(is_high=(per_ip["hits"] >= volume_threshold).astype(int))
        .groupby("competitor_name", as_index=False)
        .agg(total_ips=("from_ip", "nunique"),
             high_volume_ips=("is_high", "sum"))
    )

    # Calculate remainder
    agg["other_ips"] = agg["total_ips"] - agg["high_volume_ips"]

    # optional filter
    agg = agg[agg["total_ips"] >= min_total_ips].copy()
    agg = agg.sort_values("total_ips", ascending=False)

    # 3) Build Plotly stacked bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=agg["competitor_name"],
        y=agg["other_ips"],
        name="Other IPs",
        marker_color="lightsteelblue",
        hovertemplate="Competitor: %{x}<br>Other IPs: %{y}<extra></extra>"
    ))

    fig.add_trace(go.Bar(
        x=agg["competitor_name"],
        y=agg["high_volume_ips"],
        name="High Volume IPs",
        marker_color="midnightblue",
        hovertemplate="Competitor: %{x}<br>High Volume IPs: %{y}<extra></extra>"
    ))

    # 4) Layout
    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis_title="Competitor",
        yaxis_title="Number of IPs",
        xaxis_tickangle=-35,
        legend=dict(title="Legend", orientation="h", y=-0.2, x=0.5, xanchor="center"),
        height=500,
        margin=dict(t=60, l=60, r=40, b=100)
    )

    return fig, agg


def average_subject_length_table(df):
    df = df.drop_duplicates(subset='tracking_hit_id')

    # Filter out missing subjects
    df = df[df['subject'].notna()].copy()
    df['subject'] = df['subject'].astype(str).str.strip()

    # Compute subject length
    df['subject_length'] = df['subject'].str.len()

    # Group by competitor
    summary = df.groupby('competitor_name')['subject_length'].mean().reset_index()
    summary['subject_length'] = summary['subject_length'].round(2)

    summary.columns = ['Competitor Name', 'Average Subject Length']

    # Sort by descending subject length
    summary = summary.sort_values('Average Subject Length', ascending=False).reset_index(drop=True)

    # Build full-row color scale
    lengths = summary['Average Subject Length'].values
    row_colors = []

    for x in lengths:
        if x < 24:
            c = '#ffcccc'  # red
        elif 24 <= x < 30:
            c = '#fff3b0'  # yellow
        elif 30 <= x <= 36:
            c = '#c7f7c1'  # green
        elif 36 < x <= 42:
            c = '#fff3b0'  # yellow again
        else:  # x > 42
            c = '#ffcccc'  # red

        row_colors.append(c)

    # For full-row coloring, repeat each color for all columns
    fill_colors = [
        row_colors,  # Competitor Name column
        row_colors  # Average Subject Length column
    ]

    # Build table figure
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=summary.columns,
                    fill_color='lightgray',
                    align='left',
                    font=dict(size=20)
                ),
                cells=dict(
                    values=[summary[col] for col in summary.columns],
                    fill_color=fill_colors,
                    align='left',
                    font=dict(size=15),
                    height=35
                )
            )
        ]
    )

    fig.update_layout(
        title='Average Subject Line Length per Competitor',
        margin=dict(t=60, l=100, r=100, b=40),
        height=800,
        template="plotly_white"
    )

    return fig


def content_types(df):
    # Step 1: Prepare normalized pivot table
    df_unique = df.drop_duplicates(subset='tracking_hit_id')

    heatmap_data = (
        df_unique.pivot_table(
            index='competitor_name',
            columns='email_type',
            values='tracking_hit_id',
            aggfunc='count',
            fill_value=0
        )
    )

    # Normalize to percentages
    heatmap_data = heatmap_data.div(heatmap_data.sum(axis=1), axis=0) * 100
    heatmap_data = heatmap_data.round(1)

    # Step 2: Prepare data for imshow
    z = heatmap_data.values
    x = heatmap_data.columns.tolist()  # Email types
    y = heatmap_data.index.tolist()  # Competitor names

    # Step 3: Create Plotly heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale='Purples',
        text=z,
        texttemplate="%{text}%",  # Show percentages
        hovertemplate="Competitor: %{y}<br>Email Type: %{x}<br>Share: %{z:.1f}%<extra></extra>",
        showscale=True
    ))

    fig.update_layout(
        title="Unique Email Types per Competitor (in %)",
        xaxis_title="Email Type",
        yaxis_title="Competitor",
        height=600,
        margin=dict(t=60, l=100, r=40, b=60)
    )

    return fig


def promotional_types(df):
    # Filter and aggregate data
    temp_data = df.copy()
    temp_data['type'] = temp_data['type'].fillna("No Promo")
    temp_data["type"] = (temp_data["type"].str.replace(r"\bfree chip\b", "Free Chips", case=False, regex=True))
    temp_data["type"] = (
        temp_data["type"].str.replace(r"\bTournament Entry\b", "Tournament", case=False, regex=True))

    temp_data = temp_data[
        temp_data['email_type'].isin(['Welcome', 'Informational', 'System', 'Transactional', 'Promotional'])]

    def fix_type(row):
        message_type = row['email_type']
        promo_type = row['type']

        # Rule 1: Promotional message but type says No Promo → Branding
        if message_type == "Promotional" and promo_type == "No Promo":
            return "Branding"
        if message_type in ['Welcome', 'Informational', 'System', 'Transactional'] and promo_type == "No Promo":
            return "System Message"

        # Rule 2: For all other message types → keep original
        return promo_type

    temp_data['type'] = temp_data.apply(fix_type, axis=1)

    # Pivot: Competitor vs. Promotion Type, with counts
    promo_heatmap_data = (
        temp_data.pivot_table(
            index='competitor_name',
            columns='type',
            values='tracking_hit_id',
            aggfunc='count',
            fill_value=0
        )
    )

    # Normalize to row-wise % (per competitor)
    promo_heatmap_data = promo_heatmap_data.div(promo_heatmap_data.sum(axis=1), axis=0) * 100
    promo_heatmap_data = promo_heatmap_data.round(1)

    # --- NEW STEP: Order columns by total population (descending) ---
    column_popularity = promo_heatmap_data.sum(axis=0).sort_values(ascending=False)
    promo_heatmap_data = promo_heatmap_data[column_popularity.index]

    # Prepare data for Plotly
    z = promo_heatmap_data.values
    x = promo_heatmap_data.columns.tolist()  # Promotion Types
    y = promo_heatmap_data.index.tolist()  # Competitor Names

    # Plotly Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale='Blues',
        text=z,
        texttemplate="%{text}%",
        hovertemplate="Competitor: %{y}<br>Promo Type: %{x}<br>Share: %{z:.1f}%<extra></extra>",
        showscale=True
    ))

    fig.update_layout(
        title="Promotion Types per Competitor (in %)",
        xaxis_title="Promotion Type",
        yaxis_title="Competitor",
        height=600,
        xaxis_tickangle=45,
        yaxis=dict(automargin=True),
        margin=dict(t=60, l=100, r=40, b=60)
    )

    return fig


def tonality_types(df):
    # Step 1: Clean and filter the data
    df_unique = df.drop_duplicates(subset='tracking_hit_id')

    # Step 2: Create pivot table
    tone_heatmap_data = (
        df_unique.pivot_table(
            index='competitor_name',
            columns='tone_of_voice',
            values='tracking_hit_id',
            aggfunc='count',
            fill_value=0
        )
    )

    # Step 3: Normalize to percentages
    tone_heatmap_data = tone_heatmap_data.div(tone_heatmap_data.sum(axis=1), axis=0) * 100
    tone_heatmap_data = tone_heatmap_data.round(1)

    # --- NEW STEP: Order columns by total population (descending) ---
    column_popularity = tone_heatmap_data.sum(axis=0).sort_values(ascending=False)
    tone_heatmap_data = tone_heatmap_data[column_popularity.index]

    # Step 4: Prepare data for Plotly
    z = tone_heatmap_data.values
    x = tone_heatmap_data.columns.tolist()  # Tone of Voice types
    y = tone_heatmap_data.index.tolist()  # Competitors

    # Step 5: Create Plotly heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale='Blues',
        text=z,
        texttemplate="%{text}%",
        hovertemplate="Competitor: %{y}<br>Tone: %{x}<br>Share: %{z:.1f}%<extra></extra>",
        showscale=True
    ))

    fig.update_layout(
        title="Tone of Voice per Competitor (in %)",
        xaxis_title="Tone of Voice",
        yaxis_title="Competitor",
        height=500,
        xaxis_tickangle=0,
        margin=dict(t=60, l=100, r=40, b=60)
    )
    return fig


# Function to generate N distinct colors (HSL evenly spaced)
def generate_colors(n):
    colors = []
    golden_ratio = 0.618033988749895
    hue = 0.0

    for i in range(n):
        hue = (hue + golden_ratio) % 1.0
        lightness = 0.5
        saturation = 0.7
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append('#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255)))

    return colors


def communication_timeline(df):
    # Filter & structure journey data
    temp_df = df.copy()
    # temp_df = temp_df.drop_duplicates(subset='tracking_hit_id')

    # Build journey_df including tracking_created_at
    journey_df = temp_df[
        ['competitor_name', 'created_at', 'subject', 'promotion_type', 'email_type', 'type', 'tracking_start_at']
    ].copy()
    print(journey_df['email_type'].unique())

    # Convert to datetime
    journey_df['created_at'] = pd.to_datetime(journey_df['created_at'])

    # Day offset per competitor
    journey_df['day_offset'] = (
            journey_df['created_at'].dt.normalize() -
            journey_df.groupby('competitor_name')['created_at'].transform(lambda x: x.min().normalize())
    ).dt.days

    # No negative values (just in case)
    journey_df['day_offset'] = journey_df['day_offset']

    # Optional: sort
    journey_df = journey_df.sort_values(by=['competitor_name', 'day_offset'])

    # "Day X" labels
    journey_df['day_index'] = "Day " + journey_df['day_offset'].astype(str)

    # Optional: fill missing type labels
    journey_df['type'] = journey_df['type'].fillna("No Promo")

    # journey_df = journey_df[journey_df['email_type'].isin(['Welcome', 'Informational', 'System', 'Transactional', 'Promotional'])]

    def fix_type(row):
        message_type = row['email_type']
        promo_type = row['type']

        # Rule 1: Promotional message but type says No Promo → Branding
        if message_type == "Promotional" and promo_type == "No Promo":
            return "Branding"
        if message_type in ['Welcome', 'Informational', 'System', 'Transactional'] and promo_type == "No Promo":
            return "System Message"

        # Rule 2: For all other message types → keep original
        return promo_type

    journey_df['type'] = journey_df.apply(fix_type, axis=1)
    print(journey_df['type'].unique())

    # Get unique competitor names
    promo_types = journey_df["type"].unique()
    num_types = len(promo_types)

    palette = pc.qualitative.Light24
    color_map = {comp: palette[i % len(palette)] for i, comp in enumerate(promo_types)}

    # Create timeline scatter plot
    fig = px.scatter(
        journey_df,
        x="day_offset",
        y="competitor_name",
        color="type",
        hover_data=["subject"],
        title="Email Communication Timeline by Competitor",
        color_discrete_map=color_map
    )

    # Style
    fig.update_traces(marker=dict(size=7, opacity=0.7))
    fig.update_layout(
        height=800,
        width=1500,
        xaxis_title="Relative Day of Sending",
        yaxis_title="Competitor",
        legend_title="Promotion Type",
        margin=dict(t=60, l=60, r=40, b=60)
    )

    # Optional: Add timeline marker
    # fig.add_vline(x=pd.to_datetime("2024-05-27"), line_width=2, line_dash="dash", line_color="green")

    # Show & export
    return fig


def contains_emoji(text):
    return any(emoji.is_emoji(char) for char in text) if isinstance(text, str) else False


def extract_emojis(text):
    return [char for char in text if emoji.is_emoji(char)] if isinstance(text, str) else []


def emoji_analysis(df):
    df = df.drop_duplicates(subset='tracking_hit_id')

    subject_col = 'subject'
    content_col = 'content'
    competitor_col = 'competitor_name'
    """
    Analyze emoji usage in email subjects and content by competitor and return a Plotly table figure.

    Parameters:
        df (pd.DataFrame): DataFrame containing at least subject, content, and competitor columns.
        subject_col (str): Name of the column with email subject lines.
        content_col (str): Name of the column with email content.
        competitor_col (str): Name of the column with competitor identifiers.

    Returns:
        fig (plotly.graph_objs.Figure): Plotly table visualizing emoji usage per competitor.
    """

    def contains_emoji(text):
        return any(emoji.is_emoji(char) for char in text)

    results = []

    for competitor_name in df[competitor_col].dropna().unique():
        comp_data = df[df[competitor_col] == competitor_name]
        subj_emails = comp_data[subject_col].dropna()
        cont_emails = comp_data[content_col].dropna()

        emoji_subj = sum(subj_emails.apply(contains_emoji))
        emoji_cont = sum(cont_emails.apply(contains_emoji))

        total_subj = len(subj_emails)
        total_cont = len(cont_emails)

        percent_subj = round((emoji_subj / total_subj) * 100, 2) if total_subj else 0
        percent_cont = round((emoji_cont / total_cont) * 100, 2) if total_cont else 0

        all_emojis = []
        for text in pd.concat([subj_emails, cont_emails]):
            all_emojis.extend(extract_emojis(text))

        top_emojis = Counter(all_emojis).most_common(5)
        top_emojis_str = ', '.join([f"{e} ({c})" for e, c in top_emojis])

        results.append({
            'Competitor': competitor_name,
            'Total Emails': len(comp_data),
            'Subjects with Emojis (%)': percent_subj,
            'Content with Emojis (%)': percent_cont,
            'Top Emojis (Subject + Content)': top_emojis_str
        })

    summary_df = pd.DataFrame(results)

    fig = go.Figure(
        data=[go.Table(
            header=dict(
                values=list(summary_df.columns),
                fill_color='lightgray',
                align='left',
                font=dict(size=20, color='black')
            ),
            cells=dict(
                values=[summary_df[col] for col in summary_df.columns],
                align='left',
                font=dict(size=15),
                height=40
            )
        )]
    )

    fig.update_layout(
        title='Emoji Usage in Email Subjects and Content by Competitor',
        height=400 + 30 * len(summary_df),
    )

    return fig


def bonus_top_offers_stacked_without_no_promo(df: pd.DataFrame, top_n: int = 30):
    d = _prepare_offer_cols(df)
    d["promotion_type"] = d["type"]

    # Remove No Promo rows BEFORE grouping
    d = d[~d["type"].isin(["No Promo", "No Promo - No Values"])]

    # Aggregate: how many times each competitor uses each offer
    agg = (
        d.groupby(["competitor_name", "offer_label"])
        .size()
        .reset_index(name="count")
    )

    # --- Compute stats per offer_label ---
    offer_stats = (
        agg.groupby("offer_label")
        .agg(
            n_competitors=("competitor_name", "nunique"),
            total=("count", "sum"),
        )
        .reset_index()
        .sort_values(
            ["total", "n_competitors"],  # 👈 first by total, then by competitors
            ascending=[False, False]
        )
    )

    # Keep only top_n offers globally
    top_offers = offer_stats.head(top_n)
    top_labels = top_offers["offer_label"].tolist()

    agg["offer_label"] = pd.Categorical(
        agg["offer_label"],
        categories=top_labels,
        ordered=True
    )

    # --- Colors per competitor ---
    competitors = agg["competitor_name"].unique().tolist()
    palette = (
            pc.qualitative.Plotly
            + pc.qualitative.Pastel
            + pc.qualitative.Set2
            + pc.qualitative.Set3
    )
    color_map = {comp: palette[i % len(palette)] for i, comp in enumerate(competitors)}

    # Plot
    fig = px.bar(
        agg,
        y="offer_label",
        x="count",
        color="competitor_name",
        orientation="h",
        text="count",
        title=f"Top {top_n} Offers by Competitor Coverage & Frequency",
        labels={
            "offer_label": "Offer",
            "count": "Instances",
            "competitor_name": "Competitor"
        },
        color_discrete_map=color_map
    )

    bar_height = max(500, top_n * 30)
    fig.update_traces(
        textposition="inside",
        insidetextanchor="middle",
        textangle=0
    )
    fig.update_layout(
        barmode="stack",
        yaxis={
            "categoryorder": "array",
            # reverse so the most common offer is shown at the TOP
            "categoryarray": top_labels[::-1]
        },
        height=bar_height,
        uniformtext_minsize=12,
        #uniformtext_mode="hide"
    )

    print(offer_stats)

    return fig


def bonus_top_offers_stacked(df: pd.DataFrame, top_n: int = 30):
    d = _prepare_offer_cols(df)
    d["promotion_type"] = d["type"]

    # Aggregate
    agg = (
        d.groupby(["type", "offer_label", "competitor_name"])
        .size()
        .reset_index(name="count")
    )

    totals = agg.groupby(["type", "offer_label"])["count"].sum().reset_index(name="total")
    top_offers = totals.sort_values("total", ascending=False).head(top_n)

    agg = agg.merge(top_offers[["type", "offer_label"]], on=["type", "offer_label"])
    agg = agg.merge(totals, on=["type", "offer_label"])

    # Ordering
    order = (
        agg.groupby(["type", "offer_label"])["total"]
        .max()
        .reset_index()
        .sort_values(["type", "total"], ascending=[True, False])
    )
    agg["offer_label"] = pd.Categorical(
        agg["offer_label"], categories=order["offer_label"].tolist(), ordered=True
    )

    # --- Create unique color map for competitors ---
    competitors = agg["competitor_name"].unique().tolist()
    palette = pc.qualitative.Plotly + pc.qualitative.Pastel + pc.qualitative.Set2 + pc.qualitative.Set3
    color_map = {comp: palette[i % len(palette)] for i, comp in enumerate(competitors)}

    # Plot
    fig = px.bar(
        agg,
        y="offer_label",
        x="count",
        color="competitor_name",
        orientation="h",
        text="count",  # show counts
        title=f"Top {top_n} Offers by Frequency (Stacked by Competitor, grouped by Promotion)",
        labels={"offer_label": "Offer", "count": "Instances", "competitor_name": "Competitor"},
        color_discrete_map=color_map
    )

    # Style text as vertical
    bar_height = max(500, top_n * 30)
    fig.update_traces(
        textposition="inside",
        insidetextanchor="middle",
        textangle=0  # 👈 rotate text vertically
    )
    fig.update_layout(
        barmode="stack",
        yaxis={"categoryorder": "array", "categoryarray": order["offer_label"].tolist()},
        height=bar_height,
        uniformtext_minsize=8,
        uniformtext_mode="hide"
    )

    return fig


def play_value(df):
    promo_df = df[['competitor_name', 'promotion_type', 'value', 'play_value']].copy()

    summary = (
        promo_df
        .groupby(['competitor_name', 'promotion_type', 'value'])['play_value'].agg(pd.Series.mode)
        .reset_index()
        .sort_values(['competitor_name', 'promotion_type', 'value'])
    )
    summary = summary[summary['promotion_type'] == 'Free Spins']
    summary = summary[~summary['play_value'] == ""]
    summary = summary[summary['play_value'].notna()]

    # --- Build Figure ---
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(summary.columns),
                    fill_color="lightgray",
                    align="left",
                    font=dict(size=14, color="black"),
                    height=30
                ),
                cells=dict(
                    values=[summary[col] for col in summary.columns],
                    fill_color=[["white"] * (len(summary) - 1)],  # gray for total
                    align="left",
                    font=dict(size=12),
                    height=25
                )
            )
        ]
    )

    fig.update_layout(
        title="Offer Value",
        margin=dict(l=20, r=20, t=40, b=20),
        height=800
    )

    return fig


def bonus_treemap_promotions(df: pd.DataFrame, scope: str = "overall"):
    """
    scope: "overall" -> one treemap
           "per_lifecycle" -> dict of lifecycle -> fig
           "per_competitor" -> dict of competitor -> fig
    """
    d = _prepare_offer_cols(df)

    def _agg(g):
        # offer x competitor; instances
        out = (
            g.groupby(['competitor_name', "offer_label"], dropna=False)
            .size()
            .reset_index(name="instances")
        )
        return out

    if scope == "overall":
        agg = _agg(d)
        fig = px.treemap(
            agg,
            path=["competitor_name", "offer_label"],
            values="instances",
            title="Promotion Mix Treemap (Overall)",
            hover_data={"instances": True}
        )
        # show counts inside each box
        fig.update_traces(texttemplate="%{label}<br>%{value}")
        return fig

    elif scope == "per_lifecycle":
        figs = {}
        for lc, g in d.groupby("lifecycle"):
            agg = _agg(g)
            if agg.empty:
                continue
            fig = px.treemap(
                agg,
                path=["competitor_name", "offer_label"],
                values="instances",
                title=f"Promotion Mix Treemap – {lc}",
                hover_data={"instances": True}
            )
            fig.update_traces(texttemplate="%{label}<br>%{value}")
            figs[f"Treemap – {lc}"] = fig
        return figs

    elif scope == "per_competitor":
        figs = {}
        for comp, g in d.groupby("competitor_name"):
            agg = (
                g.groupby(["offer_label"], dropna=False)
                .size()
                .reset_index(name="instances")
            )
            if agg.empty:
                continue
            fig = px.treemap(
                agg,
                path=["offer_label"],
                values="instances",
                title=f"Promotion Mix Treemap – {comp}",
                hover_data={"instances": True}
            )
            fig.update_traces(texttemplate="%{label}<br>%{value}")
            figs[f"Treemap – {comp}"] = fig
        return figs

    else:
        raise ValueError("scope must be one of: overall | per_lifecycle | per_competitor")


def bundles_pairwise_heatmap(df: pd.DataFrame, top_offers: int = 20):
    d = _prepare_offer_cols(df)

    # Limit to top offers by overall frequency to keep the matrix readable
    top = (
        d["offer_label"].value_counts()
        .head(top_offers)
        .index
        .tolist()
    )
    d = d[d["offer_label"].isin(top)]

    # Build per-hit sets for those offers
    per_hit = (
        d.groupby("tracking_hit_id")["offer_label"]
        .apply(lambda s: set(pd.unique(s.dropna())))
    )

    # Count pairwise co-occurrence
    idx = {lab: i for i, lab in enumerate(top)}
    matrix = [[0] * len(top) for _ in range(len(top))]
    for offers in per_hit:
        for a, b in itertools.combinations(sorted(offers), 2):
            i, j = idx[a], idx[b]
            matrix[i][j] += 1
            matrix[j][i] += 1
        # diagonal = single-offer presence (optional)
        for a in offers:
            i = idx[a]
            matrix[i][i] += 1

    co_df = pd.DataFrame(matrix, index=top, columns=top)

    fig = px.imshow(
        co_df,
        aspect="auto",
        title="Offer Pairwise Co-Occurrence (Top Offers)",
        labels=dict(x="Offer", y="Offer", color="Co-occurrence"),
        color_continuous_scale="Agsunset",
        text_auto=True
    )
    return fig


def bundle_size_distribution(df: pd.DataFrame, min_size: int = 1):
    #bundles, per_hit = compute_bundles(df, min_size=min_size)
    #size_counts = per_hit["bundle_size"].value_counts().sort_index().reset_index()

    # Step 1: count how many times each tracking_hit_id appears
    counts = df['tracking_hit_id'].value_counts()

    # Step 2: count how many hits fall into each bundle size
    size_counts = counts.value_counts().sort_index().rename_axis("bundle_size").reset_index(name="hits")

    # Step 3: apply minimum filter if needed
    size_counts = size_counts[size_counts["bundle_size"] >= min_size]

    # Step 4: plot
    fig = px.bar(
        size_counts,
        x="bundle_size", y="hits",
        title="Bundle Size Distribution (offers per hit)",
        labels={"bundle_size": "# Offers in a Hit", "hits": "# Hits"}
    )
    fig.update_traces(text=size_counts["hits"], textposition="outside")
    return fig


def bundles_treemap(df: pd.DataFrame, min_size: int = 2):
    bundles, _ = compute_bundles(df, min_size=min_size)
    if bundles.empty:
        return go.Figure().update_layout(title="Bundles Treemap (no bundles found)")

    # Path: Bundle size → Bundle label
    treedf = bundles.copy()
    treedf["size_bucket"] = treedf["bundle_size"].apply(lambda k: f"Bundles of {k}")
    fig = px.treemap(
        treedf,
        path=["size_bucket", "bundle_label"],
        values="count",
        color="bundle_size",
        title="Bundles Treemap (by size and frequency)",
        hover_data={"count": True, "bundle_size": True}
    )
    # Show counts inside nodes
    fig.update_traces(texttemplate="%{label}<br>%{value}")
    return fig


def bundles_top_bar(df: pd.DataFrame, top_n: int = 20, min_size: int = 2):
    bundles, _ = compute_bundles(df, min_size=min_size)
    topb = bundles.head(top_n).copy()
    # order on y-axis
    topb["bundle_label"] = pd.Categorical(topb["bundle_label"],
                                          categories=topb["bundle_label"].tolist(),
                                          ordered=True)
    fig = px.bar(
        topb.sort_values("count"),
        x="count", y="bundle_label", orientation="h", color="bundle_size",
        title=f"Top {top_n} Bundles (combinations of offers)",
        labels={"count": "Instances", "bundle_label": "Bundle", "bundle_size": "# Offers"}
    )
    # show counts on bars
    fig.update_traces(text="count", textposition="outside", cliponaxis=False)
    # height scales with items
    fig.update_layout(height=max(500, 28 * len(topb)))
    return fig


def number_of_email_first_48h(df):
    df = df.drop_duplicates(subset='tracking_hit_id')
    df['tracking_start_at'] = pd.to_datetime(df['tracking_start_at'])
    # Assume df has columns: competitor_name, email_type, type, registration_time, send_time
    df['time_since_reg'] = (df['local_created_at'] - df['tracking_start_at']).dt.total_seconds() / 3600

    # Filter only messages within the first 48 hours
    df_48h = df[df['time_since_reg'] <= 48]

    # 1. Number of emails in first 48 hours by competitor
    fig = px.histogram(
        df_48h,
        y="competitor_name",
        color="competitor_name",
        #barmode="group",
        title="Emails Sent in First 48 Hours by Competitor",
        text_auto=True,
    )
    fig.update_traces(
        textposition="inside",
        textangle=0,
    )
    fig.update_layout(
        xaxis_title="Number of Emails",
        yaxis_title="Competitor",
        legend_title="Competitor",
        bargap=0.01,
        height=500,

    )

    return fig


def email_types_first_48h(df):
    df = df.drop_duplicates(subset='tracking_hit_id')
    df['email_type'] = df['email_type'].fillna("Informational")

    df['tracking_start_at'] = pd.to_datetime(df['tracking_start_at'])
    # Assume df has columns: competitor_name, email_type, type, registration_time, send_time
    df['time_since_reg'] = (df['local_created_at'] - df['tracking_start_at']).dt.total_seconds() / 3600

    # Filter only messages within the first 48 hours
    df_48h = df[df['time_since_reg'] <= 48]

    # 2. Email types in first 48 hours by competitor
    fig = px.histogram(
        df_48h,
        y="competitor_name",
        color="email_type",
        barmode="group",
        title="Email Types Sent in First 48 Hours by Competitor",
        text_auto=True,
    )
    fig.update_traces(
        textposition="inside",
        textangle=0,
        insidetextfont=dict(size=50, color="black", family="Arial Bold")
    )
    fig.update_layout(
        xaxis_title="Number of Emails",
        yaxis_title="Competitor",
        legend_title="Email Type",
        bargap=0.02,
        height=600
    )

    return fig


def promoion_types_first_48h(df):
    df = df.drop_duplicates(subset='tracking_hit_id')
    df['tracking_start_at'] = pd.to_datetime(df['tracking_start_at'])
    df['type'] = df['type'].fillna("No Promo")
    # Assume df has columns: competitor_name, email_type, type, registration_time, send_time
    df['time_since_reg'] = (df['local_created_at'] - df['tracking_start_at']).dt.total_seconds() / 3600

    # Filter only messages within the first 48 hours
    df_48h = df[df['time_since_reg'] <= 48]

    # 3. Promotion types in first 48 hours by competitor
    fig = px.histogram(
        df_48h,
        y="competitor_name",
        color="type",
        barmode="group",
        title="Promotion Types Sent in First 48 Hours by Competitor",
        text_auto=True,
    )
    fig.update_traces(
        textposition="inside",
        textangle=0,
        insidetextfont=dict(size=50, color="black", family="Arial Bold")
    )
    fig.update_layout(
        xaxis_title="Number of Promotions",
        yaxis_title="Competitor",
        legend_title="Promotion Type",
        bargap=0.02,
        height=600
    )

    return fig


def get_esp_provider_from_ip(ip_address):
    """Fetch ESP provider info from ipinfo.io for a given IP address."""
    url = f"http://ipinfo.io/{ip_address}/json"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("org", "Unknown")
    except requests.RequestException:
        pass
    return "Unknown"


def build_esp_provider_heatmap(df):
    """
    Build and return a Plotly heatmap showing ESP provider concentration by competitor.

    Parameters:
        df (pd.DataFrame): Must contain columns ['competitor_name', 'from_ip']

    Returns:
        fig (plotly.graph_objects.Figure): Interactive heatmap figure
    """
    # Collect providers per competitor
    rows = []
    for competitor_name in df['competitor_name'].dropna().unique():
        comp_ips = df.loc[df['competitor_name'] == competitor_name, 'from_ip'].dropna().unique()
        providers = [get_esp_provider_from_ip(ip) for ip in comp_ips]
        provider_counts = Counter(providers)
        for provider, count in provider_counts.items():
            rows.append({
                "competitor_name": competitor_name,
                "provider": provider,
                "count": count
            })

    # Build DataFrame
    df_providers = pd.DataFrame(rows)

    if df_providers.empty:
        raise ValueError("No provider data could be resolved from the given DataFrame.")

    # Pivot for heatmap
    pivot = df_providers.pivot(index="competitor_name", columns="provider", values="count").fillna(0)

    # Plot
    fig = px.imshow(
        pivot,
        labels=dict(x="Provider", y="Competitor", color="Unique IPs"),
        title="ESP Providers by Competitor",
        text_auto=True,
        aspect="auto",
    )
    fig.update_xaxes(side="bottom")
    fig.update_layout(
        title_x=0.5,
        xaxis_title="Email Service Provider (ESP)",
        yaxis_title="Competitor",
        coloraxis_colorbar=dict(title="IP Count")
    )

    return fig, df_providers


def build_esp_provider_table_from_providers(df_providers):
    """
    Chart: Table of each competitor and their unique ESP providers.
    Data: Uses the df_providers table (from the ESP heatmap) containing
          ['competitor_name', 'provider', 'count'].
    Output: Plotly table figure + formatted DataFrame.
    """
    # Validate input
    required = {'competitor_name', 'provider', 'count'}
    missing = required - set(df_providers.columns)
    if missing:
        raise KeyError(f"build_esp_provider_table_from_providers: missing columns {sorted(missing)}")

    # Aggregate unique providers per competitor
    table_df = (
        df_providers.groupby('competitor_name')['provider']
        .apply(lambda s: '; '.join(sorted(set(s.dropna()))))
        .reset_index()
        .rename(columns={'competitor_name': 'Competitor', 'provider': 'Providers'})
    )

    # Sort alphabetically
    table_df = table_df.sort_values('Competitor').reset_index(drop=True)

    # Build Plotly Table
    fig = go.Figure(
        data=[go.Table(
            header=dict(
                values=['<b>Competitor</b>', '<b>Providers</b>'],
                fill_color='lightgray',
                align='left',
                font=dict(size=16, color='black'),
                height=34
            ),
            cells=dict(
                values=[table_df['Competitor'], table_df['Providers']],
                align='left',
                font=dict(size=14),
                height=30
            )
        )]
    )

    fig.update_layout(
        title='ESP Providers by Competitor',
        margin=dict(t=60, l=60, r=60, b=40),
        height=max(360, 300 + 28 * len(table_df)),
    )

    return fig


def heatmap_by_day_hour_with_legend(df):
    """
    Chart: Single heatmap (hour × weekday) with buttons to exclude one competitor at a time.
    Data: Uses ['tracking_hit_id','competitor_name','local_created_at'] (deduped).
    Output: Plotly heatmap with 'All' + 'Exclude {competitor}' buttons (left or bottom).
    """
    # --- clean & prep ---
    d = (
        df.drop_duplicates(subset="tracking_hit_id")
        .assign(local_created_at=pd.to_datetime(df["local_created_at"], errors="coerce"))
        .dropna(subset=["local_created_at"])
        .copy()
    )
    d["hour"] = d["local_created_at"].dt.hour
    d["day_of_week"] = d["local_created_at"].dt.day_name()

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    hours = list(range(24))

    # helper to build a day×hour matrix from a filtered frame
    def build_matrix(df_filt):
        base = (
            df_filt.groupby(["day_of_week", "hour"])
            .size()
            .unstack(fill_value=0)
        )
        base = base.reindex(index=weekday_order, columns=hours, fill_value=0)
        return base

    # ALL matrix
    mat_all = build_matrix(d)

    # per-competitor exclusions
    competitors = sorted([c for c in d["competitor_name"].dropna().unique().tolist()])
    matrices = {"All": mat_all}
    for comp in competitors:
        matrices[f"Exclude: {comp}"] = build_matrix(d[d["competitor_name"] != comp])

    # initial figure with "All"
    fig = go.Figure(
        data=[go.Heatmap(
            z=matrices["All"].values,
            x=matrices["All"].columns,  # 0–23
            y=matrices["All"].index,  # Mon–Sun
            colorscale="RdYlGn_r",
            zmin=0,
            showscale=True,
            text=matrices["All"].values,
            texttemplate="%{text}",
            hovertemplate="Day: %{y}<br>Hour: %{x}:00<br>Communications: %{z}<extra></extra>",
        )]
    )

    fig.update_layout(
        title="Actual Hourly Volume by Day of Week",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        xaxis=dict(dtick=1),
        height=600,
        margin=dict(t=60, l=80, r=40, b=60),
        template="plotly_white"
    )

    # --- build buttons: one for "All", then "Exclude: {competitor}" for each ---
    buttons = []
    for label, mat in matrices.items():
        buttons.append(dict(
            label=label,
            method="restyle",
            args=[{
                "z": [mat.values],
                "x": [mat.columns],
                "y": [mat.index],
            }]
        ))

    # Place buttons on the LEFT (vertical stack). Switch to 'x=0.5,y=-0.15' for bottom.
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            direction="down",  # vertical list
            x=0.0, y=0.5,  # left side, centered vertically
            xanchor="left", yanchor="middle",
            buttons=buttons,
            bgcolor="rgba(240,240,240,0.9)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1,
            pad=dict(r=6, t=6, b=6, l=6),
            font=dict(size=12)
        )]
    )

    return fig


def seasonal_start_timeline(df):
    df, first_mentions, activity = preprocess_holiday_data(df)

    fig = px.scatter(
        first_mentions,
        x="first_mention_date",
        y="competitor",
        color="holiday_type",
        hover_data=["first_subject", "first_offer", "first_requirements"],
        title="Seasonal Start Timeline per Competitor",
        height=600,
        color_discrete_map={
            "Black Friday": "#000000",  # Black
            "Christmas": "#FF0000"  # Red
        }
    )

    fig.update_traces(marker=dict(size=12))
    fig.update_layout(yaxis=dict(categoryorder="array", categoryarray=sorted(first_mentions["competitor"].unique())))

    return fig


def seasonal_activity_barchart(df):
    df, first_mentions, activity = preprocess_holiday_data(df)

    # Manual color mapping
    color_map = {
        "Black Friday": "#000000",  # Black
        "Christmas": "#FF0000"  # Red
    }

    fig = go.Figure()

    for holiday in activity.columns:
        fig.add_trace(go.Bar(
            x=activity.index,
            y=activity[holiday],
            name=holiday,
            marker_color=color_map.get(holiday, "#888888")  # fallback grey
        ))

    fig.update_layout(
        barmode="group",
        title="Seasonal Activity per Competitor",
        xaxis_title="Competitor",
        yaxis_title="Number of Seasonal Emails",
        height=600
    )

    return fig


def holiday_offer_heatmap(df):
    df, first_mentions, activity = preprocess_holiday_data(df)

    fig = px.imshow(
        activity,
        labels=dict(x="Holiday", y="Competitor", color="Count"),
        x=activity.columns,
        y=activity.index,
        color_continuous_scale="RdBu",
        title="Seasonal Offer Intensity Heatmap"
    )

    fig.update_layout(height=700)

    return fig


def holiday_promos_table_fig(df, competitor=None):
    holiday_table = competitor_holiday_promos_table(df)

    data = holiday_table.copy()

    if competitor is not None:
        data = data[data["competitor_name"] == competitor]

    # Ensure string formatting
    data = data.assign(
        date=data["date"].dt.strftime("%d-%B-%Y"),
        wager=data["wager"].fillna(""),
        requirements=data["requirements"].fillna(""),
        offer=data["offer"].fillna("")
    )

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[
                        "Competitor",
                        "Holiday",
                        "Date",
                        "Subject",
                        "Offer",
                        "Wager",
                        "Requirements",
                    ],
                    fill_color="#222222",
                    font=dict(color="white", size=12),
                    align="left",
                ),
                cells=dict(
                    values=[
                        data["competitor_name"],
                        data["holiday"],
                        data["date"],
                        data["subject"],
                        data["offer"],
                        data["wager"],
                        data["requirements"],
                    ],
                    align="left",
                    font=dict(size=11),
                    height=24,
                ),
            )
        ]
    )
    fig.update_layout(
        title="Holiday Promotions per Competitor",
        height=1000
    )

    return fig


def holiday_communication_timeline(df):
    temp_df = df.copy()

    def get_marker_emoji(row):
        if row["holiday"] == "Christmas":
            return "🎄"
        if row["holiday"] == "Black Friday":
            return "🖤"  # or "🛍️" or "💸"
        return ""  # normal dot for non-holiday

    # --- Build base journey_df (same idea as your communication_timeline) ---
    journey_df = temp_df[
        [
            'competitor_name',
            'created_at',
            'subject',
            'promotion_type',
            'email_type',
            'type',
            'tracking_start_at',
            'holiday',  # existing column
            'translated_content'  # used only to upgrade holiday if needed
        ]
    ].copy()

    # Datetime
    journey_df['created_at'] = pd.to_datetime(journey_df['created_at'], errors='coerce')

    # --- Relative day offset per competitor ---
    journey_df['day_offset'] = (
            journey_df['created_at'].dt.normalize()
            - journey_df.groupby('competitor_name')['created_at'].transform(lambda x: x.min().normalize())
    ).dt.days

    # Sort
    journey_df = journey_df.sort_values(by=['competitor_name', 'day_offset'])

    # Optional label
    journey_df['day_index'] = "Day " + journey_df['day_offset'].astype(str)

    # --- Fix type vs email_type (same logic as your original) ---
    journey_df['type'] = journey_df['type'].fillna("No Promo")

    def fix_type(row):
        message_type = row['email_type']
        promo_type = row['type']

        if message_type == "Promotional" and promo_type == "No Promo":
            return "Branding"
        if message_type in ['Welcome', 'Informational', 'System', 'Transactional'] and promo_type == "No Promo":
            return "System Message"
        return promo_type

    journey_df['type'] = journey_df.apply(fix_type, axis=1)

    # ======================================================================
    #           U P G R A D E   /   N O R M A L I Z E   H O L I D A Y
    # ======================================================================

    # Work on a copy of the existing holiday column
    # (we will write back into journey_df['holiday'])
    h = journey_df['holiday'].astype(str).str.strip()

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

    journey_df["holiday"] = journey_df.apply(
        lambda r: normalize_holiday(r.get("holiday"), r.get("subject"), r.get("translated_content")),
        axis=1
    )

    # ======================================================================
    #                C O L O R   L A B E L   ( H O L I D A Y )
    # ======================================================================

    def color_label(row):
        if row['holiday'] == 'Christmas':
            return 'Christmas'
        if row['holiday'] == 'Black Friday':
            return 'Black Friday'
        # Non-holiday → color by message/promo type
        return row['type']

    journey_df['color_label'] = journey_df.apply(color_label, axis=1)
    journey_df["emoji"] = journey_df.apply(get_marker_emoji, axis=1)

    # Build color map: Black Friday = black, Christmas = red, others from palette
    unique_labels = journey_df['color_label'].unique().tolist()
    palette = pc.qualitative.Light24
    color_map = {
        'Black Friday': '#000000',  # black
        'Christmas': '#FF0000'  # red
    }

    idx = 0
    for label in unique_labels:
        if label in color_map:
            continue
        color_map[label] = palette[idx % len(palette)]
        idx += 1

    # ======================================================================
    #                         P L O T   T I M E L I N E
    # ======================================================================

    fig = px.scatter(
        journey_df,
        x="created_at",
        y="competitor_name",
        color="color_label",
        hover_data=["subject", "type", "holiday"],
        title="Email Communication Timeline by Competitor (Holidays Highlighted)",
        color_discrete_map=color_map
    )

    fig.update_traces(marker=dict(size=12, opacity=0.7))
    fig.update_layout(
        height=800,
        width=1500,
        xaxis_title="Relative Day of Sending",
        yaxis_title="Competitor",
        legend_title="Type / Holiday",
        margin=dict(t=60, l=60, r=40, b=60),
    )

    # Apply emoji as text markers per trace
    for i, trace in enumerate(fig.data):
        mask = (journey_df["color_label"] == trace.name)

        fig.data[i].mode = "text+markers"
        fig.data[i].text = journey_df.loc[mask, "emoji"]
        fig.data[i].textposition = "middle center"
        fig.data[i].textfont = dict(size=25)
    return fig


def demo_channel_overview(df):
    # Replace this with your real data
    # -----------------------------
    data = {
        "competitor": ["CasinoMax", "JackpotCity", "SlotsOfVegas", "Stake", "VegasCasinoOnline"],
        "email": [120, 80, 140, 200, 95],
        "sms": [30, 50, 20, 10, 55],
        "calls": [5, 2, 1, 8, 4]
    }

    df = pd.DataFrame(data)

    # Calculate totals and percentages
    df["total"] = df[["email", "sms", "calls"]].sum(axis=1)
    df["email_pct"] = df["email"] / df["total"] * 100
    df["sms_pct"] = df["sms"] / df["total"] * 100
    df["calls_pct"] = df["calls"] / df["total"] * 100

    # -----------------------------
    # Build figure
    # -----------------------------
    fig = go.Figure()

    # Email
    fig.add_trace(go.Bar(
        y=df["competitor"],
        x=df["email_pct"],
        name="Email",
        orientation="h",
        hovertemplate="<b>%{y}</b><br>Email: %{customdata} communications<extra></extra>",
        customdata=df["email"]
    ))

    # SMS
    fig.add_trace(go.Bar(
        y=df["competitor"],
        x=df["sms_pct"],
        name="SMS",
        orientation="h",
        hovertemplate="<b>%{y}</b><br>SMS: %{customdata} communications<extra></extra>",
        customdata=df["sms"]
    ))

    # Calls
    fig.add_trace(go.Bar(
        y=df["competitor"],
        x=df["calls_pct"],
        name="Calls",
        orientation="h",
        hovertemplate="<b>%{y}</b><br>Calls: %{customdata} communications<extra></extra>",
        customdata=df["calls"]
    ))

    # Layout
    fig.update_layout(
        barmode="stack",
        title="Communication Type Distribution by Competitor (Percentage)",
        xaxis_title="Percentage of Communications",
        yaxis_title="Competitor",
        xaxis=dict(ticksuffix="%"),
        height=450,
        legend_title="Communication Type",
        template="plotly_white"
    )

    return fig
