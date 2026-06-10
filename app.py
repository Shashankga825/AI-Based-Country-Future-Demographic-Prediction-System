# ==============================
# Demographic Prediction System
# ==============================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

from demographic_ml import load_and_preprocess, get_country_data, train_models, predict_for_year, build_projection_timeseries

# ==============================
# PAGE SETUP
# ==============================
st.set_page_config(
    page_title="DemoScope — Population Forecasts",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .stApp { background: #0a0e1a; color: #e0e8f0; }
    section[data-testid="stSidebar"] { background: #0d1220; border-right: 1px solid #1e3050; }
    section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label { color: #7fa8c9 !important; }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2 { color: #00e5c0 !important; letter-spacing: 2px; text-transform: uppercase; font-size: 13px !important; }
    div[data-testid="metric-container"] { background: #111827; border: 1px solid #1e3050; border-radius: 8px; padding: 16px !important; border-left: 3px solid #00e5c0; }
    div[data-testid="metric-container"] label { color: #7fa8c9 !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 1px; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #ffffff !important; font-family: 'JetBrains Mono', monospace !important; font-size: 22px !important; font-weight: 600; }
    h1 { color: #ffffff !important; font-size: 32px !important; font-weight: 700 !important; letter-spacing: -0.5px; }
    h2 { color: #00e5c0 !important; font-size: 16px !important; text-transform: uppercase; letter-spacing: 2px; font-weight: 500 !important; margin-top: 32px !important; }
    h3 { color: #7fa8c9 !important; font-size: 13px !important; text-transform: uppercase; letter-spacing: 1px; }
    .stButton > button { background: linear-gradient(135deg, #00e5c0, #0080ff) !important; color: #000 !important; border: none !important; border-radius: 6px !important; font-family: 'Space Grotesk', sans-serif !important; font-weight: 700 !important; letter-spacing: 1px; text-transform: uppercase; font-size: 12px !important; padding: 12px 0 !important; }
    .stButton > button:hover { opacity: 0.9 !important; transform: translateY(-1px); }
    .stSelectbox > div > div { background: #111827 !important; border: 1px solid #1e3050 !important; color: #e0e8f0 !important; border-radius: 6px !important; }
    .stSlider > div > div > div > div { background: #00e5c0 !important; }
    hr { border-color: #1e3050 !important; }
    .stAlert { background: #111827 !important; border: 1px solid #1e3050 !important; color: #7fa8c9 !important; border-radius: 8px !important; }
    .stDataFrame { border: 1px solid #1e3050 !important; border-radius: 8px !important; }
    .section-label { background: #0d1220; border: 1px solid #1e3050; border-left: 3px solid #00e5c0; padding: 8px 16px; margin: 24px 0 16px 0; border-radius: 0 6px 6px 0; font-size: 11px; text-transform: uppercase; letter-spacing: 2px; color: #00e5c0; }
    .score-badge { display: inline-block; background: #0d1220; border: 1px solid #1e3050; border-radius: 4px; padding: 4px 10px; font-family: 'JetBrains Mono', monospace; font-size: 12px; margin: 2px; }
    .score-good { border-color: #00e5c0; color: #00e5c0; }
    .score-ok   { border-color: #f0b429; color: #f0b429; }
    .score-poor { border-color: #ff4d6d; color: #ff4d6d; }
    .stSpinner > div { border-top-color: #00e5c0 !important; }
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0d1220",
    font=dict(family="Space Grotesk", color="#7fa8c9", size=11),
    xaxis=dict(gridcolor="#1e3050", zerolinecolor="#1e3050", color="#7fa8c9"),
    yaxis=dict(gridcolor="#1e3050", zerolinecolor="#1e3050", color="#7fa8c9"),
    margin=dict(t=40, b=40, l=10, r=10),
)

TEAL  = "#00e5c0"
BLUE  = "#0080ff"
ORANGE = "#f0b429"
RED   = "#ff4d6d"

# ==============================
# HEADER
# ==============================
col_title, col_sub = st.columns([3, 1])
with col_title:
    st.markdown("# 🌍 DemoScope")
    st.markdown("<p style='color:#7fa8c9;margin-top:-12px;font-size:14px;'>Machine-learning population forecasts · World Bank data · 218 countries</p>", unsafe_allow_html=True)

st.markdown("---")

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_demographic_data():
    return load_and_preprocess("dataset.csv")

try:
    df = load_demographic_data()
    countries = sorted(df["Country Name"].unique())

    # ==============================
    # SIDEBAR
    # ==============================
    st.sidebar.markdown("## ⚙ Settings")
    st.sidebar.markdown("---")

    selected_country = st.sidebar.selectbox(
        "Country", countries,
        index=countries.index("India") if "India" in countries else 0
    )
    selected_year = st.sidebar.slider("Forecast Year", min_value=2025, max_value=2100, value=2050, step=1)

    st.sidebar.markdown("---")

    # ==============================
    # SESSION STATE — persists across reloads and shared links
    # ==============================
    if "run_forecast" not in st.session_state:
        st.session_state["run_forecast"] = False
    if "last_country" not in st.session_state:
        st.session_state["last_country"] = selected_country
    if "last_year" not in st.session_state:
        st.session_state["last_year"] = selected_year

    # Reset when country or year changes
    if (selected_country != st.session_state["last_country"] or
            selected_year != st.session_state["last_year"]):
        st.session_state["run_forecast"] = False
        st.session_state["last_country"] = selected_country
        st.session_state["last_year"] = selected_year

    if st.sidebar.button("▶  Generate Forecast", use_container_width=True):
        st.session_state["run_forecast"] = True

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<p style='color:#4a6a8a;font-size:11px;'>Models: Ridge · SVR · Random Forest<br>"
        "GradientBoosting · AdaBoost · Lasso<br>Auto-selects best per country</p>",
        unsafe_allow_html=True
    )

    # ==============================
    # HISTORICAL OVERVIEW (always visible)
    # ==============================
    country_df = get_country_data(df, selected_country)

    st.markdown(f"<div class='section-label'>Historical Data — {selected_country}</div>", unsafe_allow_html=True)

    years_avail = int(country_df['Year'].min()), int(country_df['Year'].max())
    last  = country_df.iloc[-1]
    first = country_df.iloc[0]

    h1, h2, h3, h4 = st.columns(4)
    with h1:
        st.metric("Current Population", f"{int(last['TotalPopulation']):,}")
    with h2:
        st.metric("Data Coverage", f"{years_avail[0]}–{years_avail[1]}", f"{int(country_df['Year'].count())} years")
    with h3:
        st.metric("Birth Rate (latest)", f"{last['BirthRate']:.1f} / 1k")
    with h4:
        st.metric("Death Rate (latest)", f"{last['DeathRate']:.1f} / 1k")

    hc1, hc2 = st.columns(2)
    with hc1:
        fig_hist_pop = go.Figure()
        fig_hist_pop.add_trace(go.Scatter(
            x=country_df['Year'], y=country_df['TotalPopulation'],
            mode='lines+markers',
            line=dict(color=TEAL, width=2.5),
            marker=dict(size=5, color=TEAL),
            fill='tozeroy', fillcolor='rgba(0,229,192,0.07)',
            name="Total Population"
        ))
        fig_hist_pop.update_layout(**PLOT_LAYOUT,
            title=dict(text="Population Over Time", font=dict(color="#e0e8f0", size=13)), height=260)
        st.plotly_chart(fig_hist_pop, use_container_width=True)

    with hc2:
        fig_rates = go.Figure()
        fig_rates.add_trace(go.Scatter(
            x=country_df['Year'], y=country_df['BirthRate'],
            mode='lines', name="Birth Rate", line=dict(color=TEAL, width=2)
        ))
        fig_rates.add_trace(go.Scatter(
            x=country_df['Year'], y=country_df['DeathRate'],
            mode='lines', name="Death Rate", line=dict(color=RED, width=2, dash='dash')
        ))
        fig_rates.update_layout(**PLOT_LAYOUT,
            title=dict(text="Birth & Death Rates per 1,000", font=dict(color="#e0e8f0", size=13)),
            height=260, legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#7fa8c9")))
        st.plotly_chart(fig_rates, use_container_width=True)

    # ==============================
    # FORECAST SECTION
    # ==============================
    if st.session_state["run_forecast"]:
        with st.spinner(f"Training models for {selected_country}..."):
            models     = train_models(country_df)
            result     = predict_for_year(country_df, models, selected_year)
            projection = build_projection_timeseries(country_df, models, selected_year)

        st.markdown(
            f"<div class='section-label'>Forecast Results — {selected_country} · {selected_year}</div>",
            unsafe_allow_html=True
        )

        # Metric cards
        c1, c2, c3, c4, c5 = st.columns(5)
        last_pop  = int(country_df.iloc[-1]['TotalPopulation'])
        pred_pop  = int(result['Total Population'])
        delta_pop = f"{((pred_pop - last_pop) / last_pop * 100):+.1f}% from today"

        with c1: st.metric("Total Population", f"{pred_pop:,}", delta_pop)
        with c2: st.metric("Male",   f"{int(result['Male Population']):,}")
        with c3: st.metric("Female", f"{int(result['Female Population']):,}")
        with c4: st.metric("Urban",  f"{int(result['Urban Population']):,}")
        with c5: st.metric("Rural",  f"{int(result['Rural Population']):,}")

        r1, r2 = st.columns(2)
        with r1: st.metric("Birth Rate", f"{result['Birth Rate']:.2f} per 1,000")
        with r2: st.metric("Death Rate", f"{result['Death Rate']:.2f} per 1,000")

        # Population projection chart
        st.markdown("<div class='section-label'>Population Projection Timeline</div>", unsafe_allow_html=True)

        last_hist_year = int(country_df['Year'].max())
        hist_part = projection[projection['Year'] <= last_hist_year]
        proj_part = projection[projection['Year'] >  last_hist_year]

        fig_proj = go.Figure()
        fig_proj.add_trace(go.Scatter(
            x=hist_part['Year'], y=hist_part['TotalPopulation'],
            mode='lines', name="Historical",
            line=dict(color=TEAL, width=2.5),
            fill='tozeroy', fillcolor='rgba(0,229,192,0.06)'
        ))
        fig_proj.add_trace(go.Scatter(
            x=proj_part['Year'], y=proj_part['TotalPopulation'],
            mode='lines', name="Forecast",
            line=dict(color=ORANGE, width=2.5, dash='dot'),
            fill='tozeroy', fillcolor='rgba(240,180,41,0.05)'
        ))
        fig_proj.add_trace(go.Scatter(
            x=list(proj_part['Year']) + list(proj_part['Year'])[::-1],
            y=list(proj_part['TotalPopulation'] * 1.05) + list(proj_part['TotalPopulation'] * 0.95)[::-1],
            fill='toself', fillcolor='rgba(240,180,41,0.07)',
            line=dict(color='rgba(0,0,0,0)'),
            name="±5% Uncertainty"
        ))
        fig_proj.add_vline(x=last_hist_year, line_dash="dash", line_color="#1e3050",
                           annotation_text="Today", annotation_font_color="#7fa8c9")
        fig_proj.update_layout(**PLOT_LAYOUT, height=340,
            title=dict(text="Population Trajectory", font=dict(color="#e0e8f0", size=13)),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#7fa8c9")),
            yaxis_title="Population", xaxis_title="Year")
        st.plotly_chart(fig_proj, use_container_width=True)

        p1, p2 = st.columns(2)
        with p1:
            fig_urb = go.Figure()
            fig_urb.add_trace(go.Scatter(
                x=projection['Year'], y=projection['Urban'],
                name="Urban", mode='lines', line=dict(color=BLUE, width=2),
                fill='tozeroy', fillcolor='rgba(0,128,255,0.07)'
            ))
            fig_urb.add_trace(go.Scatter(
                x=projection['Year'], y=projection['RuralPopulation'],
                name="Rural", mode='lines', line=dict(color="#8b5cf6", width=2),
                fill='tozeroy', fillcolor='rgba(139,92,246,0.07)'
            ))
            fig_urb.add_vline(x=last_hist_year, line_dash="dash", line_color="#1e3050")
            fig_urb.update_layout(**PLOT_LAYOUT, height=280,
                title=dict(text="Urban vs Rural Population", font=dict(color="#e0e8f0", size=13)),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#7fa8c9")))
            st.plotly_chart(fig_urb, use_container_width=True)

        with p2:
            fig_rates_proj = go.Figure()
            fig_rates_proj.add_trace(go.Scatter(
                x=projection['Year'], y=projection['BirthRate'],
                name="Birth Rate", mode='lines', line=dict(color=TEAL, width=2)
            ))
            fig_rates_proj.add_trace(go.Scatter(
                x=projection['Year'], y=projection['DeathRate'],
                name="Death Rate", mode='lines', line=dict(color=RED, width=2, dash='dash')
            ))
            fig_rates_proj.add_vline(x=last_hist_year, line_dash="dash", line_color="#1e3050")
            fig_rates_proj.update_layout(**PLOT_LAYOUT, height=280,
                title=dict(text="Projected Birth & Death Rates", font=dict(color="#e0e8f0", size=13)),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#7fa8c9")))
            st.plotly_chart(fig_rates_proj, use_container_width=True)

        # ==============================
        # MODEL ACCURACY
        # ==============================
        st.markdown("<div class='section-label'>Model Accuracy</div>", unsafe_allow_html=True)

        def score_badge(label, r2):
            score = max(0.0, min(1.0, r2))
            pct   = score * 100
            cls   = "score-good" if pct >= 95 else ("score-ok" if pct >= 85 else "score-poor")
            icon  = "✓" if pct >= 95 else ("~" if pct >= 85 else "✗")
            return f"<span class='score-badge {cls}'>{icon} {label}: {pct:.1f}%</span>"

        badges = (
            score_badge("Total Population",  models.total_population_r2) +
            score_badge("Birth Rate",         models.birth_rate_r2) +
            score_badge("Death Rate",         models.death_rate_r2) +
            score_badge("Urban Population",   models.urban_population_r2)
        )
        st.markdown(badges, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        acc_col1, acc_col2 = st.columns(2)
        with acc_col1:
            rmse_df = pd.DataFrame({
                "Metric": ["Total Population", "Birth Rate", "Death Rate", "Urban Population"],
                "RMSE": [
                    f"{models.total_population_rmse:,.0f}",
                    f"{models.birth_rate_rmse:.4f}",
                    f"{models.death_rate_rmse:.4f}",
                    f"{models.urban_population_rmse:,.0f}"
                ]
            })
            st.caption("Root Mean Squared Error")
            st.dataframe(rmse_df, use_container_width=True, hide_index=True)

        with acc_col2:
            mae_df = pd.DataFrame({
                "Metric": ["Total Population", "Birth Rate", "Death Rate", "Urban Population"],
                "MAE": [
                    f"{models.total_population_mae:,.0f}",
                    f"{models.birth_rate_mae:.4f}",
                    f"{models.death_rate_mae:.4f}",
                    f"{models.urban_population_mae:,.0f}"
                ]
            })
            st.caption("Mean Absolute Error")
            st.dataframe(mae_df, use_container_width=True, hide_index=True)

        # Gender & Urban/Rural donuts
        st.markdown(
            f"<div class='section-label'>Forecast Breakdown — {selected_year}</div>",
            unsafe_allow_html=True
        )

        d1, d2 = st.columns(2)
        with d1:
            fig_gender = go.Figure(go.Pie(
                labels=["Male", "Female"],
                values=[result['Male Population'], result['Female Population']],
                hole=0.55, marker_colors=[BLUE, RED],
                textinfo='label+percent',
                textfont=dict(family='Space Grotesk', color='#e0e8f0')
            ))
            fig_gender.update_layout(**PLOT_LAYOUT, height=280,
                title=dict(text="Gender Distribution", font=dict(color="#e0e8f0", size=13)))
            st.plotly_chart(fig_gender, use_container_width=True)

        with d2:
            fig_urb2 = go.Figure(go.Pie(
                labels=["Urban", "Rural"],
                values=[result['Urban Population'], result['Rural Population']],
                hole=0.55, marker_colors=[TEAL, "#8b5cf6"],
                textinfo='label+percent',
                textfont=dict(family='Space Grotesk', color='#e0e8f0')
            ))
            fig_urb2.update_layout(**PLOT_LAYOUT, height=280,
                title=dict(text="Urban vs Rural", font=dict(color="#e0e8f0", size=13)))
            st.plotly_chart(fig_urb2, use_container_width=True)

    else:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("← Select a country and year, then click **Generate Forecast** to see predictions and projections.")

except Exception as e:
    st.error("Could not load demographic data")
    st.write(f"Error: {str(e)}")
    st.info("Check that the data file is available")

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.markdown("""
<p style='color:#4a6a8a;font-size:11px;text-align:center;'>
DemoScope · Built with World Bank demographic data ·
Models: Ridge, SVR, RandomForest, GradientBoosting, AdaBoost ·
218 countries · 2025–2100
</p>
""", unsafe_allow_html=True)
