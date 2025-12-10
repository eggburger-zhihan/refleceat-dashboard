"""
SmartSnack Monitor - Interactive Dashboard v9 (Logic Restored)
==============================================
VERSION 9 CHANGES:
1. Conditional Logic: 
   - IF Single Day: Shows only Daily Detail & Timelines (Clean view).
   - IF Multi-Day: RESTORES Correlation Analysis, Hypothesis Testing, and Interesting Findings.
2. Base: Built upon V8 (Pie charts in Overview/Single Day, Fixed Layouts).
3. Analytics: Re-implemented Scipy stats and detailed correlation logic from V5.

Author: Winnie (DE4-SIOT Final Project)
Date: 2024-12-08
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
from supabase import create_client

# Try import scipy for hypothesis testing
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# =========================================================================
# PAGE CONFIG
# =========================================================================

st.set_page_config(
    page_title="ReflecEAT",
    page_icon="🪞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #1f77b4; margin-bottom: 0;}
    .sub-header {font-size: 1rem; color: #666; margin-top: 0; margin-bottom: 1rem;}
</style>
""", unsafe_allow_html=True)

# =========================================================================
# CONSTANTS
# =========================================================================

EMOTION_COLORS = {
    'positive': "#6cffa9",      # Green
    'neutral': "#a2e5ff",       # Orange
    'negative_low': "#ffb25b",  # Purple
    'negative_high': "#ff7a6b"  # Red
}

HEALTH_COLORS = {
    'Healthy': "#009840",
    'Unhealthy': "#c61400"
}

EMOTION_EMOJI = {
    'positive': '😊', 'neutral': '😐', 'negative_low': '😟', 'negative_high': '😠'
}

WEATHER_EMOJI = {
    'Clear': '☀️', 'Clouds': '☁️', 'Rain': '🌧️', 
    'Mist': '🌫️', 'Snow': '❄️', 'Drizzle': '🌦️'
}

# Calorie database
CALORIE_DB = {'cherry_tomato': 5, 'baby_carrot': 4, 'banana': 90, 'apple': 95,
              'donut': 260, 'chip': 15, 'cookie': 46, 'nutella': 115}

def get_calories(food_type, db_cal=None):
    if db_cal and db_cal > 0:
        return db_cal
    return CALORIE_DB.get(food_type.lower().strip() if food_type else '', 100)

# Helper function to create simple donut charts
def create_donut(labels, values, colors, center_text, title=None, height=140):
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=.6,
        marker=dict(colors=colors), textinfo='percent',
        hoverinfo='label+value', showlegend=False
    )])
    fig.update_layout(
        title=dict(text=title, x=0.5, y=0.95, font=dict(size=14)) if title else None,
        annotations=[dict(text=center_text, x=0.5, y=0.5, font_size=14, showarrow=False)],
        margin=dict(l=5, r=5, t=25 if title else 5, b=5), height=height
    )
    return fig

# =========================================================================
# DATA LOADING (SUPABASE CLOUD)
# =========================================================================

SUPABASE_URL = "https://xujvkshguilaecrbdniq.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inh1anZrc2hndWlsYWVjcmJkbmlxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjUzODM0NDUsImV4cCI6MjA4MDk1OTQ0NX0.kJm6mpP7hxdYvJ4jE7658ROuCSGvYSvaSuAlPxsAPQ8"

@st.cache_data(ttl=60)
def load_data():
    
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

        def fetch_table(table_name):
            resp = (
                supabase
                .table(table_name)
                .select("*")
                .order("timestamp", desc=False) 
                .limit(5000)
                .execute()
            )
            return pd.DataFrame(resp.data)

        data = {
            "emotion": fetch_table("emotion_log"),
            "food": fetch_table("food_event_log"),
            "light": fetch_table("environment_light_log"),
            "weather": fetch_table("weather_log"),
        }

    except Exception as e:
        st.error(f"Supabase error: {e}")
        return {k: pd.DataFrame() for k in ["emotion", "food", "light", "weather"]}

    for key in data:
        if not data[key].empty and "timestamp" in data[key].columns:
            data[key]["timestamp"] = pd.to_datetime(data[key]["timestamp"]).dt.tz_localize(None)
            data[key]["date"] = data[key]["timestamp"].dt.date
            data[key]["hour"] = data[key]["timestamp"].dt.hour

    if not data["food"].empty:
        data["food"]["calories"] = data["food"].apply(
            lambda r: get_calories(r["food_type"], r.get("calories")),
            axis=1,
        )

    return data

# =========================================================================
# SIDEBAR
# =========================================================================

st.sidebar.title("🪞 ReflecEAT")
st.sidebar.write("Monitoring Dashboard")

data = load_data()

all_dates = []
for key in data:
    if not data[key].empty and 'date' in data[key].columns:
        all_dates.extend(data[key]['date'].unique())

if all_dates:
    min_date = min(all_dates)
    max_date = max(all_dates)
else:
    min_date = datetime.now().date() - timedelta(days=14)
    max_date = datetime.now().date()

st.sidebar.header("Date Selection")

date_mode = st.sidebar.radio("Select Mode", ["Single Day", "Date Range", "Quick Select"], index=2)

if date_mode == "Single Day":
    selected_date = st.sidebar.date_input("Choose a day", value=max_date, min_value=min_date, max_value=max_date)
    start_date, end_date = selected_date, selected_date
elif date_mode == "Date Range":
    c1, c2 = st.sidebar.columns(2)
    start_date = c1.date_input("From", value=min_date, min_value=min_date, max_value=max_date)
    end_date = c2.date_input("To", value=max_date, min_value=min_date, max_value=max_date)
else:
    opt = st.sidebar.selectbox("Quick Select", ["All Data", "Last 7 Days", "Last 14 Days", "This Week"])
    if opt == "All Data": start_date, end_date = min_date, max_date
    elif opt == "Last 7 Days": start_date, end_date = max_date - timedelta(6), max_date
    elif opt == "Last 14 Days": start_date, end_date = max_date - timedelta(13), max_date
    else: start_date, end_date = max_date - timedelta(max_date.weekday()), max_date

st.sidebar.markdown("---")
st.sidebar.info(f"Showing: {start_date} → {end_date}\n\nDays: {(end_date - start_date).days + 1}")

# =========================================================================
# MAIN CONTENT
# =========================================================================

st.title("ReflecEAT")
st.caption("Investigating SAD-Induced Emotional Eating")

if all(data[k].empty for k in data):
    st.warning("No data found.")
    st.stop()

def filter_df(df, start, end):
    if df.empty: return df
    return df[(df['date'] >= start) & (df['date'] <= end)]

emotion_df = filter_df(data['emotion'], start_date, end_date)
food_df = filter_df(data['food'], start_date, end_date)
light_df = filter_df(data['light'], start_date, end_date)
weather_df = filter_df(data['weather'], start_date, end_date)

# =========================================================================
# OVERVIEW (V8 Layout)
# =========================================================================

st.markdown("### Overview")
c1, c2, c3, c4, c5, c6 = st.columns([1, 2, 1, 2, 1, 1])

with c1: st.metric("Total Snacks", len(food_df))

with c2:
    if not food_df.empty:
        h = len(food_df[food_df['health_category'] == 'Healthy'])
        u = len(food_df[food_df['health_category'] == 'Unhealthy'])
        fig = create_donut(['Healthy', 'Unhealthy'], [h, u], [HEALTH_COLORS['Healthy'], HEALTH_COLORS['Unhealthy']], f"{(h/len(food_df)*100):.0f}%", title="Healthy %")
        st.plotly_chart(fig, use_container_width=True)
    else: st.metric("Healthy %", "N/A")

with c3: st.metric("Total Cal", f"{food_df['calories'].sum():.0f}" if not food_df.empty else "N/A")

with c4:
    if not emotion_df.empty:
        counts = emotion_df['emotion_class'].value_counts()
        labels = [l for l in ['positive', 'neutral', 'negative_low', 'negative_high'] if l in counts.index]
        values = [counts[l] for l in labels]
        colors = [EMOTION_COLORS[l] for l in labels]
        neg_pct = (counts.get('negative_low', 0) + counts.get('negative_high', 0)) / len(emotion_df) * 100
        fig = create_donut(labels, values, colors, f"{neg_pct:.0f}%<br>Neg", title="Emotion Breakdown")
        st.plotly_chart(fig, use_container_width=True)
    else: st.metric("Negative %", "N/A")

with c5: st.metric("Avg Lux", f"{light_df['lux_value'].mean():.0f}" if not light_df.empty else "N/A")
with c6: st.metric("Avg Daylight", f"{weather_df['daylight_duration'].mean():.1f}h" if not weather_df.empty else "N/A")

# =========================================================================
# VISUALIZATION LOGIC
# =========================================================================

st.markdown("---")
is_single_day = (start_date == end_date)

# Data Aggregations for Multi-Day (Reused in multiple sections)
daily_emotion = None
daily_food = None
daily_weather = None

if not is_single_day:
    if not emotion_df.empty:
        daily_emotion = emotion_df.groupby('date').agg(
            total=('emotion_class', 'count'),
            neg_l=('emotion_class', lambda x: (x=='negative_low').sum()),
            neg_h=('emotion_class', lambda x: (x=='negative_high').sum()),
            pos=('emotion_class', lambda x: (x=='positive').sum()),
            neu=('emotion_class', lambda x: (x=='neutral').sum())
        ).reset_index()
        daily_emotion['neg_pct'] = (daily_emotion['neg_l'] + daily_emotion['neg_h']) / daily_emotion['total'] * 100
        daily_emotion['date'] = pd.to_datetime(daily_emotion['date'])
        daily_emotion['dominant'] = daily_emotion[['pos', 'neu', 'neg_l', 'neg_h']].idxmax(axis=1).map(
            {'pos': 'positive', 'neu': 'neutral', 'neg_l': 'negative_low', 'neg_h': 'negative_high'}
        )

    if not food_df.empty:
        daily_food = pd.DataFrame({'date': pd.to_datetime(food_df['date'].unique())}).sort_values('date')
        h_cal = food_df[food_df['health_category']=='Healthy'].groupby('date')['calories'].sum()
        u_cal = food_df[food_df['health_category']=='Unhealthy'].groupby('date')['calories'].sum()
        daily_food['h_cal'] = daily_food['date'].dt.date.map(h_cal).fillna(0)
        daily_food['u_cal'] = daily_food['date'].dt.date.map(u_cal).fillna(0)

    if not weather_df.empty:
        daily_weather = weather_df.groupby('date').agg(dl=('daylight_duration', 'mean')).reset_index()
        daily_weather['date'] = pd.to_datetime(daily_weather['date'])


if is_single_day:
    # =====================================================================
    # SINGLE DAY VIEW
    # =====================================================================
    st.header(f"Daily Detail View - {start_date}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if not emotion_df.empty:
            counts = emotion_df['emotion_class'].value_counts()
            dom, pct = counts.index[0], counts.iloc[0]/len(emotion_df)*100
            st.markdown(f"""
            <div style="text-align:center; padding:15px; background:linear-gradient(135deg, {EMOTION_COLORS.get(dom,'#999')}22, {EMOTION_COLORS.get(dom,'#999')}44); border-radius:10px;">
                <div style="font-size:2.5rem;">{EMOTION_EMOJI.get(dom,'😐')}</div>
                <div style="font-weight:bold; color:{EMOTION_COLORS.get(dom,'#999')};">{dom.replace('_',' ').title()}</div>
                <div style="font-size:0.8rem; color:#666;">{pct:.0f}% of time</div>
            </div>
            """, unsafe_allow_html=True)
    with col2:
        if not weather_df.empty:
            w = weather_df.iloc[0]
            st.markdown(f"""
            <div style="text-align:center; padding:15px; background:linear-gradient(135deg, #f39c1222, #f39c1244); border-radius:10px;">
                <div style="font-size:2.5rem;">{WEATHER_EMOJI.get(w.get('weather_condition'),'☀️')}</div>
                <div style="font-weight:bold; color:#f39c12;">{w.get('daylight_duration',0):.1f}h Daylight</div>
                <div style="font-size:0.8rem; color:#666;">{w.get('weather_condition','')}</div>
            </div>
            """, unsafe_allow_html=True)
    with col3:
        if not food_df.empty:
            h = len(food_df[food_df['health_category']=='Healthy'])
            u = len(food_df[food_df['health_category']=='Unhealthy'])
            fig = create_donut(['Healthy', 'Unhealthy'], [h, u], [HEALTH_COLORS['Healthy'], HEALTH_COLORS['Unhealthy']], f"{len(food_df)}<br>Snacks", title="Snack Breakdown")
            st.plotly_chart(fig, use_container_width=True)
    with col4:
        if not food_df.empty:
            hc = food_df[food_df['health_category']=='Healthy']['calories'].sum()
            uc = food_df[food_df['health_category']=='Unhealthy']['calories'].sum()
            fig = create_donut(['Healthy', 'Unhealthy'], [hc, uc], [HEALTH_COLORS['Healthy'], HEALTH_COLORS['Unhealthy']], f"{hc+uc:.0f}<br>Kcal", title="Calorie Intake")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Emotion Flow - Smoothed with 30-min rolling window, showing gaps
    st.subheader("Emotion Flow Throughout the Day")
    if not emotion_df.empty:
        # Sort by timestamp
        emo_sorted = emotion_df.sort_values('timestamp').copy()
        
        # Create decimal hour for x-axis
        emo_sorted['hour_decimal'] = emo_sorted['timestamp'].dt.hour + emo_sorted['timestamp'].dt.minute / 60
        
        # Mark negative emotions
        emo_sorted['is_negative'] = emo_sorted['emotion_class'].isin(['negative_low', 'negative_high']).astype(int)
        
        # Calculate rolling 30-min window (approximately 6 readings at 5-min intervals)
        # Use time-based rolling for accuracy
        emo_sorted = emo_sorted.set_index('timestamp')
        emo_sorted['neg_pct_smooth'] = emo_sorted['is_negative'].rolling('30min', min_periods=1).mean() * 100
        emo_sorted = emo_sorted.reset_index()
        
        # Find gaps (where time between consecutive readings > 15 min)
        emo_sorted['time_diff'] = emo_sorted['timestamp'].diff().dt.total_seconds() / 60
        gaps = emo_sorted[emo_sorted['time_diff'] > 15].copy()
        
        # Create figure
        fig = go.Figure()
        
        # Add smoothed line
        fig.add_trace(go.Scatter(
            x=emo_sorted['hour_decimal'], 
            y=emo_sorted['neg_pct_smooth'], 
            mode='lines',
            name='Negative % (30min avg)', 
            line=dict(color='#e74c3c', width=2.5),
            fill='tozeroy', 
            fillcolor='rgba(231, 76, 60, 0.15)',
            hovertemplate='%{y:.0f}%<extra></extra>'
        ))
        
        # Add colored zones
        fig.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.08, line_width=0)
        fig.add_hrect(y0=30, y1=60, fillcolor="yellow", opacity=0.08, line_width=0)
        fig.add_hrect(y0=60, y1=100, fillcolor="red", opacity=0.08, line_width=0)
        
        # Add evening marker
        fig.add_vline(x=17, line_dash="dash", line_color="orange", 
                      annotation_text="Evening", annotation_position="top")
        
        # Add gap annotations (gaps > 20 min)
        gap_info = []
        for idx, row in gaps.iterrows():
            if row['time_diff'] >= 20:
                gap_end = row['hour_decimal']
                gap_start = gap_end - row['time_diff'] / 60
                gap_duration = int(row['time_diff'])
                
                # Add shaded region
                fig.add_vrect(x0=gap_start, x1=gap_end, fillcolor="gray", opacity=0.2, line_width=0)
                
                # Add annotation
                mid_point = (gap_start + gap_end) / 2
                fig.add_annotation(
                    x=mid_point, y=85,
                    text=f"📵 {gap_duration}min",
                    showarrow=False,
                    font=dict(size=9, color='#666'),
                    bgcolor='rgba(255,255,255,0.8)',
                    borderpad=2
                )
                
                start_h, start_m = int(gap_start), int((gap_start % 1) * 60)
                end_h, end_m = int(gap_end), int((gap_end % 1) * 60)
                gap_info.append(f"{start_h}:{start_m:02d}-{end_h}:{end_m:02d}")
        
        fig.update_layout(
            height=320, 
            xaxis_title="Time of Day", 
            yaxis_title="Negative Emotion %",
            yaxis=dict(range=[0, 100], fixedrange=True),
            xaxis=dict(
                tickmode='array',
                tickvals=[6, 8, 10, 12, 14, 16, 18, 20, 22, 24],
                ticktext=['6:00', '8:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00', '22:00', '24:00'],
                range=[6, 24]
            ),
            hovermode='x unified',
            margin=dict(t=30)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show gap summary
        if gap_info:
            st.caption(f"📵 Data gaps: {', '.join(gap_info)} (commute/meetings)")

    # Food Timeline
    st.subheader("Food Events Timeline")
    if not food_df.empty:
        food_sorted = food_df.sort_values('timestamp')
        fig = go.Figure()
        for _, event in food_sorted.iterrows():
            hour = event['timestamp'].hour + event['timestamp'].minute / 60
            is_healthy = event['health_category'] == 'Healthy'
            color = HEALTH_COLORS['Healthy'] if is_healthy else HEALTH_COLORS['Unhealthy']
            symbol = 'diamond' if is_healthy else 'x'
            fig.add_trace(go.Scatter(x=[hour], y=[1 if is_healthy else 0], mode='markers+text', marker=dict(size=18, color=color, symbol=symbol), text=[event['food_type'].title()], textposition='top center'))
        fig.update_layout(height=300, xaxis_title="Hour of Day", xaxis=dict(tickmode='linear', tick0=6, dtick=2, range=[6, 24]), yaxis=dict(tickvals=[0, 1], ticktext=['Unhealthy', 'Healthy'], range=[-0.5, 1.5], fixedrange=True), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("See Food Log Details"):
            st.dataframe(food_sorted[['timestamp', 'food_type', 'calories', 'health_category', 'emotion_before']], use_container_width=True)

    # Indoor Light
    st.subheader("Environmental Light Throughout the Day")
    if not light_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=light_df['timestamp'], y=light_df['lux_value'], mode='lines', line=dict(color='#f1c40f', width=2), fill='tozeroy', fillcolor='rgba(241, 196, 15, 0.3)'))
        fig.add_hline(y=200, line_dash="dash", line_color="red")
        fig.update_layout(height=250, xaxis_title="Time", yaxis_title="Lux")
        st.plotly_chart(fig, use_container_width=True)

else:
    # =====================================================================
    # MULTI-DAY VIEW (Timeline + Analytics)
    # =====================================================================
    st.header("Comprehensive Timeline")
    
    # 1. TIMELINE
    fig = make_subplots(rows=5, cols=1, shared_xaxes=False, vertical_spacing=0.08, subplot_titles=("Negative Emotion %", "Dominant Emotion", "Daylight Hours", "Calories", "Environment Light"))
    
    if daily_emotion is not None:
        fig.add_trace(go.Scatter(x=daily_emotion['date'], y=daily_emotion['neg_pct'], mode='lines+markers', name='Neg %', line=dict(color='#e74c3c')), row=1, col=1)
        
        colors = [EMOTION_COLORS.get(x, '#999') for x in daily_emotion['dominant']]
        emojis = [EMOTION_EMOJI.get(x, '') for x in daily_emotion['dominant']]
        fig.add_trace(go.Bar(x=daily_emotion['date'], y=[1]*len(daily_emotion), marker_color=colors, text=emojis, textposition='auto', showlegend=False), row=2, col=1)

    if daily_weather is not None:
        fig.add_trace(go.Scatter(x=daily_weather['date'], y=daily_weather['dl'], mode='lines+markers', name='Daylight', line=dict(color='#f39c12'), fill='tozeroy', fillcolor='rgba(243, 156, 18, 0.2)'), row=3, col=1)

    if daily_food is not None:
        fig.add_trace(go.Bar(x=daily_food['date'], y=daily_food['h_cal'], name='Healthy', marker_color=HEALTH_COLORS['Healthy']), row=4, col=1)
        fig.add_trace(go.Bar(x=daily_food['date'], y=daily_food['u_cal'], name='Unhealthy', marker_color=HEALTH_COLORS['Unhealthy']), row=4, col=1)

    if not light_df.empty:
        lh = light_df.groupby([light_df['date'], light_df['hour']]).agg(avg=('lux_value','mean')).reset_index()
        lh['dt'] = pd.to_datetime(lh['date'].astype(str)) + pd.to_timedelta(lh['hour'], unit='h')
        fig.add_trace(go.Scatter(x=lh['dt'], y=lh['avg'], name='Lux', line=dict(color='#f1c40f')), row=5, col=1)

    fig.update_xaxes(showticklabels=True)
    fig.update_layout(height=1000, barmode='stack', dragmode='pan')
    st.plotly_chart(fig, use_container_width=True)

    # 2. EVENING ANALYSIS
    st.markdown("---")
    st.header("Evening Pattern Analysis")
    c1, c2 = st.columns(2)
    with c1:
        if not emotion_df.empty:
            df = emotion_df.copy()
            df['Period'] = df['hour'].apply(lambda h: 'Evening' if h>=17 or h<6 else 'Daytime')
            fig = px.histogram(df, x='Period', color='emotion_class', barmode='group', color_discrete_map=EMOTION_COLORS)
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if not food_df.empty:
            df = food_df.copy()
            df['Period'] = df['hour'].apply(lambda h: 'Evening' if h>=17 or h<6 else 'Daytime')
            fig = px.histogram(df, x='Period', color='health_category', barmode='group', color_discrete_map=HEALTH_COLORS)
            st.plotly_chart(fig, use_container_width=True)

    # 3. CORRELATION ANALYSIS (RESTORED FROM V5)
    st.markdown("---")
    st.header("Correlation Analysis & Findings")
    
    # Prepare data for analysis
    corr_data = pd.DataFrame()
    if daily_emotion is not None:
        corr_data = daily_emotion[['date', 'neg_pct']].rename(columns={'neg_pct': 'Neg Emotion %'})
        if daily_weather is not None:
            corr_data = corr_data.merge(daily_weather.rename(columns={'dl': 'Daylight Hrs'}), on='date', how='outer')
        if not light_df.empty:
            ld = light_df.groupby('date')['lux_value'].mean().reset_index().rename(columns={'lux_value': 'Env Lux'})
            ld['date'] = pd.to_datetime(ld['date'])
            corr_data = corr_data.merge(ld, on='date', how='outer')
        if daily_food is not None:
            # Calculate unhealthy % by COUNT (not calories) - more meaningful for behavior analysis
            daily_unhealthy = food_df.groupby('date').apply(
                lambda x: (x['health_category'] == 'Unhealthy').mean() * 100,
                include_groups=False
            ).reset_index()
            daily_unhealthy.columns = ['date', 'Unhealthy %']
            daily_unhealthy['date'] = pd.to_datetime(daily_unhealthy['date'])
            corr_data = corr_data.merge(daily_unhealthy, on='date', how='outer')

    t1, t2, t3, t4 = st.tabs(["Heatmap", "Scatter Matrix", "Food-Emotion", "Insights"])

    with t1:
        if not corr_data.empty and len(corr_data) > 2:
            corr_matrix = corr_data.select_dtypes(include=[np.number]).corr()
            fig = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("Need more data for correlation.")

    with t2:
        if not corr_data.empty and len(corr_data) > 2:
            fig = px.scatter_matrix(corr_data.select_dtypes(include=[np.number]))
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("Need more data.")

    with t3:
        if not food_df.empty and 'emotion_before' in food_df.columns:
            m = pd.crosstab(food_df['food_type'], food_df['emotion_before'], normalize='index') * 100
            fig = px.imshow(m, aspect='auto', color_continuous_scale='RdYlGn_r', text_auto='.0f')
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Heatmap showing which foods are eaten during which emotions (row %)")

    with t4: # Key Insights & Hypothesis Testing
        st.subheader("Key Hypothesis Tests")
        if HAS_SCIPY and not corr_data.empty and len(corr_data) > 3:
            res = []
            def test_hyp(c1, c2, desc, sign):
                tmp = corr_data[[c1, c2]].dropna()
                if len(tmp) > 2:
                    r, p = stats.pearsonr(tmp[c1], tmp[c2])
                    res.append({'Hypothesis': desc, 'r': f"{r:.2f}", 'p-value': f"{p:.3f}", 'Support': '✅' if (p<0.05 and (r<0 if sign=='-' else r>0)) else '❌'})
            
            if 'Daylight Hrs' in corr_data: test_hyp('Daylight Hrs', 'Neg Emotion %', 'Less Daylight -> More Neg Emotion', '-')
            if 'Unhealthy %' in corr_data: test_hyp('Neg Emotion %', 'Unhealthy %', 'Neg Emotion -> More Unhealthy', '+')
            if 'Env Lux' in corr_data: test_hyp('Env Lux', 'Neg Emotion %', 'Low Light -> More Neg Emotion', '-')
            
            st.dataframe(pd.DataFrame(res), use_container_width=True)
        else:
            st.info("Insufficient data or Scipy not installed.")
        
        st.markdown("""
        **Key Insights Interpretation:**
        * **Daylight & Mood:** If correlation is negative, it suggests SAD patterns (less light = worse mood).
        * **Emotional Eating:** A positive correlation between Negative Emotion and Unhealthy Food % indicates emotional eating.
        * **Light Threshold:** Look at the Heatmap; darker red squares indicate strong relationships to investigate further.
        """)

# =========================================================================
# FOOTER
# =========================================================================
st.markdown("---")
st.caption("ReflecEAT - SmartSnack Monitor | Winnie Zhihan Wang | DE4-SIOT Final Project")