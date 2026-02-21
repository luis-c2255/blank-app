import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.theme import Components, Colors, apply_chart_theme, init_page

init_page("Netflix Stock Analysis", "💹")

# Load custom CSS
try:
    with open('style.css') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
        st.warning("Custom CSS file not found. Using default styling.")
# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('netflix_analyzed.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df
df = load_data()

# Title
st.markdown(
    Components.page_header(
        "💹 Netflix Stock Analysis Dashboard"), unsafe_allow_html=True)

st.markdown("---")

st.markdown(
    Components.section_header('Key Metrics', '🎯'), unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        Components.metric_card(
        title="Start Analysis Period",
        value=f"{df['Date'].min().strftime('%Y-%m-%d')}",
        delta="Start",
        card_type="info"
    ), unsafe_allow_html=True)
with col2:
    st.markdown(
        Components.metric_card(
        title="Total Trading Days",
        value=f"{len(df)}",
        delta="Trading Days",
        card_type="info"
    ), unsafe_allow_html=True)
with col3:
    st.markdown(
        Components.metric_card(
            title="End Analisys Period",
            value=f"{df['Date'].max().strftime('%Y-%m-%d')}",
            delta="End",
            card_type="info"
        ), unsafe_allow_html=True
    )

st.markdown("---")
st.markdown(
    Components.section_header("Price Trend Visualization", "📈"),
    unsafe_allow_html=True
)

with st.container():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'], 
        y=df['Close'], 
        name='Close Price',
        line=dict(color='#E50914', width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['MA_30'], 
        name='30-Day MA',
        line=dict(color='orange', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=df['Date'], 
        y=df['MA_90'], 
        name='90-Day MA',
        line=dict(color='green', dash='dot')
    ))
                                
    fig.update_layout(
        title='Netflix Stock Price Over Time',
        xaxis_title='Date',
        yaxis_title='Price (USD)'
    )
    fig = apply_chart_theme(fig)
    st.plotly_chart(fig, width='stretch', height=500)

st.markdown("---")
st.markdown(
    Components.section_header("Volume Analysis", "📇"),
    unsafe_allow_html=True
)

with st.container():
    fig2 = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Price", "Volume")
    )
    
    fig2.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Close'],
            name="Close Price",
            line=dict(color="#E50914")
        ),
        row=1, col=1
    )
    fig2.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Volume'],
            name="Volume",
            line=dict(color='orange', width=2)
        ),
        row=2, col=1
    )
    fig2.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Volume_MA_30'],
            name="Vol MA 30",
            line=dict(color='blue', width=2, dash='dash')
        ),
        row=2, col=1
    )
    fig2.update_layout(
        title_text="Netflix Stock Price and Trading Volume",
        showlegend=True
    )
    fig2 = apply_chart_theme(fig2)
    st.plotly_chart(fig2, width="stretch", height=600)