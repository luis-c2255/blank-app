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
            line=dict(color='darkgreen', width=2)
        ),
        row=2, col=1
    )
    fig2.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Volume_MA_30'],
            name="Vol MA 30",
            line=dict(color='yellow', width=2.5, dash='dash')
        ),
        row=2, col=1
    )
    fig2.update_layout(
        title_text="Netflix Stock Price and Trading Volume",
        showlegend=True
    )
    fig2 = apply_chart_theme(fig2)
    st.plotly_chart(fig2, width="stretch", height=600)

st.markdown("---")
st.markdown(
    Components.section_header("Volatility Analysis", "📇"),
    unsafe_allow_html=True
)

with st.container():
    fig3 = px.line(
    df,
    x='Date',
    y='Volatility_30',
    title="Netflix Stock Volatility (30-Day Rolling Std)")

    fig3.update_traces(line_color=Colors.CORAL_RED, line_width=3)
    fig3 = apply_chart_theme(fig3)
    st.plotly_chart(fig3, width="stretch", height=350)

st.markdown("---")
st.markdown(
    Components.section_header("Distribution of Daily Return", "🔙"),
    unsafe_allow_html=True
)
with st.container():
    fig4 = px.histogram(
        df,
        x='Daily_Return',
        nbins=50,
        title="Distribution of Daily Returns"
    )
    fig4.update_traces(
        marker_line_color='black',
        marker_line_width=3,
        opacity=0.7
    )
    mean_val = df['Daily_Return'].mean()

    fig4.add_vline(
        x=mean_val,
        line_dash='dash',
        line_color='green',
        annotation_text=f"Mean: {mean_val:.2f}%",
        annotation_position="top right"
    )
    fig4.update_layout(
        xaxis_title='Daily Return (%)',
        yaxis_title='Frequency',
        showlegend=True,
        height=500
    )
    fig4 = apply_chart_theme(fig4)
    st.plotly_chart(fig4, width="stretch")


st.markdown("---")
st.markdown(
    Components.section_header("Price Statistics", "💰"),
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        Components.metric_card(
        title="Starting Price",
        value=f"${df['Close'].iloc[0]:.2f}",
        delta="Start Price",
        card_type="success"
    ), unsafe_allow_html=True)

with col2:
    st.markdown(
        Components.metric_card(
        title="Ending Price",
        value=f"${df['Close'].iloc[-1]:.2f}",
        delta="End Price",
        card_type="warning"
    ), unsafe_allow_html=True)

with col3:
    st.markdown(
        Components.metric_card(
            title="Average Close",
            value=f"${df['Close'].mean():.2f}",
            delta="Close Price",
            card_type="info"
        ), unsafe_allow_html=True
    )

st.markdown("---")
st.markdown(
    Components.section_header("Correlation Heatmap", "🔥"),
    unsafe_allow_html=True
)

with st.container():
    corr_data = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return', 'Volatility_30']].corr()

    fig5 = px.imshow(
        corr_data,
        text_auto='.2f', 
        color_continuous_scale='viridis', 
        aspect="auto",
        title='Feature Correlation Heatmap',
        labels=dict(color="Correlation") 
    )
    fig5.update_xaxes(side='bottom')

    fig5 = apply_chart_theme(fig5)
    st.plotly_chart(fig5, width="stretch")

st.markdown("---")
st.markdown(
    Components.section_header("Advanced Analysis & Insights", "✨"),
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)
def style_financial_table(df):
    return df.style\
        .set_table_styles([
            {'selector': 'th', 'props': [
                ('background-color', '#E50914'),
                ('color', 'white'),
                ('font-family', 'Arial'),
                ('text-align', 'center')
            ]}
        ])\
        .format({'Daily_Return': '{:.2f}%', 'Close': '${:.2f}', 'Volume': '{:,}'})

with col1:
    st.markdown("Top 10 Days with Highest Positive Returns")
    top_volatile = df.nlargest(10, 'Daily_Return')[['Date', 'Close', 'Daily_Return', 'Volume']]
    top_volatile = top_volatile.sort_values('Date', ascending=True)

    st.dataframe(style_financial_table(top_volatile.head(10)), width='content', hide_index=True)



with col2:
    st.markdown("Top 10 Days with Highest Negative Returns")
    bottom_volatile = df.nsmallest(10, 'Daily_Return')[['Date', 'Close', 'Daily_Return', 'Volume']]
    bottom_volatile = bottom_volatile.sort_values('Date', ascending=False)

    st.dataframe(style_financial_table(bottom_volatile.head(10)), width='content', hide_index=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("Days with Volume Spikes")
    volume_threshold = df['Volume'].mean() + 2 * df['Volume'].std()
    high_volume_days = df[df['Volume'] > volume_threshold][['Date', 'Close', 'Volume', 'Daily_Return']]

    st.dataframe(style_financial_table(high_volume_days), width='content', hide_index=True)

with col2:
    st.markdown("Monthly Performance Summary")
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    monthly_performance = df.groupby(['Year', 'Month']).agg({
        'Close': ['first', 'last', 'min', 'max'],
        'Volume': 'sum',
        'Daily_Return': 'sum'
        }).reset_index()

    monthly_performance.columns=['Year', 'Month', 'Open_Price', 'Close_Price', 'Low', 'High', 'Total_Volume', 'Monthly_Return']

    monthly_performance['Monthly_Return_Pct'] = ((monthly_performance['Close_Price']-monthly_performance['Open_Price'])/monthly_performance['Open_Price'])*100
    st.dataframe(style_financial_table(monthly_performance), width='content', hide_index=True)

st.markdown(
    Components.section_header("Volatility Analysis", "📇"),
    unsafe_allow_html=True
)

st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        Components.metric_card(
        title="Overall Return",
        value=f"{((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100):.2f}%",
        delta="Return",
        card_type="info"
    ), unsafe_allow_html=True)

with col2:
    st.markdown(
        Components.metric_card(
        title="Highest Price",
        value=f"${df['High'].max():.2f}",
        delta=f"{df.loc[df['High'].idxmax(), 'Date'].strftime('%Y-%m-%d')}",
        card_type="error"
    ), unsafe_allow_html=True)

with col3:
    st.markdown(
        Components.metric_card(
            title="Lowest Price",
            value=f"${df['Low'].min():.2f}",
            delta=f"{df.loc[df['Low'].idxmin(), 'Date'].strftime('%Y-%m-%d')}",
            card_type="success"
        ), unsafe_allow_html=True
    )

st.markdown("---")
st.markdown(
    Components.section_header("Returns & Volatility", "📇"),
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        Components.metric_card(
        title="Average Daily Returns",
        value=f"{df['Daily_Return'].mean():.3f}%",
        delta="Average",
        card_type="info"
    ), unsafe_allow_html=True)

with col2:
    st.markdown(
        Components.metric_card(
        title="Median Daily Return",
        value=f"{df['Daily_Return'].median():.3f}%",
        delta="Median",
        card_type="info"
    ), unsafe_allow_html=True)

with col3:
    st.markdown(
        Components.metric_card(
            title="Daily Return Std Dev",
            value=f"{df['Daily_Return'].std():.3f}%",
            delta="Standard Deviation",
            card_type="warning"
        ), unsafe_allow_html=True
    )
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        Components.metric_card(
        title="Best Day",
        value=f"+{df['Daily_Return'].max():.2f}%",
        delta=f"{df.loc[df['Daily_Return'].idxmax(), 'Date'].strftime('%Y-%m-%d')}",
        card_type="success"
    ), unsafe_allow_html=True)

with col2:
    st.markdown(
        Components.metric_card(
        title="Worst Day",
        value=f"{df['Daily_Return'].min():.2f}%",
        delta=f"{df.loc[df['Daily_Return'].idxmin(), 'Date'].strftime('%Y-%m-%d')}",
        card_type="error"
    ), unsafe_allow_html=True)

with col3:
    st.markdown(
        Components.metric_card(
            title="Average 30-Day Volatility",
            value=f"{df['Volatility_30'].mean():.3f}%",
            delta="Average",
            card_type="info"
        ), unsafe_allow_html=True
    )

st.markdown("---")
st.markdown(
    Components.section_header("Volume Statistics", "📊"),
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        Components.metric_card(
        title="Average Daily Volume",
        value=f"{df['Volume'].mean():,.0f}",
        delta="Average",
        card_type="info"
    ), unsafe_allow_html=True)

with col2:
    st.markdown(
        Components.metric_card(
        title="Highest Volume",
        value=f"{df['Volume'].max():,.0f}",
        delta=f"{df.loc[df['Volume'].idxmax(), 'Date'].strftime('%Y-%m-%d')}",
        card_type="success"
    ), unsafe_allow_html=True)

with col3:
    st.markdown(
        Components.metric_card(
            title="Lowest Volume",
            value=f"{df['Volume'].min():,.0f}",
            delta=f"{df.loc[df['Volume'].idxmin(), 'Date'].strftime('%Y-%m-%d')}",
            card_type="error"
        ), unsafe_allow_html=True
    )


st.markdown("---")
st.markdown(
    Components.section_header("Technical Indicators", "🎯"),
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        Components.metric_card(
        title="Current RSI",
        value=f"{df['RSI'].iloc[-1]:.2f}",
        delta="RSI",
        card_type="info"
    ), unsafe_allow_html=True)

with col2:
    st.markdown(
        Components.metric_card(
        title="7-Day MA",
        value=f"${df['MA_7'].iloc[-1]:.2f}",
        delta="7-Day",
        card_type="info"
    ), unsafe_allow_html=True)

with col3:
    st.markdown(
        Components.metric_card(
            title="30-Day MA",
            value=f"${df['MA_30'].iloc[-1]:.2f}",
            delta="30-Day",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col4:
    st.markdown(
        Components.metric_card(
            title="90-Day MA",
            value=f"${df['MA_90'].iloc[-1]:.2f}",
            delta="90-Day",
            card_type="info"
        ), unsafe_allow_html=True
    )