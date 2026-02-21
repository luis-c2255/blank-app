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
    Components.section_header("Volume vs. Daily Correlation", "💱"),
    unsafe_allow_html=True
)
with st.container():
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df['Volume'],
        y=df['Daily_Return'],
        mode='markers',
        marker=dict(
            size=10,
            color='steelblue',
            opacity=0.3
        ),
        name='Data'
    ))
    fig3.add_hline(y=0, line=dict(color='red', dash='dash', width=1))
    fig3.update_layout(
        width=1000,
        height=600,
        xaxis_title=dict(text='Trading Volume', font=dict(size=12)),
        yaxis_title=dict(text='Daily_Return (%)', font=dict(size=12)),
        title=dict(
            text='Volume vs. Daily Return Correlation', 
            font=dict(size=16, family='Arial, sans-serif'), 
            x=0.5, 
            xanchor='center'
        ),
        xaxis=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.3)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.3)'),
        plot_bgcolor='white',
        showlegend=False
    )
    st.plotly_chart(fig3, width="stretch")

st.markdown("---")
st.markdown(
    Components.section_header("Key Volume Metrics", "💹"),
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
        ), unsafe_allow_html=True
    )

with col2:
    st.markdown(
        Components.metric_card(
            title="Highest Volume",
            value=f"{df['Volume'].max():,.0f}",
            delta=f"{df.loc[df['Volume'].idxmax(), 'Date'].strftime('%Y-%m-%d')}",
            card_type="success"
        ), unsafe_allow_html=True
    )

with col3:
    st.markdown(
        Components.metric_card(
            title="Lowest Volume",
            value=f"{df['Volume'].min():,.0f}",
            delta=f"{df.loc[df['Volume'].idxmin(), 'Date'].strftime('%Y-%m-%d')}",
            card_type="warning"
        ), unsafe_allow_html=True
    )

st.markdown("---")
st.markdown(
    Components.section_header("RSI (Relative Strength Index)", "📈"),
    unsafe_allow_html=True
)
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
df['RSI'] = calculate_rsi(df['Close'])

with st.container():
    # Plot RSI
    fig4 = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.67, 0.33],
        subplot_titles=('', '')
    )
    # Price chart
    fig4.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Close'],
            mode='lines',
            line=dict(color='#E50914', width=1.5),
            name='Close Price',
            showlegend=False
        ),
        row=1, col=1
    )
    # RSI chart
    fig4.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['RSI'],
            mode='lines',
            line=dict(color='purple', width=1.5),
            name='RSI'
        ),
        row=2, col=1
    )
    # Overbought line
    fig4.add_trace(
        go.Scatter(
            x=df['Date'],
            y=[70] * len(df),
            mode='lines',
            line=dict(color='red', width=1, dash='dash'),
            name='Overbought (70)'
        ),
        row=2, col=1    
    )
    # Oversold line
    fig4.add_trace(
        go.Scatter(
            x=df['Date'],
            y=[30] * len(df),
            mode='lines',
            line=dict(color='green', width=1, dash='dash'),
            name='Oversold (30)'
        ),
        row=2, col=1
    )
    # Fill between 30 and 70
    fig4.add_trace(
        go.Scatter(
            x=df['Date'].tolist() + df['Date'].tolist()[::-1],
            y=[70] * len(df) + [30] * len(df),
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.1)',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=2, col=1
    )
    # Update layout
    fig4.update_xaxes(title_text='Date', title_font=dict(size=12), row=2, col=1)
    fig4.update_yaxes(title_text='Close Price (USD)', title_font=dict(size=12), row=1, col=1)
    fig4.update_yaxes(title_text='RSI', title_font=dict(size=12), range=[0, 100], row=2, col=1)

    fig4.update_layout(
        title=dict(
            text='Netflix Stock Price with RSI Indicator',
            font=dict(size=16, family='Arial, sans-serif'),
            x=0.5,
            xanchor='center'
        ),
        height=800,
        width=1400,
        showlegend=True,
        legend=dict(x=0.01, y=0.35, xanchor='left', yanchor='top'),
        hovermode='x unified',
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.3)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.3)'),
        xaxis2=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.3)'),
        yaxis2=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.3)')    
    )
    st.plotly_chart(fig4, width="stretch")

col1, col2, = st.columns(2)

overbought = df[df['RSI'] > 70][['Date', 'Close', 'RSI']]
oversold = df[df['RSI'] < 30][['Date', 'Close', 'RSI']]

with col1:
    st.markdown(
        Components.insight_box(
            title=f"Overbought periods (RSI > 70): {len(overbought)} days",
            content=f"Date: 2018-03-06,\nClose: 325.22,\nRSI: 89.23",
            box_type="warning"
        ), unsafe_allow_html=True
    )

with col2:
    st.markdown(
        Components.insight_box(
            title=f"\nOversold periods (RSI < 30): {len(oversold)} days",
            content=f"Date: 2018-07-31,\nClose: 334.95,\nRSI: 13.60",
            box_type="success"
        ), unsafe_allow_html=True
    )

st.markdown("---")
st.markdown(
    Components.section_header("Volatility Analysis", "📇"),
    unsafe_allow_html=True
)
with st.container():
    fig5 = px.line(
    df,
    x='Date',
    y='Volatility_30',
    title="Netflix Stock Volatility (30-Day Rolling Std)")

    fig5.update_traces(line_color=Colors.CORAL_RED, line_width=3)
    fig5 = apply_chart_theme(fig5)
    st.plotly_chart(fig5, width="stretch", height=350)

st.markdown("---")
st.markdown(
    Components.section_header("Distribution of Daily Return", "🔙"),
    unsafe_allow_html=True
)
with st.container():
    fig6 = px.histogram(
        df,
        x='Daily_Return',
        nbins=50,
        title="Distribution of Daily Returns"
    )
    fig6.update_traces(
        marker_line_color='black',
        marker_line_width=3,
        opacity=0.7
    )
    mean_val = df['Daily_Return'].mean()

    fig6.add_vline(
        x=mean_val,
        line_dash='dash',
        line_color='green',
        annotation_text=f"Mean: {mean_val:.2f}%",
        annotation_position="top right"
    )
    fig6.update_layout(
        xaxis_title='Daily Return (%)',
        yaxis_title='Frequency',
        showlegend=True,
        height=500
    )
    fig6 = apply_chart_theme(fig6)
    st.plotly_chart(fig6, width="stretch")

df['Year'] = df['Date'].apply(lambda x: x.year)
df['Month'] = df['Date'].apply(lambda x: x.month)

monthly_performance = df.groupby(['Year', 'Month']).agg({
        'Close': ['first', 'last', 'min', 'max'],
        'Volume': 'sum',
        'Daily_Return': 'sum'
        }).reset_index()

monthly_performance.columns=['Year', 'Month', 'Open_Price', 'Close_Price', 'Low', 'High', 'Total_Volume', 'Monthly_Return']
monthly_performance['Monthly_Return_Pct'] = ((monthly_performance['Close_Price']-monthly_performance['Open_Price'])/monthly_performance['Open_Price'])*100

with st.container():
    colors = ['green' if x > 0 else 'red' for x in monthly_performance['Monthly_Return_Pct']]
    fig8 = go.Figure()
    fig8.add_trace(go.Bar(
        x=list(range(len(monthly_performance))),
        y=monthly_performance['Monthly_Return_Pct'],
        marker=dict(color=colors, opacity=0.7),
        showlegend=False
    ))
    fig8.add_hline(y=0, line=dict(color='black', width=0.8))

    fig8.update_layout(
        title=dict(text='Netflix Monthly Returns (%)', font=dict(size=16, family='Arial Black')),
        xaxis_title='Month Index',
        yaxis_title='Monthly Return (%)',
        width=1400,
        height=600,
        yaxis=dict(gridcolor='rgba(128, 128, 128, 0.3)', showgrid=True),
        xaxis=dict(showgrid=False)
    )
    st.plotly_chart(fig8, width="stretch")

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
    Components.section_header("Returns Distribution", "📉"),
    unsafe_allow_html=True
)
df['DayOfWeek'] = df['Date'].dt.day_name()
df['DayOfWeek_Num'] = df['Date'].dt.dayofweek

# Average return by day of week
dow_performance = df.groupby('DayOfWeek')['Daily_Return'].agg(['mean', 'median', 'std', 'count']).reset_index()
dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
dow_performance['DayOfWeek'] = pd.Categorical(dow_performance['DayOfWeek'], categories=dow_order, ordered=True)
dow_performance = dow_performance.sort_values('DayOfWeek')

col1, col2 = st.columns(2)

with col1:
    colors = ['green' if x > 0 else 'red' for x in dow_performance['mean']]

    fig_bar = go.Figure()

    fig_bar.add_trace(go.Bar(
        x=dow_performance['DayOfWeek'],
        y=dow_performance['mean'],
        marker=dict(
            color=colors,
            opacity=0.7,
            line=dict(color='black', width=1)
        )
    ))

    fig_bar.add_hline(y=0, line=dict(color='black', width=0.8))

    fig_bar.update_layout(
        title=dict(
            text='Average Daily Return by Day of Week',
            font=dict(size=16, family='Arial, sans-serif'),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Day of Week',
        yaxis_title='Average Return (%)',
        yaxis=dict(gridcolor='rgba(128, 128, 128, 0.3)'),
        xaxis=dict(showgrid=False),
        width=1000,
        height=600,
        showlegend=False
    )
    st.plotly_chart(fig_bar, width="stretch")

# Monthly seasonality
df['Month_Name'] = df['Date'].dt.month_name()
month_performance = df.groupby('Month_Name')['Daily_Return'].agg(['mean', 'median', 'std', 'count']).reset_index()

month_order = ['January', 'February', 'March', 'April', 'May', 'June',
'July', 'August', 'September', 'October', 'November', 'December']
month_performance['Month_Name'] = pd.Categorical(month_performance['Month_Name'], categories=month_order, ordered=True)
month_performance = month_performance.sort_values('Month_Name')

with col2:
    colors = ['green' if x > 0 else 'red' for x in month_performance['mean']]

    fig_bar2 = go.Figure()

    fig_bar2.add_trace(go.Bar(
        x=month_performance['Month_Name'],
        y=month_performance['mean'],
        marker=dict(
            color=colors,
            opacity=0.7,
            line=dict(color='black', width=1)
        )
    ))

    fig_bar2.add_hline(y=0, line=dict(color='black', width=0.8))

    fig_bar2.update_layout(
        title=dict(
            text='Average Daily Return by Month',
            font=dict(size=16, family='Arial, sans-serif')
        ),
        xaxis=dict(
            title='Month',
            tickangle=-45
        ),
        yaxis=dict(
            title='Average Return (%)',
            gridcolor='rgba(128, 128, 128, 0.3)'
        ),
        width=1200,
        height=600,
        showlegend=False
    )
    st.plotly_chart(fig_bar2, width="stretch")



st.markdown("---")
st.markdown(
    Components.section_header("Returns & Volatility Metrics", "📇"),
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
    Components.section_header("Monthly Returns", "↩"),
    unsafe_allow_html=True
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
st.markdown("---")
st.markdown(
    Components.section_header("Correlation Heatmap", "🔥"),
    unsafe_allow_html=True
)

with st.container():
    corr_data = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return', 'Volatility_30']].corr()

    fig7 = px.imshow(
        corr_data,
        text_auto='.2f', 
        color_continuous_scale='viridis', 
        aspect="auto",
        title='Feature Correlation Heatmap',
        labels=dict(color="Correlation") 
    )
    fig7.update_xaxes(side='bottom')

    fig7 = apply_chart_theme(fig7)
    st.plotly_chart(fig7, width="stretch")

st.markdown("---")
st.markdown(
    Components.section_header("Advanced Analysis & Insights", "✨"),
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)
def style_table(df,color_theme):
    html = (
        df.style
        .hide()
        .format({
            "Close": "${:.2f}",
            "Daily_Return": "{:.2%}",
            "Volume": "{:,.0f}"
        })
        .map(
            lambda v: f"color: {'green' if v > 0 else 'red'}; font-weight: bold;"
            if isinstance(v, float) else "",
            subset=["Daily_Return"]
        )
        .set_table_styles([
            {"selector": "table", "props": "width: 100%; border-collapse: collapse;"},
            {"selector": "th", "props": f"background-color: {color_theme}; color: white; padding: 8px; text-align: center; position: sticky; top: 0;"},
            {"selector": "td", "props": "padding: 8px; text-align: center; border-bottom: 1px solid #ddd;"},
            {"selector": "tr:hover", "props": "background-color: #f5f5f5;"},
        ])
        .to_html()
    )
    return f'<div style="height: 380px; overflow: auto; border: 1px solid #ccc; border-radius: 8px;">{html}</div>'


with col1:
    st.markdown("Top 10 Days with Highest Positive Returns")
    df['Date'] = df['Date'].dt.date
    top_volatile = df.nlargest(10, 'Daily_Return')[['Date', 'Close', 'Daily_Return', 'Volume']]
    top_volatile = top_volatile.sort_values('Date', ascending=True)

    st.markdown(style_table(top_volatile, color_theme="#2e7d32"), unsafe_allow_html=True)



with col2:
    st.markdown("Top 10 Days with Highest Negative Returns")
    bottom_volatile = df.nsmallest(10, 'Daily_Return')[['Date', 'Close', 'Daily_Return', 'Volume']]
    bottom_volatile = bottom_volatile.sort_values('Date', ascending=False)

    st.markdown(style_table(bottom_volatile, color_theme="#c62828"), unsafe_allow_html=True)

with col3:
    st.markdown("Days with Volume Spikes")
    volume_threshold = df['Volume'].mean() + 2 * df['Volume'].std()
    high_volume_days = df[df['Volume'] > volume_threshold][['Date', 'Close', 'Volume', 'Daily_Return']]

    st.markdown(style_table(high_volume_days, color_theme="#FFB84D"), unsafe_allow_html=True)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown("Monthly Performance Summary")
    df['Year'] = df['Date'].apply(lambda x: x.year)
    df['Month'] = df['Date'].apply(lambda x: x.month)

    monthly_performance = df.groupby(['Year', 'Month']).agg({
        'Close': ['first', 'last', 'min', 'max'],
        'Volume': 'sum',
        'Daily_Return': 'sum'
        }).reset_index()

    monthly_performance.columns=['Year', 'Month', 'Open_Price', 'Close_Price', 'Low', 'High', 'Total_Volume', 'Monthly_Return']
    monthly_performance['Monthly_Return_Pct'] = ((monthly_performance['Close_Price']-monthly_performance['Open_Price'])/monthly_performance['Open_Price'])*100

    html = (
        monthly_performance.style
        .hide()
        .format({
            "Open_Price": "${:.2f}",
            "Close_Price": "${:.2f}",
            "Low": "${:.2f}",
            "High": "${:.2f}",
            "Total_Volume": "{:,.0f}",
            "Monthly_Return": "{:.2%}",
            "Monthly_Return_Pct": "{:.2f}%",
        })
        .map(
            lambda v: "color: green; font-weight: bold;" if isinstance(v, float) and v > 0 else "color: red; font-weight: bold;" if isinstance(v, float) and v < 0 else "",
            subset=["Monthly_Return_Pct"]
        )
        .set_table_styles([
            {"selector": "table", "props": "width: 100%; border-collapse: collapse;"},
            {"selector": "th", "props": "background-color: #264653; color: white; padding: 8px; text-align: center; position: sticky; top: 0;"},
            {"selector": "td", "props": "padding: 8px; text-align: center; border-bottom: 1px solid #ddd;"},
            {"selector": "tr:hover", "props": "background-color: #f5f5f5;"},
        ])
        .to_html()
    )
    st.markdown(
        f'<div style="height: 400px; overflow: auto; border: 1px solid #ccc; border-radius: 8px;">{html}</div>',
        unsafe_allow_html=True
    )