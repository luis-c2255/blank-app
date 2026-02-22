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
    mode='lines',
    name='Close Price',
    line=dict(color='#E50914', width=1.5)
))

fig.add_trace(go.Scatter(
    x=df['Date'],
    y=df['MA_30'],
    mode='lines',
    name='30-Day MA',
    line=dict(color='orange', dash='dash')
))

fig.add_trace(go.Scatter(
    x=df['Date'],
    y=df['MA_90'],
    mode='lines',
    name='90-Day MA',
    line=dict(color='green', dash='dash')
))

fig.update_layout(
    title=dict(
        text='Netflix Stock Price Over Time',
        font=dict(size=16, family='Arial, sans-serif')
    ),
    xaxis_title='Date',
    yaxis_title='Price (USD)',
    width=1400,
    height=600,
    showlegend=True,
    xaxis=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.3)'),
    yaxis=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.3)')
)

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
    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=('', ''))

# Price
fig2.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', line=dict(color='#E50914', width=1.5), name='Close Price'), row=1, col=1)

# Volume
fig2.add_trace(go.Bar(x=df['Date'], y=df['Volume'], marker=dict(color='steelblue', opacity=0.6), name='Volume'), row=2, col=1)
fig2.add_trace(go.Scatter(x=df['Date'], y=df['Volume_MA_30'], mode='lines', line=dict(color='orange', width=2), name='30-Day MA'), row=2, col=1)

# Update layout
fig2.update_xaxes(title_text='Date', title_font=dict(size=12), row=2, col=1)
fig2.update_yaxes(title_text='Close Price (USD)', title_font=dict(size=12), row=1, col=1)
fig2.update_yaxes(title_text='Volume', title_font=dict(size=12), row=2, col=1)

fig2.update_layout(title=dict(text='Netflix Stock Price and Trading Volume', font=dict(size=16, family='Arial, sans-serif'), x=0.5, xanchor='center'), height=800, width=1400, showlegend=True, hovermode='x unified')

# Add grid
fig2.update_xaxes(showgrid=True, gridcolor='lightgray', gridwidth=0.5)
fig2.update_yaxes(showgrid=True, gridcolor='lightgray', gridwidth=0.5)
    
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
    name=''
))

fig3.add_hline(y=0, line_color='red', line_dash='dash', line_width=1)

fig3.update_layout(
    width=1000,
    height=600,
    xaxis_title=dict(text='Trading Volume', font=dict(size=12)),
    yaxis_title=dict(text='Daily Return (%)', font=dict(size=12)),
    title=dict(text='Volume vs. Daily Return Correlation', font=dict(size=16, family='Arial, sans-serif'), x=0.5, xanchor='center'),
    xaxis=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.3)'),
    yaxis=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.3)'),
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
    # Create subplots with 2 rows, shared x-axis
    fig4 = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.67, 0.33],
        subplot_titles=('', '')
    )

    # Price chart (top subplot)
    fig4.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Close'],
            mode='lines',
            line=dict(color='#E50914', width=1.5),
            name='Close Price'
        ),
        row=1, col=1
    )

    # RSI chart (bottom subplot)
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

    # Overbought line (70)
    fig4.add_trace(
        go.Scatter(
            x=df['Date'],
            y=[70] * len(df),
            mode='lines',
            line=dict(color='red', width=1, dash='dash'),
            name='Overbought (70)',
            showlegend=True
        ),
        row=2, col=1    
    )

    # Oversold line (30)
    fig4.add_trace(
        go.Scatter(
            x=df['Date'],
            y=[30] * len(df),
            mode='lines',
            line=dict(color='green', width=1, dash='dash'),
            name='Oversold (30)',
            showlegend=True
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
    fig4.update_layout(
        title=dict(
            text='Netflix Stock Price with RSI Indicator',
            font=dict(size=16, family='Arial, bold')
        ),
        height=800,
        width=1400,
        showlegend=True,
        hovermode='x unified'
    )

# Update y-axes
fig4.update_yaxes(title_text='Close Price (USD)', title_font=dict(size=12), row=1, col=1, showgrid=True, gridcolor='rgba(128, 128, 128, 0.3)')
fig4.update_yaxes(title_text='RSI', title_font=dict(size=12), range=[0, 100], row=2, col=1, showgrid=True, gridcolor='rgba(128, 128, 128, 0.3)')

# Update x-axis
fig4.update_xaxes(title_text='Date', title_font=dict(size=12), row=2, col=1, showgrid=True, gridcolor='rgba(128, 128, 128, 0.3)')
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
    fig5 = go.Figure()
fig5.add_trace(go.Scatter(
    x=df['Date'],
    y=df['Volatility_30'],
    mode='lines',
    line=dict(color='crimson', width=1.5),
    name='Volatility'
))
fig5.update_layout(
    title=dict(
        text='Netflix Stock Volatility (30-Day Rolling Std)',
        font=dict(size=16, family='Arial, sans-serif')
    ),
    xaxis_title='Date',
    yaxis_title='Volatility (%)',
    xaxis=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.3)'),
    yaxis=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.3)'),
    width=1400,
    height=500
)
st.plotly_chart(fig5, width="stretch", height=350)

st.markdown("---")
st.markdown(
    Components.section_header("Distribution of Daily Return", "🔙"),
    unsafe_allow_html=True
)
with st.container():
    data = df['Daily_Return'].dropna()
    mean_val = data.mean()

fig6 = go.Figure()

fig6.add_trace(go.Histogram(
    x=data,
    nbinsx=50,
    marker=dict(
        color='#E50914',
        opacity=0.7,
        line=dict(color='black', width=1)
    ),
    name='Daily Return'
))

fig6.add_vline(
    x=mean_val,
    line=dict(color='green', dash='dash', width=2),
    annotation_text=f'Mean: {mean_val:.2f}%',
    annotation_position='top'
)

fig6.update_layout(
    title=dict(
        text='Distribution of Daily Returns',
        font=dict(size=16, family='Arial, sans-serif'),
        x=0.5,
        xanchor='center'
    ),
    xaxis_title='Daily Return (%)',
    yaxis_title='Frequency',
    width=1000,
    height=500,
    showlegend=True,
    xaxis=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.3)'),
    yaxis=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.3)')
)

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
        title="Daily_Volatility_Pct",
        value=f"{df['Daily_Return'].std():.2f}",
        delta="PCT",
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
    Components.section_header("Monthly & Weekly Performance", "🔝"),
    unsafe_allow_html=True
)
data = {
	'Day_of_Week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
	'mean': [0.243, 0.243, 0.110, 0.174, -0.354],
	'median': [0.248, 0.088, 0.078, 0.229, -0.229],
	'std': [2.773, 2.303, 2.987, 2.419, 2.750],
	'count': [191, 207, 205, 204, 201]
}
df_week = pd.DataFrame(data)

data = {
	'Month': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
	'mean': [0.025, 0.322, 0.052, 0.258, 0.036, 0.377, -0.230, 0.212, -0.052, -0.032, 0.020, 0.021],
	'median': [-0.459, 0.010, -0.108, 0.181, 0.042, 0.451, -0.087, 0.222, 0.042, 0.084, 0.217, -0.026],
	'std': [4.370, 2.297, 3.459, 2.513, 1.585, 1.923, 2.519, 2.208, 2.274, 3.050, 2.301, 2.382],
	'count': [81, 77, 87, 84, 84, 85, 86, 88, 81, 89, 82, 84]
}
df_month = pd.DataFrame(data)

col1, col2 = st.columns(2)
def style_table(df,color_theme):
    html = (
        df.style
        .hide()
        .set_table_styles([
            {"selector": "table", "props": "width: 100%; display: block; table-layout: fixed; border-collapse: collapse;"},
            {"selector": "th", "props": f"background-color: {color_theme}; color: white; padding: 8px; text-align: center; position: sticky; top: 0; width: 25%;"},
            {"selector": "td", "props": "padding: 8px; text-align: center; border-bottom: 1px solid #ddd; width: 25%;"},
            {"selector": "tr:hover", "props": "background-color: #f5f5f5;"},
        ])
        .to_html()
    )
    return f'<div style="height: 380px; width: 100%; overflow: auto; border: 1px solid #ccc; border-radius: 8px;">{html}</div>'

with col1:
    st.markdown(style_table(df_week, color_theme="#FF9F1C"), unsafe_allow_html=True)

with col2:
    st.markdown(style_table(df_month, color_theme="#508CA4"), unsafe_allow_html=True)


st.markdown("---")
st.markdown(
    Components.section_header("Technical Indicators", "🎯"),
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        Components.metric_card(
        title="7-Day MA",
        value=f"${df['MA_7'].iloc[-1]:.2f}",
        delta="7-Day",
        card_type="info"
    ), unsafe_allow_html=True)

with col2:
    st.markdown(
        Components.metric_card(
            title="30-Day MA",
            value=f"${df['MA_30'].iloc[-1]:.2f}",
            delta="30-Day",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col3:
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
    # Select numeric columns
    corr_data = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return', 'Volatility_30']].corr()

fig7 = go.Figure(data=go.Heatmap(
    z=corr_data.values,
    x=corr_data.columns,
    y=corr_data.index,
    colorscale='RdBu_r',
    zmid=0,
    text=corr_data.values,
    texttemplate='%{text:.2f}',
    textfont={"size": 10},
    colorbar=dict(title='Correlation'),
    xgap=1,
    ygap=1
))

fig7.update_layout(
    title={
        'text': 'Feature Correlation Heatmap',
        'font': {'size': 16, 'family': 'Arial, sans-serif'},
        'x': 0.5,
        'xanchor': 'center'
    },
    width=1000,
    height=700,
    xaxis={'side': 'bottom'},
    yaxis={'autorange': 'reversed'}
)
st.plotly_chart(fig7, width="stretch")

st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.markdown(
        Components.insight_box(
            title="Current RSI",
            content=f"{df['RSI'].iloc[-1]:.2f}",
            box_type="info"
    ),
    unsafe_allow_html=True
)

with col2:
    rsi_status = "Overbought" if df['RSI'].iloc[-1] > 70 else "Oversold" if df['RSI'].iloc[-1] < 30 else "Neutral"
    st.markdown(
        Components.insight_box(
            title="RSI Status",
            content=f"{rsi_status}",
            box_type="info"
    ),
    unsafe_allow_html=True
)
st.markdown("---")
st.markdown(
    Components.section_header("Predictive Modeling", "🔮"),
    unsafe_allow_html=True
)
with st.container():
    st.markdown("Simple Moving Average Crossover Strategy")
    # Trading signal: Buy when MA_7 crosses above MA_30, Sell when it crosses below
    df['Signal'] = 0
    df.loc[df['MA_7'] > df['MA_30'], 'Signal'] = 1 # Buy signal
    df.loc[df['MA_7'] < df['MA_30'], 'Signal'] = -1 # Sell signal

    # Identify crossover points
    df['Position'] = df['Signal'].diff()
    fig8 = go.Figure()

fig8.add_trace(go.Scatter(
    x=df['Date'],
    y=df['Close'],
    mode='lines',
    name='Close Price',
    line=dict(color='purple', width=1.5)
))

fig8.add_trace(go.Scatter(
    x=df['Date'],
    y=df['MA_7'],
    mode='lines',
    name='7-Day MA',
    line=dict(color='blue', dash='dash')
))

fig8.add_trace(go.Scatter(
    x=df['Date'],
    y=df['MA_30'],
    mode='lines',
    name='30-Day MA',
    line=dict(color='orange', dash='dash')
))

fig8.update_layout(
    width=1400,
    height=600
)
st.plotly_chart(fig8, width="stretch")

st.markdown("---")

# Mark buy signals
buy_signals = df[df['Position'] == 2]

# Mark sell signals
sell_signals = df[df['Position'] == -2]

fig9 = go.Figure()

fig9.add_trace(go.Scatter(
    x=buy_signals['Date'],
    y=buy_signals['Close'],
    mode='markers',
    marker=dict(color='blue', symbol='triangle-up', size=14, line=dict(width=0)),
    name='Buy Signal'
))

fig9.add_trace(go.Scatter(
    x=sell_signals['Date'],
    y=sell_signals['Close'],
    mode='markers',
    marker=dict(color='orange', symbol='triangle-down', size=14, line=dict(width=0)),
    name='Sell Signal'
))

fig9.update_layout(
    title=dict(text='Netflix Stock - Moving Average Crossover Strategy', font=dict(size=16, family='Arial, sans-serif')),
    xaxis_title='Date',
    yaxis_title='Price (USD)',
    showlegend=True,
    xaxis=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.3)'),
    yaxis=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.3)'),
)
st.plotly_chart(fig9, width="stretch")

st.markdown("---")
st.markdown(
    Components.section_header("Time Series Forecasting with Prophet", "🎲"),
    unsafe_allow_html=True
)
from prophet import Prophet

col1, col2 = st.columns(2)

with col1:
    # Prepare data for Prophet (requires 'ds' and 'y' columns)
        prophet_df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

# Initialize and fit model
model = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
model.fit(prophet_df)

# Make future predictions (30 days)
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
df_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)

# Plot forecast using Plotly
fig10 = go.Figure()

# Add actual data points
fig10.add_trace(go.Scatter(
    x=prophet_df['ds'],
    y=prophet_df['y'],
    mode='markers',
    name='Actual',
    marker=dict(color='black', size=3)
))

# Add forecast line
fig10.add_trace(go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat'],
    mode='lines',
    name='Forecast',
    line=dict(color='#0072B2', width=2)
))

# Add uncertainty interval
fig10.add_trace(go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat_upper'],
    mode='lines',
    line=dict(width=0),
    showlegend=False,
    hoverinfo='skip'
))

fig10.add_trace(go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat_lower'],
    mode='lines',
    line=dict(width=0),
    fillcolor='rgba(0, 114, 178, 0.2)',
    fill='tonexty',
    name='Uncertainty',
    hoverinfo='skip'
))

fig10.update_layout(
    title=dict(text='Netflix Stock Price Forecast (Prophet)', font=dict(size=16, family='Arial, bold')),
    xaxis_title='Date',
    yaxis_title='Price (USD)',
    hovermode='x unified',
    template='plotly_white'
)
st.plotly_chart(fig10, width="stretch")

# Determine which components exist
components = []
if 'trend' in forecast.columns:
    components.append('trend')
if 'weekly' in forecast.columns:
    components.append('weekly')
if 'yearly' in forecast.columns:
    components.append('yearly')

fig11 = make_subplots(rows=len(components), cols=1, subplot_titles=components)

for i, component in enumerate(components, 1):
    fig11.add_trace(
        go.Scatter(
            x=forecast['ds'],
            y=forecast[component],
            mode='lines',
            name=component.capitalize(),
            line=dict(color='#0072B2')
        ),
        row=i, col=1
    )
    fig11.update_xaxes(title_text='Date', row=i, col=1)
    fig11.update_yaxes(title_text=component.capitalize(), row=i, col=1)

fig11.update_layout(height=300*len(components), showlegend=False)
st.plotly_chart(fig11, width="stretch")

with col2:
    def style_table(df_forecast,color_theme):
        html = (
            df_forecast.style
            .hide()
            .set_table_styles([
                {"selector": "table", "props": "width: 100%; border-collapse: collapse;"},
                {"selector": "th", "props": f"background-color: {color_theme}; color: white; padding: 8px; text-align: center; position: sticky; top: 0;"},
                {"selector": "td", "props": "padding: 8px; text-align: center; border-bottom: 1px solid #ddd;"},
                {"selector": "tr:hover", "props": "background-color: #f5f5f5;"},
            ])
            .to_html()
        )
        return f'<div style="height: 380px; overflow: auto; border: 1px solid #ccc; border-radius: 8px;">{html}</div>'
st.markdown(style_table(df_forecast, color_theme="#2e7d32"), unsafe_allow_html=True)


st.markdown("---")
st.markdown(
    Components.section_header("Trend & Risk Analysis", "💹"),
    unsafe_allow_html=True
)
col1, col2, col3 = st.columns(3)
with col1:
    # Trend analysis
    if df['Close'].iloc[-1] > df['MA_30'].iloc[-1] > df['MA_90'].iloc[-1]:
        trend = "Strong Uptrend"
    elif df['Close'].iloc[-1] < df['MA_30'].iloc[-1] < df['MA_90'].iloc[-1]:
        trend = "Strong Downtrend"
    else:
        trend = "Sideways/Mixed"
    st.markdown(
        Components.metric_card(
            title="Current Trend",
            value=f"{trend}",
            delta="⬇",
            card_type="warning"
        ), unsafe_allow_html=True
    )
with col2:
    # Calculate Sharpe Ratio (assuming risk-free rate of 2% annually)
    risk_free_rate = 0.02 / 252 # Daily risk-free rate
    sharpe_ratio = (df['Daily_Return'].mean() / 100 - risk_free_rate) / (df['Daily_Return'].std() / 100)
    st.markdown(
        Components.metric_card(
            title="Sharpe Ratio",
            value=f"{sharpe_ratio:.3f}",
            delta="⬅",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col3:
    cumulative_returns = (1 + df['Daily_Return'] / 100).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    st.markdown(
        Components.metric_card(
            title="Drawdown",
            value=f"{max_drawdown * 100:.2f}%",
            delta="⬇",
            card_type="error"
        ), unsafe_allow_html=True
    )
st.markdown("---")
st.markdown(
    Components.section_header("Netflix Stock Analysis Dashboard", "📊"),
    unsafe_allow_html=True
)
with st.container():
    # Create subplots
    fig12 = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=('Stock Price with Moving Averages', 'Trading Volume', 'RSI Indicator'),
    row_heights=[0.5, 0.25, 0.25]
    )

    # Candlestick chart
    fig12.add_trace(
    go.Candlestick(
    x=df['Date'],
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name='OHLC'
    ),
    row=1, col=1
    )

    # Moving averages
    fig12.add_trace(
    go.Scatter(x=df['Date'], y=df['MA_7'], name='7-Day MA', line=dict(color='blue', width=1)),
    row=1, col=1
    )

    fig12.add_trace(
    go.Scatter(x=df['Date'], y=df['MA_30'], name='30-Day MA', line=dict(color='orange', width=1)),  
    row=1, col=1
    )

    fig12.add_trace(
    go.Scatter(x=df['Date'], y=df['MA_90'], name='90-Day MA', line=dict(color='green', width=1)),
    row=1, col=1
    )

    # Volume bar chart
    fig12.add_trace(
    go.Bar(x=df['Date'], y=df['Volume'], name='Volume', marker_color='steelblue', opacity=0.6),
    row=2, col=1    
    )

    # Volume MA
    fig12.add_trace(
    go.Scatter(x=df['Date'], y=df['Volume_MA_30'], name='Volume 30-Day MA', line=dict(color='orange', width=2)),
    row=2, col=1
    )

    # RSI
    fig12.add_trace(
    go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='purple', width=1.5)),
    row=3, col=1
    )

# RSI thresholds
fig12.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1, annotation_text="Overbought")
fig12.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1, annotation_text="Oversold")

# Update layout
fig12.update_layout(
title='Netflix Stock Analysis Dashboard',
height=900,
showlegend=True,
xaxis_rangeslider_visible=False,
hovermode='x unified'
)

fig12.update_yaxes(title_text="Price (USD)", row=1, col=1)
fig12.update_yaxes(title_text="Volume", row=2, col=1)
fig12.update_yaxes(title_text="RSI", row=3, col=1)
fig12.update_xaxes(title_text="Date", row=3, col=1)

st.plotly_chart(fig12, width="stretch")

st.markdown("---")
st.markdown(
    Components.section_header("Technical Analysis Signals", "🎯"),
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        Components.insight_box(
            title="Assessments",
            content=st.markdown("""
                f"⚠️ Trend: BEARISH - Price below both moving averages"\n
                f"🟡 RSI (32.7): NEUTRAL - No extreme conditions"\n
                f"✅ MACD: BULLISH - MACD above signal line"\n
                f"🟡 Bollinger Bands: Price within normal range"
                """, unsafe_allow_html=True),
            box_type="info"
    ),
    unsafe_allow_html=True
)

with col2:
    st.markdown(
        Components.insight_box(
            title="Volatility Analysis",
            content=f"⚠️ Current volatility (5.34%) is HIGH - Increased risk",
            box_type="error"
    ),
    unsafe_allow_html=True
)

with col3:
    st.markdown(
        Components.insight_box(
            title="Volume Analysis",
            content=f"⚡ Recent volume is ELEVATED - Strong interest/momentum",
            box_type="success"
    ),
    unsafe_allow_html=True
)

with col4:
    st.markdown(
        Components.insight_box(
            title="Risk Assessment",
            content=f"🟡 Sharpe Ratio (0.03): MODERATE - Returns compensate for risk"\n
                    f"⚠️ Max Drawdown (-48.0%): HIGH - Significant downside risk experienced",
            box_type="warning"
    ),
    unsafe_allow_html=True
)

st.markdown("---")
st.markdown(
    Components.section_header("Advanced Analytics", "🚀"),
    unsafe_allow_html=True
)
with st.container():
    st.markdown("Bollinger Bands")
    
    # Calculate Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)

fig14 = go.Figure()

# Add filled area between upper and lower bands
fig14.add_trace(go.Scatter(
    x=df['Date'],
    y=df['BB_Upper'],
    mode='lines',
    line=dict(color='red', dash='dash', width=2),
    opacity=0.7,
    name='Upper Band',
    showlegend=True
))

fig14.add_trace(go.Scatter(
    x=df['Date'],
    y=df['BB_Lower'],
    mode='lines',
    line=dict(color='green', dash='dash', width=2),
    opacity=0.7,
    name='Lower Band',
    fill='tonexty',
    fillcolor='rgba(128, 128, 128, 0.1)',
    showlegend=True
))

# Add middle band
fig14.add_trace(go.Scatter(
    x=df['Date'],
    y=df['BB_Middle'],
    mode='lines',
    line=dict(color='blue', dash='dash', width=2),
    opacity=0.7,
    name='Middle Band (20-Day SMA)'
))

# Add close price
fig14.add_trace(go.Scatter(
    x=df['Date'],
    y=df['Close'],
    mode='lines',
    line=dict(color='#E50914', width=2),
    name='Close Price'
))

fig14.update_layout(
    title=dict(
        text='Netflix Stock with Bollinger Bands',
        font=dict(size=16, family='Arial, bold')
    ),
    xaxis_title='Date',
    yaxis_title='Price (USD)',
    width=1400,
    height=700,
    hovermode='x unified',
    legend=dict(orientation='v'),
    xaxis=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.3)'),
    yaxis=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.3)')
)
st.plotly_chart(fig14, width="stretch")

st.markdown("---")

with st.container():
    st.markdown("MACD (Moving Average Convergence Divergence)")
    # Calculate MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']

# Create subplots with height ratios
fig17 = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.67, 0.33],
    subplot_titles=('', '')
)

# Price chart
fig17.add_trace(
    go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#E50914', width=2)
    ),
    row=1, col=1
)

# MACD line
fig17.add_trace(
    go.Scatter(
        x=df['Date'],
        y=df['MACD'],
        mode='lines',
        name='MACD',
        line=dict(color='blue', width=1.5)
    ),
    row=2, col=1
)

# Signal line
fig17.add_trace(
    go.Scatter(
        x=df['Date'],
        y=df['Signal_Line'],
        mode='lines',
        name='Signal Line',
        line=dict(color='red', width=1.5)
    ),
    row=2, col=1
)

# MACD Histogram
fig17.add_trace(
    go.Bar(
        x=df['Date'],
        y=df['MACD_Histogram'],
        name='MACD Histogram',
        marker=dict(color='gray', opacity=0.3)
    ),
    row=2, col=1
)

# Add horizontal line at y=0 for MACD subplot
fig17.add_hline(y=0, line=dict(color='black', width=0.8), row=2, col=1)

# Update layout
fig17.update_xaxes(title_text='Date', title_font=dict(size=12), row=2, col=1)
fig17.update_yaxes(title_text='Price (USD)', title_font=dict(size=12), row=1, col=1)
fig17.update_yaxes(title_text='MACD', title_font=dict(size=12), row=2, col=1)

fig17.update_layout(
    title=dict(
        text='Netflix Stock with MACD Indicator',
        font=dict(size=16, family='Arial, sans-serif')
    ),
    height=800,
    width=1400,
    showlegend=True,
    hovermode='x unified',
    xaxis=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.3)'),
    yaxis=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.3)'),
    xaxis2=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.3)'),
    yaxis2=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.3)')
)
st.plotly_chart(fig17, width="stretch")

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