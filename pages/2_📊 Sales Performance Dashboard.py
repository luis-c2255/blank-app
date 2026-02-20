import streamlit as st
import pandas as pd
import plotly.express as px

from utils.theme import Components, Colors, apply_chart_theme, init_page

init_page("Sales Performance Dashboard", "📊")

# Load custom CSS
try:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Custom CSS file not found. Using default styling.")

st.markdown(
    Components.page_header(
        "📊 Sales Performance Dashboard"
    ), unsafe_allow_html=True
)

# Load data
df = pd.read_csv('Sales_Data.csv')

df['Date'] = pd.to_datetime(df['Date'])

# Filters
st.markdown(
    Components.section_header("Filters", "🔍"), unsafe_allow_html=True
)

date_range = st.date_input("Date Range", [df['Date'].min(), df['Date'].max()])
regions = st.multiselect("Region", df['Region'].unique(), default=df['Region'].unique())
categories = st.multiselect("Category", df['Product Category'].unique(),
default=df['Product Category'].unique())

# Filter data
if len(date_range) == 2:
    start_date, end_date = date_range
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filtered_df = df[
        (df['Date'] >= start_date) &
        (df['Date'] <= end_date) &
        (df['Region'].isin(regions)) &
        (df['Product Category'].isin(categories))
    ]
else:
    filtered_df = df.copy()

st.markdown("---")
st.markdown(
    Components.section_header("Metrics", "🎯"), unsafe_allow_html=True
)

# KPI Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        Components.metric_card(
            title="Total Revenue",
            value=f"${filtered_df['Total Revenue'].sum():,.0f}",
            delta="Revenue",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col2:
    st.markdown(
        Components.metric_card(
            title="Transactions",
            value=f"{len(filtered_df):,}",
            delta="Transactions",
            card_type="warning"
        ), unsafe_allow_html=True
    )
with col3:
    st.markdown(
        Components.metric_card(
            title="Units Sold",
            value=f"{filtered_df['Units Sold'].sum():,}",
            delta="Units",
            card_type="success"
        ), unsafe_allow_html=True
    )
with col4:
    st.markdown(
        Components.metric_card(
            title="Avg Order",
            value=f"${filtered_df['Total Revenue'].mean():,.2f}",
            delta="Orders",
            card_type="error"
        ), unsafe_allow_html=True
    )

st.markdown("---")
st.markdown(
    Components.section_header("Visualizations", "📊"), unsafe_allow_html=True
)

# Visualizations
with st.container():
    fig1 = px.bar(
        filtered_df.groupby('Product Category')['Total Revenue'].sum().reset_index(),
        x='Product Category', 
        y='Total Revenue', 
        title="Revenue by Category", 
        text='Total Revenue', 
        color='Total Revenue',
    )
    fig1 = apply_chart_theme(fig1)
    fig1.update_traces(texttemplate='$%{text:,.0f}', textposition='outside', marker_color=Colors.CHART_COLORS)
    fig1.update_layout(showlegend=False)
    st.plotly_chart(fig1, width="stretch", height=600)

st.markdown("---")
with st.container():
    fig2 = px.pie(
        filtered_df.groupby('Region')['Total Revenue'].sum().reset_index(),
        values='Total Revenue', 
        names='Region', 
        title="Regional Distribution", 
        hole=0.4,
        color_discrete_sequence=Colors.CHART_COLORS
    )
    fig2 = apply_chart_theme(fig2)
    fig2.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig2, width="stretch", height=600)

st.markdown("---")
with st.container():
# Monthly trend
    monthly = filtered_df.groupby(filtered_df['Date'].dt.to_period('M'))['Total Revenue'].sum().reset_index()
    monthly['Date'] = monthly['Date'].astype(str)
    fig3 = px.line(
        monthly, 
        x='Date', 
        y='Total Revenue', 
        title="Monthly Revenue Trend"
    )
    fig3 = apply_chart_theme(fig3)
    fig3.update_traces(
    line_color=Colors.BLUE_ENERGY,
    line_width=3,
    mode='lines+markers',
    marker=dict(size=8)
    )
    st.plotly_chart(fig3, width="stretch", height=600)
    
payment_analysis = filtered_df.groupby('Payment Method').agg(
    total_revenue = ('Total Revenue', 'sum'),
    transactions = ('Transaction ID', 'count'),
    avg_transaction_value = ('Total Revenue', 'mean')
).round(2)
payment_analysis = payment_analysis.sort_values('total_revenue', ascending=False)

st.markdown("---")
with st.container():
    fig4 = px.bar(
        payment_analysis.reset_index(), 
        x='Payment Method', 
        y=['total_revenue', 'transactions', 'avg_transaction_value'], 
        title='Payment Method Performance',
        barmode='group', 
        text_auto=True
    )
    fig4 = apply_chart_theme(fig4)
    fig4.update_layout(yaxis_title='Value')
    st.plotly_chart(fig4, width="stretch", height=800)

top_products = filtered_df.groupby("Product Name")["Total Revenue"].sum().sort_values(ascending=False).head(10)
top_10_df = top_products.reset_index().head(10)
top_10_df.columns = ['Product Name', 'Revenue']

st.markdown("---")
with st.container():
    fig5 = px.bar(
        top_10_df, 
        y="Product Name", 
        x="Revenue", 
        orientation="h", 
        title="Top 10 Products by Revenue",
        text="Revenue", 
        color="Revenue"
    )
    fig5 = apply_chart_theme(fig5)
    fig5.update_traces(texttemplate="$%{text:,.0f}", textposition='outside', marker_color=Colors.CHART_COLORS)
    fig5.update_layout(showlegend=False)
    st.plotly_chart(fig5, width="stretch", height=600)

st.markdown("---")

# Data table
st.markdown(
    Components.section_header("Transaction Details", "📋"), unsafe_allow_html=True
)

st.dataframe(filtered_df, width="stretch")

# Download Button
csv = filtered_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Download Transaction Details",
    data=csv,
    file_name="transaction_details.csv",
    mime="text/csv"
)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>📊 Multiple Analysis Dashboard</strong></p>
    <p>Multiple Dashboards from several datasets analyzed</p>
    <p style='font-size: 0.9rem;'>Navigate using the sidebar to explore different datasets</p>
</div>
""", unsafe_allow_html=True)