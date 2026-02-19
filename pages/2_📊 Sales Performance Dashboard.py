import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Sales Dashboard", layout="wide")
st.title("📊 Sales Performance Dashboard")

# Load data
df = pd.read_csv('Sales_Data.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Filters
st.header("Filters")
date_range = st.date_input("Date Range", [df['Date'].min(), df['Date'].max()])
regions = st.multiselect("Region", df['Region'].unique(), default=df['Region'].unique())
categories = st.multiselect("Category", df['Product Category'].unique(),
default=df['Product Category'].unique())

# Filter data
filtered_df = df[
(df['Date'].between(*date_range)) &
(df['Region'].isin(regions)) &
(df['Product Category'].isin(categories))
]

# KPI Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", f"${filtered_df['Total Revenue'].sum():,.0f}")
col2.metric("Transactions", f"{len(filtered_df):,}")
col3.metric("Units Sold", f"{filtered_df['Units Sold'].sum():,}")
col4.metric("Avg Order", f"${filtered_df['Total Revenue'].mean():,.2f}")

# Visualizations
with st.container():
    fig1 = px.bar(filtered_df.groupby('Product Category')['Total Revenue'].sum().reset_index(),
    x='Product Category', y='Total Revenue', title="Revenue by Category")
    st.plotly_chart(fig1, width="stratch", height=800)

with st.container():
    fig2 = px.pie(filtered_df.groupby('Region')['Total Revenue'].sum().reset_index(),
    values='Total Revenue', names='Region', title="Regional Distribution")
    st.plotly_chart(fig2, width="stretch", height=800)

with st.container():
# Monthly trend
    monthly = filtered_df.groupby(filtered_df['Date'].dt.to_period('M'))['Total Revenue'].sum().reset_index()
    monthly['Date'] = monthly['Date'].astype(str)
    fig3 = px.line(monthly, x='Date', y='Total Revenue', title="Monthly Revenue Trend")
    st.plotly_chart(fig3, width="stretch", height=800)
    
payment_analysis = filtered_df.groupby('Payment Method').agg(
    total_revenue = ('Total Revenue', 'sum'),
    transactions = ('Transaction ID', 'count'),
    avg_transaction_value = ('Total Revenue', 'mean')
).round(2)
payment_analysis = payment_analysis.sort_values('total_revenue', ascending=False)
with st.container():
    fig4 = px.bar(payment_analysis, x='Payment Method', y=['total_revenue', 'transactions', 'avg_transaction_value'], title='Payment Method Performance',
                  barmode='group', text_auto=True)
    fig4.update_layout(yaxis_title='Value')
    st.plotly_chart(fig4, width="stretch", height=800)

top_products = filtered_df.groupby("Product Name")["Total Revenue"].sum().sort_values(ascending=False).head(10)
top_10_df = top_products.reset_index().head(10)
top_10_df.columns = ['Product Name', 'Revenue']
with st.container():
    fig5 = px.bar(top_10_df, y="Product Name", y="Revenue", orientation="h", title="Top 10 Products by Revenue",
                  text="Revenue", color="Revenue")
    fig5.update_traces(texttemplate="$%{text:,.0f}", textposition='outside')
    fig5.update_layout(showlegend=False)
    st.plotly_chart(fig5, width="stretch", height=800)
# Data table
st.subheader("📋 Transaction Details")
st.dataframe(filtered_df, use_container_width=True)