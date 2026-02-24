import streamlit as st  
import pandas as pd  
import numpy as np  
import plotly.express as px  
import plotly.graph_objects as go  
from datetime import datetime  

from utils.theme import Components, Colors, apply_chart_theme, init_page

init_page("Retail Inventory Analysis Dashboard", "📦")

# Load custom CSS
try:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Custom CSS file not found. Using default styling.")

@st.cache_data  
def load_data():  
    df = pd.read_csv('retail_store_inventory.csv')  
    df['Date'] = pd.to_datetime(df['Date'])  
  
    # Feature engineering  
    df['Year'] = df['Date'].dt.year  
    df['Month'] = df['Date'].dt.month  
    df['Quarter'] = df['Date'].dt.quarter  
    df['Revenue'] = df['Units Sold'] * df['Price'] * (1 - df['Discount']/100)  
    df['Forecast_Error'] = df['Units Sold'] - df['Demand Forecast']
    df['Forecast_Accuracy'] = 1 - abs(df['Forecast_Error']) / df['Demand Forecast']  
    df['Stock_to_Sales_Ratio'] = df['Inventory Level'] / (df['Units Sold'] + 1)  
    df['Price_vs_Competitor'] = df['Price'] - df['Competitor Pricing']  
    df['Stockout_Risk'] = (df['Inventory Level'] < df['Demand Forecast']).astype(int)  
    df['Overstock_Risk'] = (df['Inventory Level'] > 2 * df['Demand Forecast']).astype(int)  
  
    return df  
  
df = load_data()  

# Title
st.markdown(
    Components.page_header(
        "📦  Retail Inventory Analysis Dashboard"
    ), unsafe_allow_html=True
)
st.markdown("""
<div style='text-align: center; font-size: 1.8rem'>
    <p>Comprehensive insights into sales, inventory, and demand forecasting</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown(
    Components.section_header(
        "Filters", "🔍"
    ), unsafe_allow_html=True
)

# Date range filter
min_date = df['Date'].min()
max_date = df['Date'].max()
date_range = st.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Category filter
categories = st.multiselect(
    "Select Categories",
    options=df['Category'].unique(),
    default=df['Category'].unique()
)

# Region filter
regions = st.multiselect(
    "Select Regions",
    options=df['Region'].unique(),
    default=df['Region'].unique()
)

# Weather filter
weather = st.multiselect(
    "Weather Conditions",
    options=df['Weather Condition'].unique(),
    default=df['Weather Condition'].unique()
)

# Apply filters
filtered_df = df[
    (df['Date'] >= pd.to_datetime(date_range[0])) &
    (df['Date'] <= pd.to_datetime(date_range[1])) &
    (df['Category'].isin(categories)) &
    (df['Region'].isin(regions)) &
    (df['Weather Condition'].isin(weather))
]