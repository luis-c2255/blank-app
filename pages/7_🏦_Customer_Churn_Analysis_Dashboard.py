import streamlit as st 
import pandas as pd 
import numpy as np 
import plotly.express as px 
import plotly.graph_objects as go 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import StandardScaler 
import pickle

from utils.theme import Components, Colors, apply_chart_theme, init_page

init_page("Customer Churn Analytics Dashboard", "🏦")

# Load custom CSS
try:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Custom CSS file not found. Using default styling.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('Churn_Modelling.csv')
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1) 
    return df

st.sidebar.header("🔍 Filters")

geography_options = ['All'] + df['Geography'].unique().tolist()
selected_geography = st.sidebar.selectbox('Geography', geography_options)

age_range = st.sidebar.slider('Age Range', int(df['Age'].min()), int(df['Age'].max()), (int(df['Age'].min)), int(df['Age'].max()))

active_filter = st.sidebar.radio("Active Member Status", ['All', 'Active', 'Inactive'])

filtered_df = df.copy()
if selected_geography != 'All':filtered_df = filtered_df[filtered_df['Geography'] == selected_geography]
    filtered_df = filtered_df[(filtered_df['Age'] >= age_range[0]) & (filtered_df['Age'] <= age_range[1])]
    if active_filter == 'Active': filtered_df = filtered_df[filtered_df['IsActiveMember'] == 1]
    elif active_filter == 'Inactive': filtered_df = filtered_df[filtered_df['IsActiveMember'] == 0]