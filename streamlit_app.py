import streamlit as st
import sys
import os
from utils.theme import Components, Colors, apply_chart_theme, init_page
import base64

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)
# --- Helper to load images as base64 ---
def load_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# --- Load images ---
img_employee = load_base64("utils/employee.png")
img_sales = load_base64("utils/sales.png")
img_health = load_base64("utils/healthcare.png")
img_weather = load_base64("utils/weather.png")
img_stock = load_base64("utils/stocks.png")
img_retail = load_base64("utils/retail.png")


st.markdown("""
    <style>
        /* Stylize the search input */
        div[data-testid="stSidebarNavSearch"] input {
            border: 1px solid #4cc9a6 !important;
            border-radius: 5px;
        }
        /* Highlight the navigation links to look like buttons */
        [data-testid="stSidebarNav"] ul {
            padding-top: 2rem;
        }
        [data-testid="stSidebarNav"] li {
            background-color: #1a2c42; /* Darker blue background */
            border-radius: 10px;
            margin-bottom: 10px;
            border: 1px solid #4cc9a6; /* Your Mint Leaf color */
            transition: 0.3s;
        }
        [data-testid="stSidebarNav"] li:hover {
            background-color: #4cc9a6;
            transform: translateX(5px);
        }
        [data-testid="stSidebarNav"] span {
            color: white !important;
            font-weight: bold;
        }
        .card {
            border-radius: 18px;
            padding: 0;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.10);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.25);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
            transition: transform 0.25 ease, box-shadow 0.25 ease, border-color 0.25 ease;
            cursor: pointer;
        }
        .card:hover {
            transform: translateY(-6px) scale(1.02);
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.35);
            border-color: rgba(0, 200, 255, 0.55);
        }
        .card img {
            width: 100%;
            height: 170px;
            object-fit: cover;
        }
        .card-title  {
            padding: 14px;
            font-size: 1.15rem;
            font-weight: 600;
            text-align: center;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Load custom CSS
try:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Custom CSS file not found. Using default styling.")


st.markdown(
    Components.page_header("📊 Multiple Analysis Dashboard"), unsafe_allow_html=True)

cards = [
    ("Employee Dashboard", img_employee, "pages/1_🎯_Employee_Analytics_Dashboard.py"),
    ("Sales Performance", img_sales, "pages/2_📊_Sales_Performance_Dashboard.py"),
    ("Healthcare Symptoms", img_health, "pages/3_🏥_Healthcare_Symptoms_Analytics_Dashboard.py"),
    ("Weather Analysis", img_weather, "pages/4_🌤️_Madrid_Daily_Weather_Analysis_Dashboard.py"),
    ("Stock Analysis", img_stock, "pages/5_💹_Netflix_Stock_Analysis_Dashboard.py"),
    ("Retail Inventory", img_retail, "pages/6_📦_Retail_Inventory_Analysis_Dashboard.py"),
]

# --- Responsive grid ---
cols = st.columns(2, gap="large")

for i, (title, img, page) in enumerate(cards):
    with cols[i % 2]:
        st.page_link(
            page,
            label=f"""
            <div class="card">
            <img src="data:image/png;base64, {img}" />
            <div class="card-title">{title}</div>
            </div>
            """, width="stretch"
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