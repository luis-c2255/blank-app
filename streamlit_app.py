import streamlit as st
import sys
import os

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)

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


st.title("Multiple Analysis Dashboard")

cards = [
    {
        "title": "Employee Dashboard",
        "image": "utils/employee.png",
        "page": "pages/01_🎯_Employee_Analytics_Dashboard.py"
    },
        {
        "title": "Sales Performance",
        "image": "utils/sales.png",
        "page": "pages/02_📊_Sales_Performance_Dashboard.py"
    },
        {
        "title": "Healthcare Symptoms",
        "image": "utils/healthcare.png",
        "page": "pages/03_🏥_Healthcare_Symptoms_Analytics_Dashboard.py"
    },
        {
        "title": "Weather Analysis",
        "image": "utils/weather.png",
        "page": "pages/04_🌤️_Madrid_Daily_Weather_Analysis_Dashboard.py"
    },
        {
        "title": "Stock Analysis",
        "image": "utils/stocks.png",
        "page": "pages/05_💹_Netflix_Stock_Analysis_Dashboard.py"
    },
        {
        "title": "Retail Inventor",
        "image": "utils/retail.png",
        "page": "pages/06_📦_Retail_Inventory_Analysis_Dashboard.py"
    }
]

# --- Responsive grid ---
cols = st.columns(2, gap="large")

for i, card in enumerate(cards):
    with cols[i % 2]:
        st.page_link(
            card["page"],
            label=f"""
            <div class="card">
            <img src="{card['image']}" />
            <div class="card-title">{card['title']}</div>
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