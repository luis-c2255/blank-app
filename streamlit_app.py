import streamlit as st
import sys
import os
from utils.theme import Components, Colors, apply_chart_theme, init_page


st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)
pages = [
    {
        "title": "Employee Analytics",
        "emoji": "🎯",
        "description": "Workforce insights, HR metrics & team performance",
        "path": "pages/1_🎯_Employee_Analytics_Dashboard.py",
        "color": "#4F46E5",
        "bg": "#EEF2FF",
    },
    {
        "title": "Sales Performance",
        "emoji": "📊",
        "description": "Revenue trends, pipeline & sales KPIs",
        "path": "pages/2_📊_Sales_Performance_Dashboard.py",
        "color": "#059669",
        "bg": "#ECFDF5",
    },
    {
        "title": "Healthcare Symptoms",
        "emoji": "🏥",
        "description": "Patient data, symptom patterns & health analytics",
        "path": "pages/3_🏥_Healthcare_Symptoms_Analytics_Dashboard.py",
        "color": "#DC2626",
        "bg": "#FEF2F2",
    },
    {
        "title": "Madrid Weather",
        "emoji": "🌤️",
        "description": "Daily weather patterns & climate analysis for Madrid",
        "path": "pages/4_🌤️_Madrid_Daily_Weather_Analysis_Dashboard.py",
        "color": "#D97706",
        "bg": "#FFFBEB",
    },
    {
        "title": "Netflix Stock",
        "emoji": "💹",
        "description": "Stock price history, trends & financial metrics",
        "path": "pages/5_💹_Netflix_Stock_Analysis_Dashboard.py",
        "color": "#E50914",
        "bg": "#FFF1F2",
    },
    {
        "title": "Retail Inventory",
        "emoji": "📦",
        "description": "Stock levels, turnover rates & supply chain insights",
        "path": "pages/6_📦_Retail_Inventory_Analysis_Dashboard.py",
        "color": "#7C3AED",
        "bg": "#F5F3FF",
    },
]
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
        .card-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.25rem;
        }
        .card {
            border-radius: 16px;
            padding: 2rem;
            text-decoration: none;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            border: 1.5px solid transparent;
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
            position: relative;
            overflow: hidden;
            cursor: pointer;
        }
        .card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.12);
        }
        .card-title  {
            font-size: 1.2rem;
            font-weight: 700;
            margin: 0;
        }
        .card-desc {
            font-size: 0.875rem;
            margin: 0;
            opacity: 0.75;
            line-height: 1.5;
        }
        .card-arrow {
            position: absolute;
            right: 1.5rem;
            top: 50%;
            transform: translateY(-50%);
            font-size: 1.25rem;
            opacity: 0.4;
            transition: opacity 0.2s, right 0.2s;
        }
        .card:hover .card-arrow {
            opacity: 0.9;
            right: 1.25rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    Components.page_header("📊 Multiple Analysis Dashboard"), unsafe_allow_html=True)
    
card_html = '<div class="card-grid">'
for page in pages:
    cards_html += f"""
    <a class="card" href="{page['path'].replace('pages/', '').replace('.py', '').replace(' ', '_')}"
        style="background-color: {page['bg']}; color: {page['color']}; border-color: {page['color']}22;">
        <span class="card-emoji">{page['emoji']}</span>
        <p class="card-title">{page['title']}</p>
        <p class="card-desc">{page['description']}</p>
        <span class="card-arrow">→</span>
    </a>
    """
cards_html += '</div>'

st.markdown(cards_html, unsafe_allow_html=True)

# Load custom CSS
try:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Custom CSS file not found. Using default styling.")

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