import streamlit as st
from utils.theme import Components, Colors, apply_chart_theme, init_page


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
        .stLinkButton { 
        border-radius: 18px; padding: 0; overflow: hidden; 
        background: rgba(255, 255, 255, 0.10); 
        backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.25); 
        box-shadow: 0 4px 12px rgba(0,0,0,0.25); 
        transition: transform 0.25s ease, 
        box-shadow 0.25s ease, border-color 0.25s ease; cursor: pointer; 
        }
        .stLinkButton:hover { 
        transform: translateY(-6px) scale(1.02); 
        box-shadow: 0 10px 28px rgba(0,0,0,0.35); 
        border-color: rgba(0, 200, 255, 0.55); /* subtle cyan glow */ 
        } 
        .stLinkButton-title { 
        padding: 14px; font-size: 1.15rem; 
        font-weight: 600; text-align: center; 
        color: white; 
        } 
        </style>
        """, unsafe_allow_html=True)

st.markdown(
    Components.page_header("📊 Multiple Analysis Dashboard"), unsafe_allow_html=True)

with st.container(height="content", width="stretch", horizontal_alignment="center"):    
    st.image("img.svg")

col1, col2, col3 = st.columns(3)
with col1:
    st.link_button("Employee Analytics Dashboard", 
    "https://blank-app-ssh25yo5mc.streamlit.app/Employee_Analytics_Dashboard",
    icon="🎯", icon_position="left", width="stretch"
    )

with col2:
    st.link_button("Sales Performance Dashboard", 
    "https://blank-app-ssh25yo5mc.streamlit.app/Sales_Performance_Dashboard",
    icon="📊", icon_position="left", width="stretch"
    )
with col3:
    st.link_button("Healthcare Symptoms Analytics Dashboard", 
    "https://blank-app-ssh25yo5mc.streamlit.app/Healthcare_Symptoms_Analytics_Dashboard", 
    icon="🏥", icon_position="left", width="stretch"
    )
st.markdown("---")
col4, col5, col6 = st.columns(3)
with col4:
    st.link_button("Madrid Weather Dashboard", 
    "https://blank-app-ssh25yo5mc.streamlit.app/Madrid_Daily_Weather_Analysis_Dashboard", 
    icon="🌤️", icon_position="left", width="stretch"
    )
with col5:
    st.link_button("Netflix Stock Analysis Dashboard", 
    "https://blank-app-ssh25yo5mc.streamlit.app/Netflix_Stock_Analysis_Dashboard", 
    icon="💹", icon_position="left", width="stretch"
    )
with col6:
    st.link_button("Retail Inventory Analysis Dashboard", 
    "https://blank-app-ssh25yo5mc.streamlit.app/Retail_Inventory_Analysis_Dashboard", 
    icon="📦", icon_position="left", width="stretch"
    )

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