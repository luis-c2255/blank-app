import streamlit as st

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

    </style>
""", unsafe_allow_html=True)

# 1. Page Configuration
st.set_page_config(
    page_title="Multiple Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title(
    st.markdown( "# 📊 Multiple Analysis Dashboard"),
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