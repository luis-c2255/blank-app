import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


from utils.theme import Components, Colors, apply_chart_theme, init_page

init_page("Madrid Weather Dashboard", "🌤️")

# Load custom CSS
try:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Custom CSS file not found. Using default styling.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('madrid_weather.csv')
    df['CET'] = pd.to_datetime(df['CET'])
    return df

df = load_data()

# Title
st.markdown(
    Components.page_header(
        "🌤️ Madrid Daily Weather Analysis Dashboard"
    ), unsafe_allow_html=True
)
st.markdown("""
<div style='text-align: center; font-size: 1.8rem'>
    <p>Comprehensive weather data from 1997-2015</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
# Filters
st.markdown(
    Components.section_header("Filters", "🔍"),
    unsafe_allow_html=True
)
year_range = st.slider("Select Year Range",
int(df['Year'].min()),
int(df['Year'].max()),
(int(df['Year'].min()), int(df['Year'].max())))

season_filter = st.multiselect("Select Seasons",
options=['Winter', 'Spring', 'Summer', 'Fall'],
default=['Winter', 'Spring', 'Summer', 'Fall'])

# Filter data
filtered_df = df[(df['Year'] >= year_range[0]) &
(df['Year'] <= year_range[1]) &
(df['Season'].isin(season_filter))]

# Filters
st.markdown(
    Components.section_header("Key Metrics", "🎯"),
    unsafe_allow_html=True
)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        Components.metric_card(
            title="Average Temperature",
            value= f"{filtered_df['Mean TemperatureC'].mean():.1f}°C",
            delta="Temperature",
            card_type="info"
        ),
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        Components.metric_card(
            title="Total Precipitation",
            value= f"{filtered_df['Precipitationmm'].sum():.0f} mm",
            delta="Precipitation",
            card_type="warning"
        ),
        unsafe_allow_html=True
    )
with col3:
    st.markdown(
        Components.metric_card(
            title="Average Humidity",
            value= f"{filtered_df['Mean Humidity'].mean():.1f}%",
            delta="Humidity",
            card_type="success"
        ),
        unsafe_allow_html=True
    )
with col4:
    rainy_days = filtered_df['Events'].str.contains('Rain', na=False, case=False).sum()
    st.markdown(
        Components.metric_card(
            title="Rainy Days",
            value= f"{rainy_days}",
            delta="Rainy Days",
            card_type="error"
        ),
        unsafe_allow_html=True
    )
st.markdown("---")

st.markdown(
    Components.page_header("🌡️ Temperature Analysis"),
    unsafe_allow_html=True
)

st.markdown(
    Components.section_header("Annual Temperature Trend", "📈"),
    unsafe_allow_html=True
)

with st.container():
    annual_temp = filtered_df.groupby('Year')['Mean TemperatureC'].mean().reset_index()
    fig1 = px.line(
        annual_temp,
        x='Year',
        y='Mean TemperatureC',
        title='Average Temperature by Year')
    fig1.update_traces(line_color=Colors.CORAL_RED, line_width=3)
    fig1 = apply_chart_theme(fig1)
    st.plotly_chart(fig1, width="stretch", height=500)

st.markdown("---")
st.markdown(
    Components.section_header("Monthly Temperature Pattern", "📊"),
    unsafe_allow_html=True
)
with st.container():
    monthly_temp = filtered_df.groupby('Month')['Mean TemperatureC'].mean().reset_index()
    fig2 = px.bar(
        monthly_temp,
        x='Month',
        y='Mean TemperatureC',
        title='Average Temperature by Month',
        color='Mean TemperatureC',
        color_continuous_scale=Colors.CHART_COLORS)
    fig2 = apply_chart_theme(fig2)
    st.plotly_chart(fig2, width="stretch", height=500)

st.markdown("---")
st.markdown(
    Components.section_header("Precipitation Analysis", "💧"),
    unsafe_allow_html=True
)
with st.container():
    annual_precip = filtered_df.groupby('Year')['Precipitationmm'].sum().reset_index()
    fig3 = px.area(
        annual_precip,
        x='Year',
        y='Precipitationmm',
        title='Annual Total Precipitation')
    fig3.update_traces(fillcolor='steelblue', line_color='darkblue')
    fig3 = apply_chart_theme(fig3)
    st.plotly_chart(fig3, width="stretch", height=500)

st.markdown("---")
st.markdown(
    Components.section_header("Weather Events Distribution", "🌦️"),
    unsafe_allow_html=True
)
with st.container():
    event_counts = filtered_df['Events'].value_counts().reset_index()
    event_counts.columns = ['Event', 'Count']
    fig4 = px.pie(
        event_counts,
        names='Event',
        values='Count',
        title='Weather Events Breakdown',
        hole=0.4,
        color_discrete_sequence=Colors.CHART_COLORS)
    fig4 = apply_chart_theme(fig4)
    fig4.update_traces(textposition='inside')
    fig4.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
    st.plotly_chart(fig4, width="stretch", height=500)

st.markdown("---")
st.markdown(
    Components.section_header("Temperature vs Humidity", "🌡️"),
    unsafe_allow_html=True
)

with st.container():
    fig5 = px.scatter(
        filtered_df.sample(min(2000, len(filtered_df))),
        x='Mean TemperatureC',
        y='Mean Humidity',
        color='Season', 
        size='Precipitationmm',
        title='Relationship between Temperature and Humidity',
        opacity=0.6)
    fig5 = apply_chart_theme(fig5)
    st.plotly_chart(fig5, width="stretch", height=500)

st.markdown("---")
st.markdown(
    Components.section_header("Wind Speed Patterns", "💨"),
    unsafe_allow_html=True
)
with st.container():
    monthly_wind = filtered_df.groupby('Month')['Mean Wind SpeedKm/h'].mean().reset_index()
    fig6 = px.line(
        monthly_wind,
        x='Month',
        y='Mean Wind SpeedKm/h',
        title='Average Wind Speed by Month',
        markers=True)
    fig6.update_traces(line_color='purple', line_width=3, marker_size=10)
    fig6 = apply_chart_theme(fig6)
    st.plotly_chart(fig6, width="stretch", height=500)

st.markdown("---")
st.markdown(
    Components.section_header("Seasonal Weather Comparison", "🍂"),
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    fig7 = px.box(filtered_df, x='Season', y='Mean TemperatureC',
    title='Temperature by Season', color='Season')
    fig7 = apply_chart_theme(fig7)
    st.plotly_chart(fig7, width="content")
with col2:
    fig8 = px.box(filtered_df, x='Season', y='Mean Humidity',
    title='Humidity by Season', color='Season')
    fig8 = apply_chart_theme(fig8)
    st.plotly_chart(fig8, width='content')
with col3:
    fig9 = px.box(filtered_df, x='Season', y='Precipitationmm',
    title='Precipitation by Season',
    color='Season')
    fig9 = apply_chart_theme(fig9)
    st.plotly_chart(fig9, width='content')

st.markdown("---")
st.markdown(
    Components.section_header("Raw Data Preview", "📋"),
    unsafe_allow_html=True
)

st.dataframe(filtered_df.head(100), width="stretch")

csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=csv,
    file_name='madrid_weather_filtered.csv',
    mime='text/csv',
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