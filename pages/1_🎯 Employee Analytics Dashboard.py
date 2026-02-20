import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.theme import Components, Colors, apply_chart_theme, init_page

init_page("Employee Analytics Dashboard", "📊")

# Load custom CSS
try:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Custom CSS file not found. Using default styling.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('emp_att.csv')

    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df['high_performer'] = ((df['last_evaluation'] >= 0.7) &
    (df['number_project'] >= 4)).astype(int)
    df['overworked'] = (df['average_montly_hours'] > 250).astype(int)
    return df

df = load_data()

# Title
st.markdown(
    Components.page_header(
        "📊 Employee Analytics Dashboard"
    ), unsafe_allow_html=True
)

st.markdown("---")

# KPI Row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(
        Components.metric_card(
            title="Total Employees",
            value= f"{len(df):,}",
            delta="Employees",
            card_type="info"
        ),
        unsafe_allow_html=True
    )

with col2:
    avg_sat = df['satisfaction_level'].mean()
    st.markdown(
        Components.metric_card(
            title="Avg Satisfaction",
            value= f"{avg_sat:.2f}",
            delta=f"{(avg_sat-0.5):.2f}" if avg_sat >= 0.5 else f"{(avg_sat-0.5):.2f}",
            card_type="success"
        ),
        unsafe_allow_html=True
    )

with col3:
    overworked_pct = (df['overworked'].sum() / len(df)) * 100
    st.markdown(
        Components.metric_card(
            title="Overworked %",
            value= f"{overworked_pct:.1f}%",
            delta="Overworked",
            card_type="info"
        ),
        unsafe_allow_html=True
    )

with col4:
    promotion_rate = df['promotion_last_5years'].mean() * 100
    st.markdown(
        Components.metric_card(
            title="Promotion Rate",
            value= f"{promotion_rate:.1f}%",
            delta="Promotion Rate",
            card_type="info"
        ),
        unsafe_allow_html=True
    )

with col5:
    high_risk = len(df[(df['satisfaction_level'] < 0.4) & (df['last_evaluation'] > 0.7)])
    st.markdown(
        Components.metric_card(
            title="High-Risk Employees",
            value="high_risk", 
            delta=f"{(high_risk/len(df)*100):.1f}%",
            card_type="warning"
        ),
        unsafe_allow_html=True
    )

st.markdown("---")

# Filters
st.markdown(
    Components.section_header("Filters", "🔍"),
    unsafe_allow_html=True
)
dept_filter = st.multiselect("Department", options=df['dept'].unique(),
default=df['dept'].unique())
salary_filter = st.multiselect("Salary Level",
options=['low', 'medium', 'high'],
default=['low', 'medium', 'high'])

# Apply filters
df_filtered = df[(df['dept'].isin(dept_filter)) & (df['salary'].isin(salary_filter))]

# Visualizations

with st.container():
    st.subheader("📊 Satisfaction Distribution")
    fig1 = px.histogram(
        df_filtered, 
        x='satisfaction_level', 
        nbins=30,
        color='salary', 
        barmode='overlay',
        title="Satisfaction by Salary Level"
    )
    fig1 = apply_chart_theme(fig1)
    fig1.update_traces(marker_color=Colors.CHART_COLORS)
    fig1.update_layout(legend_traceorder="reversed")
    st.plotly_chart(fig1, width="stretch", height=600)

with st.container():
    st.subheader("🏢 Department Satisfaction")
    dept_sat = df_filtered.groupby('dept')['satisfaction_level'].mean().sort_values()
    fig2 = px.bar(
        dept_sat, 
        orientation='h',
        title="Average Satisfaction by Department"
    )
    fig2 = apply_chart_theme(fig2)
    fig2.update_traces(marker_color=Colors.CHART_COLORS)
    fig2.update_layout(showlegend=False, yaxis_title="", xaxis_title="Satisfaction")
    st.plotly_chart(fig2, width="stretch", height=600)



with st.container():
    st.subheader("⚡ Workload Analysis")
    fig3 = px.scatter(
        df_filtered, 
        x='number_project', 
        y='average_montly_hours',
        color='satisfaction_level', 
        size='last_evaluation',
        hover_data=['dept', 'salary'],
        title="Projects vs Hours (sized by evaluation)",
        color_discrete_sequence=Colors.CHART_COLORS
    )
    fig3 = apply_chart_theme(fig3)
    fig3.add_hline(y=250, line_dash="dash", line_color="red",
    annotation_text="Overwork Threshold")
    st.plotly_chart(fig3, width="stretch", height=600)

with st.container():
    st.subheader("🎯 Performance vs Satisfaction")
    fig4 = px.density_contour(
        df_filtered, 
        x='last_evaluation',
        y='satisfaction_level',
        title="Performance-Satisfaction Density"
    )
    fig4 = apply_chart_theme(fig4)
    fig4.update_traces(contours_coloring="fill", contours_showlabels=True)
    st.plotly_chart(fig4, width="stretch", height=600)

with st.container():
    st.subheader("Satisfaction by Department & Salary")
    fig5 = px.box(
        df_filtered, 
        x='dept', 
        y='satisfaction_level', 
        color='salary',
        title='Satisfaction by Department & Salary'
    )
    fig5 = apply_chart_theme(fig5)
    fig5.update_layout(xaxis_title='Department', yaxis_title='Satisfaction Level')
    st.plotly_chart(fig5, width='stretch', height=600)
     
with st.container():
    fig6 = px.scatter_3d(
        df_filtered, 
        x='number_project', 
        y='average_montly_hours', 
        z='satisfaction_level', 
        color='salary',
        title='Employee Performance 3D View', 
        labels={'number_project': 'Projects', 'average_montly_hours': 'Monthly Hours',
                        'satisfaction_level': 'Satisfaction'}, 
        opacity=0.7,
        color_discrete_sequence=Colors.CHART_COLORS
    )
    fig6 = apply_chart_theme(fig6)
    st.plotly_chart(fig6, width='stretch', height=800)
    
# High-risk employees table
st.markdown("---")
st.markdown(
    Components.section_header("High-Risk Employees (Low Satisfaction + High Performance)", "⚠️"),
    unsafe_allow_html=True
)

at_risk = df_filtered[(df_filtered['satisfaction_level'] < 0.4) &
(df_filtered['last_evaluation'] > 0.7)][
['emp_id', 'satisfaction_level', 'last_evaluation', 'number_project',
'average_montly_hours', 'dept', 'salary', 'promotion_last_5years']
].sort_values('satisfaction_level')

st.dataframe(at_risk, width="stretch")
# Download Button
csv = at_risk.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download High-Risk List",
    data=csv,
    file_name="high_risk_employees.csv",
    mime="text/csv"
)

# Department deep dive
st.markdown("---")
st.markdown(
    Components.section_header("Department Deep Dive", "🔬"), unsafe_allow_html=True)

selected_dept = st.selectbox("Select Department", df_filtered['dept'].unique())
dept_data = df_filtered[df_filtered['dept'] == selected_dept]

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        Components.metric_card(
            title="Employees",
            value=f"{len(dept_data)}", 
            delta="Employees",
            card_type="info"
        ),
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        Components.metric_card(
            title="Avg Satisfaction",
            value= f"{dept_data['satisfaction_level'].mean():.2f}",
            delta="Satisfaction",
            card_type="info"
        ),
        unsafe_allow_html=True
    )
with col3:
    st.markdown(
        Components.metric_card(
            title="Avg Evaluation",
            value= f"{dept_data['last_evaluation'].mean():.2f}",
            delta="Evaluation",
            card_type="info"
        ),
        unsafe_allow_html=True
    )

st.markdown("---")

# Department metrics
fig = go.Figure()
metrics = ['satisfaction_level', 'last_evaluation', 'promotion_last_5years']
for metric in metrics:
    fig.add_trace(go.Box(y=dept_data[metric], name=metric.replace('_', ' ').title()))
    fig.update_layout(title=f"{selected_dept.title()} - Key Metrics Distribution")
    fig = apply_chart_theme(fig)
    st.plotly_chart(fig, width="stretch")


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