import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Employee Analytics Dashboard", layout="wide")

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
st.title("🎯 Employee Analytics Dashboard")
st.markdown("---")

# KPI Row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Employees", f"{len(df):,}")

with col2:
    avg_sat = df['satisfaction_level'].mean()
    st.metric("Avg Satisfaction", f"{avg_sat:.2f}",
    delta=f"{(avg_sat-0.5):.2f}" if avg_sat >= 0.5 else f"{(avg_sat-0.5):.2f}")

with col3:
    overworked_pct = (df['overworked'].sum() / len(df)) * 100
    st.metric("Overworked %", f"{overworked_pct:.1f}%")

with col4:
    promotion_rate = df['promotion_last_5years'].mean() * 100
    st.metric("Promotion Rate", f"{promotion_rate:.1f}%")

with col5:
    high_risk = len(df[(df['satisfaction_level'] < 0.4) & (df['last_evaluation'] > 0.7)])
    st.metric("High-Risk Employees", high_risk, delta=f"{(high_risk/len(df)*100):.1f}%")

    st.markdown("---")

# Filters
st.header("🔍 Filters")
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
    fig1 = px.histogram(df_filtered, x='satisfaction_level', nbins=30,
    color='salary', barmode='overlay',
    title="Satisfaction by Salary Level")
    st.plotly_chart(fig1, width="stretch", height=500)

with st.container():
    st.subheader("🏢 Department Satisfaction")
    dept_sat = df_filtered.groupby('dept')['satisfaction_level'].mean().sort_values()
    fig2 = px.bar(dept_sat, orientation='h',
    title="Average Satisfaction by Department")
    fig2.update_layout(showlegend=False, yaxis_title="", xaxis_title="Satisfaction")
    st.plotly_chart(fig2, width="stretch", height=500)



with st.container():
    st.subheader("⚡ Workload Analysis")
    fig3 = px.scatter(df_filtered, x='number_project', y='average_montly_hours',
    color='satisfaction_level', size='last_evaluation',
    hover_data=['dept', 'salary'],
    color_continuous_scale='RdYlGn',
    title="Projects vs Hours (sized by evaluation)")
    fig3.add_hline(y=250, line_dash="dash", line_color="red",
    annotation_text="Overwork Threshold")
    st.plotly_chart(fig3, width="stretch", height=500)

with st.container():
    st.subheader("🎯 Performance vs Satisfaction")
    fig4 = px.density_contour(df_filtered, x='last_evaluation',
    y='satisfaction_level',
    title="Performance-Satisfaction Density")
    fig4.update_traces(contours_coloring="fill", contours_showlabels=True)
    st.plotly_chart(fig4, width="stretch", height=500)

# High-risk employees table
st.markdown("---")
st.subheader("⚠️ High-Risk Employees (Low Satisfaction + High Performance)")

at_risk = df_filtered[(df_filtered['satisfaction_level'] < 0.4) &
(df_filtered['last_evaluation'] > 0.7)][
['emp_id', 'satisfaction_level', 'last_evaluation', 'number_project',
'average_montly_hours', 'dept', 'salary', 'promotion_last_5years']
].sort_values('satisfaction_level')

st.dataframe(at_risk, use_container_width=True)
st.download_button("📥 Download High-Risk List",
at_risk.to_csv(index=False).encode('utf-8'),
"high_risk_employees.csv", "text/csv")

# Department deep dive
st.markdown("---")
st.subheader("🔬 Department Deep Dive")

selected_dept = st.selectbox("Select Department", df_filtered['dept'].unique())
dept_data = df_filtered[df_filtered['dept'] == selected_dept]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Employees", len(dept_data))
with col2:
    st.metric("Avg Satisfaction", f"{dept_data['satisfaction_level'].mean():.2f}")
with col3:
    st.metric("Avg Evaluation", f"{dept_data['last_evaluation'].mean():.2f}")

# Department metrics
fig5 = go.Figure()
metrics = ['satisfaction_level', 'last_evaluation', 'promotion_last_5years']
for metric in metrics:
    fig5.add_trace(go.Box(y=dept_data[metric], name=metric.replace('_', ' ').title()))
    fig5.update_layout(title=f"{selected_dept.title()} - Key Metrics Distribution")
    st.plotly_chart(fig5, use_container_width=True)