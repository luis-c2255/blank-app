import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

from utils.theme import Components, Colors, apply_chart_theme, init_page

init_page("Healthcare Symptoms Analytics Dashboard", "🏥")

# Load custom CSS
try:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Custom CSS file not found. Using default styling.")

st.markdown(
    Components.page_header(
        "🏥 Healthcare Symptoms Analytics Dashboard"
    ), unsafe_allow_html=True
)

# Load data and model
@st.cache_data
def load_data():
    df = pd.read_csv('Healthcare_symptoms_cleaned.csv')
    return df

@st.cache_resource
def load_model():
    model = joblib.load('disease_prediction_model.pkl')
    gender_enc = joblib.load('gender_encoder.pkl')
    return model, gender_enc

df = load_data()
model, gender_enc = load_model()

st.markdown(
    Components.section_header("Metrics", "🎯"), unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        Components.metric_card(
            title="Total Patients",
            value=f"{len(df):,}",
            delta="Patients",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col2: 
    st.markdown(
        Components.metric_card(
            title="Unique Diseases",
            value=f"{df['Disease'].nunique()}",
            delta="Diseases",
            card_type="warning"
        ), unsafe_allow_html=True
    )
with col3:
    st.markdown(
        Components.metric_card(
            title="Avg Patient Age",
            value=f"{df['Age'].mean():.1f}",
            delta="Age",
            card_type="success"
        ), unsafe_allow_html=True
    )
with col4: 
    st.markdown(
        Components.metric_card(
            title="Avg Symptoms/Patient",
            value=f"{df['Symptom_Count'].mean():.2f}",
            delta="Symptoms",
            card_type="error"
        ), unsafe_allow_html=True
    )
st.markdown("---")
st.markdown(
    Components.section_header("Disease Distribution", "📈"), unsafe_allow_html=True
)
with st.container():
    disease_counts = df['Disease'].value_counts().head(15)
    fig = px.bar(
        disease_counts, 
        orientation='h',
        labels={'value': 'Number of Patients', 'index': 'Disease'},
        title='Top 15 Most Common Diseases'
    )
    fig = apply_chart_theme(fig)
    fig.update_layout(showlegend=False)
    fig.update_traces(marker_color=Colors.CHART_COLORS)
    st.plotly_chart(fig, width="stretch", height=600)

st.markdown("---")
st.markdown(
    Components.section_header("Age Distribution", "📅"), unsafe_allow_html=True
)
with st.container():
    fig_age = px.histogram(
        df, 
        x='Age', 
        nbins=30,
        title='Patient Age Distribution', 
    )
    fig_age = apply_chart_theme(fig_age)
    fig_age.update_traces(opacity=0.75, marker_color=Colors.CHART_COLORS)
    st.plotly_chart(fig_age, width="stretch", height=600)

st.markdown("---")
st.markdown(
    Components.section_header("Gender Distribution", "👥"), unsafe_allow_html=True
)

with st.container():
    fig_gender = px.pie(
        df, 
        names='Gender',
        title='Gender Distribution',
        color_discrete_sequence=Colors.CHART_COLORS
    )
    fig_gender = apply_chart_theme(fig_gender)
    st.plotly_chart(fig_gender, width="stretch", height=600)

st.markdown("---")
st.markdown(
    Components.page_header("🔬 Disease Analysis"), unsafe_allow_html=True
)

# Disease selector
diseases = sorted(df['Disease'].unique())
selected_disease = st.selectbox("Select Disease", diseases)

disease_df = df[df['Disease'] == selected_disease]

col1, col2, col3 = st.columns(3)
with col1: 
    st.markdown(
        Components.metric_card(
            title="Total Cases",
            value=f"{len(disease_df)}",
            delta="Cases",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col2: 
    st.markdown(
        Components.metric_card(
            title="Avg Age",
            value=f"{disease_df['Age'].mean():.1f}",
            delta="Age",
            card_type="warning"
        ), unsafe_allow_html=True
    )
with col3: 
    st.markdown(
        Components.metric_card(
            title="Avg Symptoms",
            value=f"{disease_df['Symptom_Count'].mean():.2f}",
            delta="Symptoms",
            card_type="success"
        ), unsafe_allow_html=True
    )

st.markdown("---")
st.markdown(
    Components.section_header("Age Distribution for This Disease", "📊"), unsafe_allow_html=True
)

with st.container():
    fig = px.box(
        disease_df, 
        y='Age', 
        title=f'Age Distribution - {selected_disease}'
    )
    fig = apply_chart_theme(fig)
    fig.update_traces(marker_color=Colors.CHART_COLORS)
    st.plotly_chart(fig, width="stretch", height=600)

st.markdown("---")
st.markdown(
    Components.section_header("Gender Distribution for This Disease", "👥"), unsafe_allow_html=True
)
with st.container():
    gender_dist = disease_df['Gender'].value_counts()
    fig1 = px.pie(
        values=gender_dist.values, 
        names=gender_dist.index,
        title=f'Gender Distribution - {selected_disease}', 
        hole=0.4,
        color_discrete_sequence=Colors.CHART_COLORS
    )
    fig1 = apply_chart_theme(fig1)
    st.plotly_chart(fig1, width="stretch", height=600)

st.markdown("---")
st.markdown(
    Components.section_header(f"Most Common Symptoms for {selected_disease}", "🎯"), unsafe_allow_html=True
)
with st.container():
    # Most common symptoms for this disease
    all_symptoms_disease = []
    for symptoms in disease_df['Symptoms']:
        all_symptoms_disease.extend([s.strip() for s in symptoms.split(',')])

    symptom_counts = pd.Series(all_symptoms_disease).value_counts().head(10)
    fig2 = px.bar(
        symptom_counts, 
        orientation='h',
        labels={'value': 'Frequency', 'index': 'Symptom'},
        title=f'Top 10 Symptoms for {selected_disease}'
    )
    fig2 = apply_chart_theme(fig2)
    fig2.update_traces(marker_color=Colors.CHART_COLORS)
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, width="stretch", height=600)

st.markdown("---")
# Page 3: Symptom Analyst
st.markdown(
    Components.page_header("💊 Symptom Analysis"), unsafe_allow_html=True
)

# Extract all symptoms
all_symptoms = []
for symptoms in df['Symptoms']:
    all_symptoms.extend([s.strip() for s in symptoms.split(',')])

symptom_counts = pd.Series(all_symptoms).value_counts()

st.markdown("---")
st.markdown(
    Components.section_header("Most Common Symptoms Across All Patients", "📝"), unsafe_allow_html=True
)

with st.container():
    fig = px.bar(
        symptom_counts.head(20), orientation='h',
        labels={'value': 'Frequency', 'index': 'Symptom'},
        title='Top 20 Most Common Symptoms'
    )
    fig = apply_chart_theme(fig)
    fig.update_layout(showlegend=False)
    fig.update_traces(marker_color=Colors.CHART_COLORS)
    st.plotly_chart(fig, width="stretch", height=600)

st.markdown("---")

with st.container():
    # Relationship between age and symptom count
    fig2 = px.scatter(
        df.sample(1000), x='Age', y='Symptom_Count',
        color='Gender', 
        opacity=0.5,
        title='Age vs Symptom Count (Sample)',
        trendline='lowess',
        color_discrete_sequence=Colors.CHART_COLORS
    )
    fig2 = apply_chart_theme(fig2)
    st.plotly_chart(fig2, width="stretch", height=600)


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