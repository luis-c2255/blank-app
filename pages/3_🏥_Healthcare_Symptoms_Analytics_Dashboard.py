import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(page_title="Healthcare Analytics Dashboard", layout="wide")

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


st.title("🏥 Healthcare Symptoms Analytics Dashboard")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Patients", f"{len(df):,}")
col2.metric("Unique Diseases", df['Disease'].nunique())
col3.metric("Avg Patient Age", f"{df['Age'].mean():.1f}")
col4.metric("Avg Symptoms/Patient", f"{df['Symptom_Count'].mean():.2f}")

st.subheader("Disease Distribution")
disease_counts = df['Disease'].value_counts().head(15)
fig = px.bar(disease_counts, orientation='h',
labels={'value': 'Number of Patients', 'index': 'Disease'},
title='Top 15 Most Common Diseases', color='count', color_continuous_scale='Viridis')
st.plotly_chart(fig, width="stretch", height=600)

with st.container():
    st.subheader("Age Distribution")
    fig_age = px.histogram(df, x='Age', nbins=30,
    title='Patient Age Distribution', color_continuous_scale='teal'
    st.plotly_chart(fig_age, width="stretch", height=600)

with st.container():
    st.subheader("Gender Distribution")
    fig_gender = px.pie(df, names='Gender',
    title='Gender Distribution')
    st.plotly_chart(fig_gender, width="stretch", height=600)

# Page 2: Disease Analysis
st.header("🔬 Disease Analysis")

# Disease selector
diseases = sorted(df['Disease'].unique())
selected_disease = st.selectbox("Select Disease", diseases)

disease_df = df[df['Disease'] == selected_disease]

col1, col2, col3 = st.columns(3)
col1.metric("Total Cases", len(disease_df))
col2.metric("Avg Age", f"{disease_df['Age'].mean():.1f}")
col3.metric("Avg Symptoms", f"{disease_df['Symptom_Count'].mean():.2f}")

with st.container():
    st.subheader("Age Distribution for This Disease")
    fig = px.box(disease_df, y='Age', title=f'Age Distribution - {selected_disease}')
    st.plotly_chart(fig, width="stretch", height=600)

with st.container():
    st.subheader("Gender Distribution for This Disease")
    gender_dist = disease_df['Gender'].value_counts()
    fig = px.pie(values=gender_dist.values, names=gender_dist.index,
    title=f'Gender Distribution - {selected_disease}')
    st.plotly_chart(fig, width="stretch", height=600)

# Most common symptoms for this disease
st.subheader(f"Most Common Symptoms for {selected_disease}")
all_symptoms_disease = []
for symptoms in disease_df['Symptoms']:
    all_symptoms_disease.extend([s.strip() for s in symptoms.split(',')])

symptom_counts = pd.Series(all_symptoms_disease).value_counts().head(10)
fig = px.bar(symptom_counts, orientation='h',
labels={'value': 'Frequency', 'index': 'Symptom'},
title=f'Top 10 Symptoms for {selected_disease}')
st.plotly_chart(fig, width="stretch", height=600)

# Page 3: Symptom Analysis

st.header("💊 Symptom Analysis")

# Extract all symptoms
all_symptoms = []
for symptoms in df['Symptoms']:
    all_symptoms.extend([s.strip() for s in symptoms.split(',')])

symptom_counts = pd.Series(all_symptoms).value_counts()

st.subheader("Most Common Symptoms Across All Patients")
fig = px.bar(symptom_counts.head(20), orientation='h',
labels={'value': 'Frequency', 'index': 'Symptom'},
title='Top 20 Most Common Symptoms')
st.plotly_chart(fig, width="stretch", height=600)

# Symptom count distribution
st.subheader("Symptom Count Distribution")

with st.container():
    fig = px.histogram(df, x='Symptom_Count',
    title='Distribution of Number of Symptoms per Patient',
    labels={'Symptom_Count': 'Number of Symptoms'})
    st.plotly_chart(fig, width="stretch", height=600)

with st.container():
    # Relationship between age and symptom count
    fig = px.scatter(df.sample(1000), x='Age', y='Symptom_Count',
    color='Gender', opacity=0.5,
    title='Age vs Symptom Count (Sample)',
    trendline='lowess')
    st.plotly_chart(fig, width="stretch", height=600)


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