import streamlit as st 
import pandas as pd 
import numpy as np 
import plotly.express as px 
import plotly.graph_objects as go 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split  
from sklearn.metrics import roc_auc_score, confusion_matrix  
import seaborn as sns  
import matplotlib.pyplot as plt  

from utils.theme import Components, Colors, apply_chart_theme, init_page

init_page("Customer Churn Analytics Dashboard", "🏦")

# Load custom CSS
try:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Custom CSS file not found. Using default styling.")

st.markdown(
    Components.page_header(
        "🏦 Customer Churn Analytics Dashboard"
    ), unsafe_allow_html=True
)
st.markdown("""
<div style='text-align: center; font-size: 1.8rem'>
    <p><strong>Comprehensive analysis of customer churn patterns and predictions</strong></p>
</div>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('Churn_Modelling.csv')
    return df

df = load_data()
# Sidebar filters
st.sidebar.header("🔍 Filters")
selected_geography = st.sidebar.multiselect(
    'Select Geography',
    options=df['Geography'].unique(),
    default=df['Geography'].unique()
)

selected_gender = st.sidebar.multiselect(
    'Select Gender',
    options=df['Gender'].unique(),
    default=df['Gender'].unique()
)

age_range = st.sidebar.slider(
    'Age Range',
    min_value=int(df['Age'].min()),
    max_value=int(df['Age'].max()),
    value=(int(df['Age'].min()), int(df['Age'].max()))
)

# Filter data
df_filtered = df[
    (df['Geography'].isin(selected_geography)) &
    (df['Gender'].isin(selected_gender)) &
    (df['Age'] >= age_range[0]) &
    (df['Age'] <= age_range[1])
]

st.subheader("📈 Key Performance Indicators")


st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_customers = len(df_filtered)
    st.markdown(
        Components.metric_card(
            title="Total Customers",
            value=f"{total_customers:,}",
            delta="👥",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col2:
    churn_rate = df_filtered['Exited'].mean() * 100
    st.markdown(
        Components.metric_card(
            title="Churn Rate",
            value=f"{churn_rate:.2f}%",
            delta="🚷",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col3:
    avg_balance = df_filtered['Balance'].mean()
    st.markdown(
        Components.metric_card(
            title="Avg Balance",
            value=f"${avg_balance:,.0f}",
            delta="💲",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col4:
    avg_tenure = df_filtered['Tenure'].mean()
    st.markdown(
        Components.metric_card(
            title="Avg Tenure",
            value=f"{avg_tenure:.1f} years",
            delta="🚩",
            card_type="info"
        ), unsafe_allow_html=True
    )

st.markdown("---")
# Tab navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🔍 Deep Dive",
    "🤖 Predictions",
    "👥 Segmentation",
    "💡 Insights"
])

# Tab 1: Overview
with tab1:
    st.markdown(
        Components.page_header("📊 Overview Analytics"), unsafe_allow_html=True
    )

with st.container():
    # Churn by Geography
    churn_geo = df_filtered.groupby('Geography')['Exited'].agg([
        'mean', 'count']).reset_index()
    churn_geo['mean'] = churn_geo['mean'] * 100
    fig1 = px.bar(
            churn_geo,
            x='Geography',
            y='mean',
            title='Churn Rate by Geography',
            labels={'mean': 'Churn Rate (%)'},
            text='count',
            color='mean',
            color_continuous_scale='Reds')

    fig1.update_traces(texttemplate='%{text} customers', textposition='inside')
    fig1.update_layout(height=500)
    st.plotly_chart(fig1, width="stretch")
st.markdown("---")
with st.container():
    # Churn by Gender
    churn_gender = df_filtered.groupby('Gender')['Exited'].agg([
        'mean', 'count']).reset_index()
    churn_gender['mean'] = churn_gender['mean'] * 100
    fig2 = px.bar(
            churn_gender,
            x='Gender',
            y='mean',
            title='Churn Rate by Gender',
            labels={'mean': 'Churn Rate (%)'},
            text='count',
            color='mean',
            color_continuous_scale='Blues')
    fig2.update_traces(texttemplate='%{text} customers', textposition='inside')
    st.plotly_chart(fig2, width="stretch")
st.markdown("---")
with st.container():
    # Age distribution by churn
    fig3 = px.histogram(
        df_filtered,
        x='Age',
        color='Exited',
        title='Age Distribution by Churn Status',
        barmode='overlay',
        opacity=0.7,
        labels={'Exited': 'Churned'},
        color_discrete_map={0: 'green', 1: 'red'})
    st.plotly_chart(fig3, width="stretch")
st.markdown("---")
with st.container():
    # Number of products vs churn
    churn_products = df_filtered.groupby('NumOfProducts')['Exited'].agg([
        'mean', 'count']).reset_index()
    churn_products['mean'] = churn_products['mean'] * 100
    fig4 = px.bar(
            churn_products,
            x='NumOfProducts',
            y='mean',
            title='Churn Rate by Number of Products',
            labels={'mean': 'Churn Rate (%)'},
            text='count',
            color='mean',
            color_continuous_scale='OrRd')
    fig4.update_traces(texttemplate='%{text} customers', textposition='outside')
    st.plotly_chart(fig4, width="stretch")
        
# Tab 2: Deep Dive
with tab2:
       st.markdown(
        Components.page_header("🔍 Deep Dive"), unsafe_allow_html=True
    )

with st.container():
    # Balance vs Age scatter
    fig5 = px.scatter(
        df_filtered,
        x='Age',
        y='Balance',
        color='Exited',
        title='Customer Balance vs Age',
        labels={'Exited': 'Churned'},
        color_discrete_map={0: 'blue', 1: 'red'},
        opacity=0.6,
        hover_data=['CreditScore', 'Tenure'])
    st.plotly_chart(fig5, width='stretch')
st.markdown("---")
with st.container():
    # Credit score distribution
    fig6 = px.box(
        df_filtered,
        x='Exited',
        y='CreditScore',
        title='Credit Score Distribution by Churn Status',
        labels={'Exited': 'Churned'},
        color='Exited',
        color_discrete_map={0: 'green', 1: 'red'})
    st.plotly_chart(fig6, width='stretch')

st.markdown("---")
# Active vs Inactive members
st.subheader("Activity Status Impact") 

activity_churn = df_filtered.groupby([
    'IsActiveMember', 'Exited']).size().reset_index(name='count')
activity_churn['IsActiveMember'] = activity_churn['IsActiveMember'].map({0: 'Inactive', 1: 'Active'})
activity_churn['Exited'] = activity_churn['Exited'].map({0: 'Retained', 1: 'Churned'})

fig7 = px.bar(
    activity_churn,
    x='IsActiveMember',
    y='count',
    color='Exited',
    barmode='group',
    title='Customer Retention by Activity Status',
    color_discrete_map={'Retained': 'green', 'Churned': 'red'})
st.plotly_chart(fig7, width='stretch')

st.markdown("---")
# Correlation heatmap
st.subheader("Feature Correlations")  
df_corr = df_filtered.select_dtypes(include=[np.number]).corr()
fig8, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df_corr, annot=True, cmap='coolwarm', center=0, ax=ax, fmt='.2f')
plt.title('Feature Correlation Matrix')
st.pyplot(fig8)

# Tab 3: Predictions
with tab3:
    st.markdown(
        Components.page_header("🤖 Churn Prediction Model"), unsafe_allow_html=True
    )
# Prepare data for modeling
@st.cache_resource
def train_model(data):
    df_model = data.copy()
    df_model = df_model.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    df_model = pd.get_dummies(df_model, columns=['Geography'], drop_first=True)
    df_model['Gender'] = df_model['Gender'].map({'Male': 1, 'Female': 0})
    X = df_model.drop('Exited', axis=1)
    y = df_model['Exited']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)

    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    return model, auc_score, cm, feature_importance, X.columns
model, auc_score, cm, feature_importance, feature_cols = train_model(df)

st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Model Performance")
    st.markdown(
        Components.metric_card(
            title="ROC-AUC Score",
            value=f"{auc_score:.4f}",
            delta="📈",
            card_type="info"
        ), unsafe_allow_html=True
    )
    fig9 = px.imshow(
        cm,
        labels=dict(x='Predicted', y='Actual', color='Count'),
        x=['Retained', 'Churned'],
        y=['Retained', 'Churned'],
        title='Confusion Matrix',
        color_continuous_scale='Blues',
        text_auto=True)
    st.plotly_chart(fig9, width='stretch')
with col2:
    st.subheader("Top Features")
    fig10 = px.bar(
        feature_importance.head(10),
        x='Importance',
        y='Feature',
        orientation='h',
        title='Top 10 Most Important Features',
        color='Importance',
        color_continuous_scale='Viridis')
    st.plotly_chart(fig10, width='stretch')

st.markdown("---")
st.subheader("🔮 Predict Individual Customer Churn")

st.markdown("Enter customer details to predict churn probability:")

pred_col1, pred_col2, pred_col3 = st.columns(3)

with pred_col1:
    pred_credit = st.number_input("Credit Score", 300, 850, 650)
    pred_age = st.number_input("Age", 18, 100, 40)
    pred_ternure = st.number_input("Tenure (years)", 0, 10, 5)
    pred_balance = st.number_input("Balance", 0, 250000, 100000)

with pred_col2:
    pred_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    pred_card = st.selectbox("Has Credit Card", ['Yes', 'No'])
    pred_active = st.selectbox("Is Active Member", ['Yes', 'No'])
    pred_salary = st.number_input("Estimated Salary", 0, 200000, 100000)

with pred_col3:
    pred_geography = st.selectbox("Geography", ['France', 'Germany', 'Spain'])
    pred_gender = st.selectbox("Gender", ['Male', 'Female'])

# Initialize state variables
if "churn_prob" not in st.session_state:
    st.session_state.churn_prob = None
if "churn_prediction" not in st.session_state:
    st.session_state.churn_prediction = None

if st.button ("🎯 Predict Churn Probability"):
    # Prepare input data
    input_data = pd.DataFrame({
        'CreditScore': [pred_credit],
        'Age': [pred_age],
        'Tenure': [pred_ternure],
        'Balance': [pred_balance],
        'NumOfProducts': [pred_products],
        'HasCrCard': [1 if pred_card == 'Yes' else 0],
        'IsActiveMember': [1 if pred_active == 'Yes' else 0],
        'EstimatedSalary': [pred_salary],
        'Gender': [1 if pred_gender == 'Male' else 0],
        'Geography_Germany': [1 if pred_geography == "Germany" else 0],
        'Geography_Spain': [1 if pred_geography == 'Spain' else 0]
    })

    # Ensure all required columns are present
    for col in feature_cols:
        if col not in input_data.columns:
            input_data[col] = 0
            input_data = input_data[feature_cols]


    # Make prediction
    churn_prob = model.predict_proba(input_data)[0][1]
    churn_prediction = "HIGH RISK ⚠️" if churn_prob > 0.5 else "LOW RISK ✅"
    # Save results in session state
    st.session_state.churn_prob = churn_prob
    st.session_state.churn_prediction = churn_prediction


st.markdown("---")  
result_col1, result_col2 = st.columns(2)

with result_col1:
    st.markdown(
        Components.metric_card(
            title="Churn Probability",
            value=f"{st.session_state.churn_prob:.2%}" if st.session_state.churn_prob is not None else "--",
            delta="Updated" if st.session_state.churn_prob else "",
            card_type="info"
        ), unsafe_allow_html=True
    )
with result_col2:
    st.markdown(
        Components.metric_card(
            title="Risk Level",
            value=st.session_state.churn_prediction if st.session_state.churn_prediction else "--",
            delta="Updated" if st.session_state.churn_prediction else "",
            card_type=( "warning" if st.session_state.churn_prediction == "HIGH RISK ⚠️" else "success" if st.session_state.churn_prediction == "LOW RISK ✅" else "info")
        ), unsafe_allow_html=True
    )