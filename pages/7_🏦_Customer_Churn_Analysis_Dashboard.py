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
    fig4.update_traces(texttemplate='%{text} customers', textposition='inside')
    st.plotly_chart(fig4, width="stretch")
        
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
    fig5.update_layout(height=500)
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
    fig6.update_layout(height=500)
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
fig8 = px.imshow(
    df_corr.round(2),
    text_auto=True,
    aspect="auto",
    title="Feature Correlation Matrix",
    color_continuous_scale="Viridis")
fig8.update_layout(height=800)
st.plotly_chart(fig8, width="stretch")

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
    fig9 = px.imshow(
        cm,
        labels=dict(x='Predicted', y='Actual', color='Count'),
        x=['Retained', 'Churned'],
        y=['Retained', 'Churned'],
        title='Confusion Matrix',
        color_continuous_scale='Blues',
        text_auto=True)
    st.plotly_chart(fig9, width='stretch')
    st.markdown(
    Components.metric_card(
        title="ROC-AUC Score",
        value=f"{auc_score:.4f}",
        delta="📈",
        card_type="info"
    ), unsafe_allow_html=True
)
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
    st.markdown(
    Components.metric_card(
        title="Top Feature",
        value=f"Age",
        delta="📈",
        card_type="info"
    ), unsafe_allow_html=True
)

st.markdown("---")
st.subheader("🔮 Predict Individual Customer Churn")

st.markdown("Enter customer details to predict churn probability:")

pred_col1, pred_col2, pred_col3 = st.columns(3)

with pred_col1:
    pred_credit = st.number_input("Credit Score", 300, 850, 650)
    pred_age = st.number_input("Age", 18, 100, 40)
    pred_tenure = st.number_input("Tenure (years)", 0, 10, 5)
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
        'Tenure': [pred_tenure],
        'Balance': [pred_balance],
        'NumOfProducts': [pred_products],
        'HasCrCard': [1 if pred_card == 'Yes' else 0],
        'IsActiveMember': [1 if pred_active == 'Yes' else 0],
        'EstimatedSalary': [pred_salary],
        'Gender': [1 if pred_gender == 'Male' else 0],
        'Geography_Germany': [1 if pred_geography == "Germany" else 0],
        'Geography_Spain': [1 if pred_geography == 'Spain' else 0]
    })


    input_data = input_data.reindex(columns=feature_cols, fill_value=0)


    # Make prediction
    churn_prob = float(model.predict_proba(input_data)[0][1])
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

# Gauge chart
if st.session_state.churn_prob is not None:
    fig_gauge = go.Figure(go.Indicator(
        mode='gauge+number',
        value=st.session_state.churn_prob * 100,
        title={'text': "Churn Risk Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': 'darkred' if st.session_state.churn_prob > 0.5 else 'green'},
            'steps': [
                {'range': [0, 30], 'color': 'lightgreen'},
                {'range': [30, 70], 'color': 'yellow'},
                {'range': [70, 100], 'color': 'lightcoral'}
            ],
            'threshold': {
            'line': {'color': 'red', 'width': 4},
            'thickness': 0.75,
            'value': 50
            }
        }
    ))
    st.plotly_chart(fig_gauge, width="stretch")

st.markdown(
        Components.page_header("👥 Customer Segmentation"), unsafe_allow_html=True
    )

from sklearn.cluster import KMeans  
from sklearn.preprocessing import StandardScaler  
from sklearn.decomposition import PCA  

@st.cache_data
def perform_clustering(data, n_clusters=4):
    cluster_features = ['Age', 'Balance', 'CreditScore', 'NumOfProducts', 'Tenure']
    X_cluster = data[cluster_features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    return clusters, X_pca, cluster_features

n_clusters = st.slider("Select Number of Clusters", 2, 8, 4)
clusters, X_pca, cluster_features = perform_clustering(df_filtered, n_clusters)

df_filtered['Cluster'] = clusters

# Cluster visualization with PCA
st.subheader("Customer Segments Visualization")
df_pca = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'Cluster': clusters,
    'Exited': df_filtered['Exited'].values
})

with st.container():
    fig11 = px.scatter(
        df_pca,
        x='PC1',
        y='PC2',
        color='Cluster',
        symbol='Exited',
        title='Customer Segments (PCA Projection)',
        labels={'Exited': 'Churned'},
        color_continuous_scale='Viridis')
    st.plotly_chart(fig11, width="stretch")

with st.container():
    # Cluster profiles
    st.subheader("Cluster Profiles")
    cluster_summary = df_filtered.groupby('Cluster').agg({
        'Age': 'mean',
        'Balance': 'mean',
        'CreditScore': 'mean',
        'Tenure': 'mean',
        'EstimatedSalary': 'mean',
        'Exited': 'mean',
        'Exited': 'sum'
    }).round(2)

cluster_summary.columns = ['Avg Age', 'Avg Balance', 'Avg Credit', 'Avg Products', 'Avg Tenure', 'Avg Salary', 'Churn Rate', 'Customer Count']
cluster_summary['Churn Rate'] = (cluster_summary['Churn Rate'] * 100).round(2)

with st.container():
    st.dataframe(cluster_summary, width="stretch")

# Churn rate by cluster
col1, col2 = st.columns(2)

with col1:
    cluster_churn = df_filtered.groupby('Cluster')['Exited'].mean().reset_index()
    cluster_churn['Exited'] = cluster_churn['Exited'] * 100
    fig12 = px.bar(
        cluster_churn,
        x='Cluster',
        y='Exited',
        title='Churn Rate by Customer Segment',
        labels={'Exited': 'Churn Rate (%)'},
        color='Exited',
        color_continuous_scale='Reds')
    st.plotly_chart(fig12, width="stretch")

with col2:
    cluster_size = df_filtered['Cluster'].value_counts().reset_index()
    cluster_size.columns = ['Cluster', 'Count']
    fig13 = px.pie(
        cluster_size,
        values='Count',
        names='Cluster',
        title='Customer Distribution Across Segments',
        color_continuous_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig13, width="stretch")